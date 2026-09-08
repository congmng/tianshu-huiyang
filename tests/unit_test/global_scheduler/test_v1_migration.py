# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llumnix.instance_info import InstanceInfo
from llumnix.manager import Manager
from llumnix.global_scheduler.migration_policy import PairMigrationConstraints
from llumnix.global_scheduler.v1_migration_scheduler import (
    V1MigrationScheduler,
    v1_migration_capable,
)


def test_v1_migration_capable_requires_base_contract():
    info = InstanceInfo(
        instance_id="v1",
        migration_capabilities=frozenset({
            "token_boundary_freeze", "kv_snapshot", "native_nccl",
        }),
    )
    assert v1_migration_capable(info)
    legacy = InstanceInfo(instance_id="legacy")
    assert not v1_migration_capable(legacy)


def test_v1_migration_scheduler_filters_non_capable_instances():
    scheduler = V1MigrationScheduler("defrag", 0.2)
    infos = {
        "src": InstanceInfo(
            instance_id="src",
            migration_load_metric=0.9,
            migration_capabilities=frozenset({
                "token_boundary_freeze", "kv_snapshot", "native_nccl",
            }),
        ),
        "src-legacy": InstanceInfo(
            instance_id="src-legacy",
            migration_load_metric=1.0,
        ),
        "dst": InstanceInfo(
            instance_id="dst",
            migration_load_metric=0.1,
            migration_capabilities=frozenset({
                "token_boundary_freeze", "kv_snapshot", "native_nccl",
            }),
        ),
        "dst-legacy": InstanceInfo(
            instance_id="dst-legacy",
            migration_load_metric=-0.1,
        ),
    }
    scheduler.update_instance_infos(infos)
    pairs = scheduler.pair_migration(PairMigrationConstraints.NO_CONSTRAINTS)
    assert pairs == [("src", "dst")]


def _remote(value=None, side_effect=None):
    return SimpleNamespace(remote=AsyncMock(
        return_value=value, side_effect=side_effect
    ))


@pytest.mark.asyncio
async def test_manager_selects_source_owned_request_for_v1_migration():
    manager = object.__new__(Manager)
    manager.v1_migrating_requests = set()
    manager.v1_request_last_migration_time = {}
    manager.request_instance = {"active": "src"}
    manager.instances = {
        "src": SimpleNamespace(get_all_request_ids=_remote(["other", "active"])),
    }
    request_id = await manager._select_v1_migration_request("src")
    assert request_id == "active"


@pytest.mark.asyncio
async def test_manager_selects_v1_request_respects_residency_cooldown():
    manager = object.__new__(Manager)
    manager.v1_migrating_requests = set()
    manager.v1_request_last_migration_time = {"active": time.monotonic()}
    manager.request_instance = {"active": "src"}
    manager.instances = {
        "src": SimpleNamespace(get_all_request_ids=_remote(["active"])),
    }
    assert await manager._select_v1_migration_request("src") is None


@pytest.mark.asyncio
async def test_manager_migrate_v1_pair_skips_full_target():
    manager = object.__new__(Manager)
    manager.instance_migrating = {}
    manager.global_scheduler = SimpleNamespace(
        instance_info={
            "src": InstanceInfo(
                instance_id="src", max_num_seqs=1, num_running_requests=1,
                kv_endpoint="10.0.0.1:19000",
                migration_capabilities=frozenset({
                    "token_boundary_freeze", "kv_snapshot", "native_nccl",
                }),
            ),
            "dst": InstanceInfo(
                instance_id="dst", max_num_seqs=1, num_running_requests=1,
                kv_endpoint="10.0.0.2:19000",
                migration_capabilities=frozenset({
                    "token_boundary_freeze", "kv_snapshot", "native_nccl",
                }),
            ),
        }
    )
    manager.instances = {"src": object(), "dst": object()}
    await manager._migrate_v1_pair("src", "dst")
    assert manager.instance_migrating["src"] is False
    assert manager.instance_migrating["dst"] is False


@pytest.mark.asyncio
async def test_manager_migrate_v1_request_preserves_two_phase_order():
    manager = object.__new__(Manager)
    src = SimpleNamespace(
        migration_prepare_out_wire=_remote("snapshot-wire"),
        migration_source_blocks=_remote([[1, 2]]),
        migration_layer_names=_remote(["layer-0"]),
        migration_send_layer=_remote("manifest-wire"),
        migration_commit=_remote(),
        finish_migrated_out=_remote(),
    )
    dst = SimpleNamespace(
        migration_prepare_in_wire=_remote([[10, 20]]),
        migration_receive_layer=_remote(),
        migration_commit=_remote(),
        register_migrated_request=_remote(),
    )
    manager.instances = {"src": src, "dst": dst}
    await manager._migrate_v1_request(
        "src", "dst", "request-1", 1,
        "10.0.0.1:19000", "10.0.0.2:19000", None,
    )
    src.migration_prepare_out_wire.remote.assert_awaited_once_with("request-1", 1)
    dst.migration_prepare_in_wire.remote.assert_awaited_once_with("snapshot-wire")
    src.migration_send_layer.remote.assert_awaited_once_with(
        "request-1", 1, "layer-0", [1, 2], [10, 20], "10.0.0.2:19000", 0,
    )
    dst.migration_receive_layer.remote.assert_awaited_once_with(
        "request-1", 1, "manifest-wire", "10.0.0.1:19000", (),
    )
    dst_commit = dst.migration_commit.remote.await_args
    src_commit = src.migration_commit.remote.await_args
    assert dst_commit.args == ("request-1", 1)
    assert dst_commit.kwargs["incoming"] is True
    assert src_commit.args == ("request-1", 1)
    assert src_commit.kwargs["incoming"] is False


@pytest.mark.asyncio
async def test_manager_migrate_v1_request_runs_incremental_precopy():
    manager = object.__new__(Manager)
    src = SimpleNamespace(
        incremental_begin_wire=_remote("begin"),
        incremental_immutable_blocks=_remote(([[1, 2]], (2,))),
        incremental_preview_wire=_remote("preview"),
        incremental_send_layer_wire=_remote("manifest"),
        incremental_append_wire=_remote("append"),
        incremental_abort=_remote(),
        migration_prepare_out_wire=_remote("snapshot-wire"),
        migration_source_blocks=_remote([[1, 2, 3]]),
        migration_layer_names=_remote(["layer-0"]),
        migration_send_layer=_remote("manifest-wire"),
        migration_commit=_remote(),
        finish_migrated_out=_remote(),
    )
    dst = SimpleNamespace(
        incremental_prepare_in_wire=_remote([[10, 20]]),
        incremental_receive_layer=_remote(),
        incremental_commit_in_wire=_remote("committed"),
        incremental_abort=_remote(),
        migration_prepare_in_wire=_remote([[10, 20, 30]]),
        migration_receive_layer=_remote(),
        migration_commit=_remote(),
        register_migrated_request=_remote(),
    )
    manager.instances = {"src": src, "dst": dst}

    await manager._migrate_v1_request(
        "src", "dst", "request-1", 1,
        "10.0.0.1:19000", "10.0.0.2:19000", None,
        incremental_precopy=True,
    )

    src.incremental_immutable_blocks.remote.assert_awaited_once_with("request-1")
    dst.incremental_prepare_in_wire.remote.assert_awaited_once_with("begin", (2,))
    src.incremental_send_layer_wire.remote.assert_awaited_once_with(
        "preview", "layer-0", [1, 2], [10, 20], "10.0.0.2:19000",
    )
    dst.incremental_receive_layer.remote.assert_awaited_once_with(
        "preview", "manifest", "10.0.0.1:19000",
    )
    src.migration_send_layer.remote.assert_awaited_once_with(
        "request-1", 1, "layer-0", [1, 2, 3], [10, 20, 30],
        "10.0.0.2:19000", 2,
    )
    dst.migration_receive_layer.remote.assert_awaited_once_with(
        "request-1", 1, "manifest-wire", "10.0.0.1:19000",
        ((1, 10), (2, 20)),
    )
    src.incremental_abort.remote.assert_awaited_once_with("request-1")
    dst.incremental_abort.remote.assert_awaited_once_with("request-1")


@pytest.mark.asyncio
async def test_manager_migrate_v1_request_with_retry_retries_transient_failure():
    manager = object.__new__(Manager)
    manager.request_instance = {}
    manager.request_instances = {}
    manager.v1_migration_retries = {}
    manager.v1_request_last_migration_time = {}
    manager._migrate_v1_request = AsyncMock(
        side_effect=[RuntimeError("transient"), None]
    )
    manager._cleanup_v1_migration = AsyncMock()
    manager._check_instance_error = AsyncMock(return_value=[False, False])

    await manager._migrate_v1_request_with_retry(
        "src", "dst", "request-1", 1,
        "10.0.0.1:19000", "10.0.0.2:19000", None,
    )
    assert manager._migrate_v1_request.await_count == 2
    manager._cleanup_v1_migration.assert_awaited_once()
    assert manager.request_instance["request-1"] == "dst"
    assert manager.request_instances["request-1"] == {"dst"}
    assert "request-1" not in manager.v1_migration_retries


@pytest.mark.asyncio
async def test_manager_migrate_v1_request_skips_finished_request_without_cleanup():
    manager = object.__new__(Manager)
    manager.request_instance = {}
    manager.request_instances = {}
    manager.v1_migration_retries = {}
    manager.v1_request_last_migration_time = {}
    manager._migrate_v1_request = AsyncMock(
        side_effect=RuntimeError("request is not migratable: request-1")
    )
    manager._cleanup_v1_migration = AsyncMock()
    manager._check_instance_error = AsyncMock(return_value=[False, False])

    await manager._migrate_v1_request_with_retry(
        "src", "dst", "request-1", 1,
        "10.0.0.1:19000", "10.0.0.2:19000", None,
    )
    manager._migrate_v1_request.assert_awaited_once()
    manager._cleanup_v1_migration.assert_not_awaited()


@pytest.mark.asyncio
async def test_manager_migrate_v1_schedules_all_capable_pairs():
    """Cross-round scheduling must fan out to every capability-filtered pair."""
    manager = object.__new__(Manager)
    manager.global_scheduler = SimpleNamespace(
        pair_migration_v1=lambda _type: [("a", "b"), ("c", "d")]
    )
    manager.instance_migrating = {}
    manager._migrate_v1_pair = AsyncMock()

    await manager._migrate_v1(PairMigrationConstraints.NO_CONSTRAINTS)

    assert manager._migrate_v1_pair.await_count == 2
    calls = [call.args for call in manager._migrate_v1_pair.await_args_list]
    assert ("a", "b") in calls
    assert ("c", "d") in calls
