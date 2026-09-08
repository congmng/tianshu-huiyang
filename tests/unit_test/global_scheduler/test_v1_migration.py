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
    manager.request_instance = {"active": "src"}
    manager.instances = {
        "src": SimpleNamespace(get_all_request_ids=_remote(["other", "active"])),
    }
    request_id = await manager._select_v1_migration_request("src")
    assert request_id == "active"


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
        "request-1", 1, "layer-0", [1, 2], [10, 20], "10.0.0.2:19000",
    )
    dst.migration_receive_layer.remote.assert_awaited_once_with(
        "request-1", 1, "manifest-wire", "10.0.0.1:19000",
    )
    dst_commit = dst.migration_commit.remote.await_args
    src_commit = src.migration_commit.remote.await_args
    assert dst_commit.args == ("request-1", 1)
    assert dst_commit.kwargs["incoming"] is True
    assert src_commit.args == ("request-1", 1)
    assert src_commit.kwargs["incoming"] is False


@pytest.mark.asyncio
async def test_manager_migrate_v1_request_with_retry_retries_transient_failure():
    manager = object.__new__(Manager)
    manager.request_instance = {}
    manager.request_instances = {}
    manager.v1_migration_retries = {}
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
