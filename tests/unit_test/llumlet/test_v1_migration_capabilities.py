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

from types import SimpleNamespace
from unittest.mock import patch

from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.llumlet.llumlet import Llumlet


def test_instance_info_defaults_to_no_migration_capabilities():
    """Legacy/simulator instances must publish an empty capability set."""
    info = InstanceInfo(instance_id="legacy")
    assert info.migration_capabilities == frozenset()


def test_instance_info_carries_explicit_capability_contract():
    info = InstanceInfo(
        instance_id="v1",
        migration_capabilities=frozenset({"token_boundary_freeze", "native_nccl"}),
    )
    assert "token_boundary_freeze" in info.migration_capabilities
    assert "native_nccl" in info.migration_capabilities
    assert "legacy_block_manager" not in info.migration_capabilities


def test_llumlet_legacy_returns_empty_migration_capabilities():
    llumlet = object.__new__(Llumlet)
    llumlet.is_vllm_v1 = False
    assert llumlet.migration_capabilities() == frozenset()


def test_llumlet_v1_returns_backend_migration_capabilities():
    llumlet = object.__new__(Llumlet)
    llumlet.is_vllm_v1 = True
    llumlet.backend_engine = SimpleNamespace(
        migration_capabilities=lambda: frozenset({"kv_snapshot", "seeded_rng"})
    )
    capabilities = llumlet.migration_capabilities()
    assert capabilities == frozenset({"kv_snapshot", "seeded_rng"})


def _make_v1_llumlet():
    llumlet = object.__new__(Llumlet)
    llumlet.is_vllm_v1 = True
    llumlet.instance_id = "instance-v1"
    llumlet.node_id = "node-1"
    llumlet.instance_args = SimpleNamespace(instance_type=InstanceType.NO_CONSTRAINTS)
    llumlet.instance_load_calculator = SimpleNamespace(
        compute_instance_load=lambda info: None
    )
    llumlet.backend_engine = SimpleNamespace(
        update_instance_info=lambda info: setattr(info, "num_running_requests", 0),
        get_kv_endpoint=lambda host: "10.0.0.8:19000",
        migration_capabilities=lambda: frozenset({"token_boundary_freeze"}),
    )
    return llumlet


def test_llumlet_get_instance_info_publishes_v1_capabilities():
    llumlet = _make_v1_llumlet()
    with patch(
        "llumnix.llumlet.llumlet.ray.get_runtime_context",
        return_value=SimpleNamespace(get_node_ip_address=lambda: "10.0.0.8"),
    ):
        info = llumlet.get_instance_info()
    assert info.migration_capabilities == frozenset({"token_boundary_freeze"})
    assert info.kv_endpoint == "10.0.0.8:19000"


def test_llumlet_get_instance_info_handles_missing_ray_address():
    llumlet = _make_v1_llumlet()

    class MissingRuntimeContext:
        def get_node_ip_address(self):
            raise AttributeError("no address")

    with patch(
        "llumnix.llumlet.llumlet.ray.get_runtime_context",
        return_value=MissingRuntimeContext(),
    ):
        info = llumlet.get_instance_info()
    assert info.migration_capabilities == frozenset({"token_boundary_freeze"})
