from unittest.mock import patch

import pytest

from llumnix.utils import initialize_placement_group


def test_strict_pack_rejects_cross_node_tensor_parallel_topology():
    with patch("llumnix.utils.ray.cluster_resources", return_value={"GPU": 2}), \
         patch("llumnix.utils.ray.nodes", return_value=[
             {"Alive": True, "Resources": {"GPU": 1}},
             {"Alive": True, "Resources": {"GPU": 1}},
         ]):
        with pytest.raises(ValueError, match="STRICT_PACK placement needs 2 GPUs"):
            initialize_placement_group("tp2", num_cpus=2, num_gpus=2, block=False)


def test_v1_tensor_parallel_placement_packs_devices_for_parent_actor():
    with patch("llumnix.utils.ray.cluster_resources", return_value={"GPU": 2}), \
         patch("llumnix.utils.ray.nodes", return_value=[
             {"Alive": True, "Resources": {"GPU": 2}},
         ]), \
         patch("llumnix.utils.ray.util.placement_group") as placement_group:
        initialize_placement_group(
            "tp2", num_cpus=2, num_gpus=2, block=False,
            pack_gpus_in_first_bundle=True,
        )
    assert placement_group.call_args.args[0] == [{"CPU": 2, "GPU": 2}]
