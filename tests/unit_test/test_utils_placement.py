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
