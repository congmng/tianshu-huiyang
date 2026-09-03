from types import SimpleNamespace

from llumnix.backends.backend_interface import BackendType
from llumnix.backends.utils import get_engine_world_size


def test_v1_world_size_does_not_construct_engine_config_in_manager():
    """Placement planning runs in a CPU-only Manager actor on V1."""
    args = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=3)
    assert get_engine_world_size(args, BackendType.VLLM) == 6
