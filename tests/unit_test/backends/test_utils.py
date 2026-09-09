import sys
from types import ModuleType, SimpleNamespace

from llumnix.backends.backend_interface import BackendType
from llumnix.backends.utils import get_engine_world_size, rebuild_engine_args_for_runtime


def test_v1_world_size_does_not_construct_engine_config_in_manager():
    """Placement planning runs in a CPU-only Manager actor on V1."""
    args = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=3)
    assert get_engine_world_size(args, BackendType.VLLM) == 6


def test_rebuild_engine_args_for_runtime_adapts_serialized_legacy_args(monkeypatch):
    class RuntimeAsyncEngineArgs:
        def __init__(self, model, tensor_parallel_size=1, dtype=None):
            self.model = model
            self.tensor_parallel_size = tensor_parallel_size
            self.dtype = dtype

    vllm_module = ModuleType("vllm")
    engine_module = ModuleType("vllm.engine")
    arg_utils_module = ModuleType("vllm.engine.arg_utils")
    arg_utils_module.AsyncEngineArgs = RuntimeAsyncEngineArgs
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_module)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_module)

    serialized_args = SimpleNamespace(
        model="/models/test",
        tensor_parallel_size=2,
        dtype="bfloat16",
        legacy_only_field="must be dropped",
    )

    rebuilt = rebuild_engine_args_for_runtime(serialized_args)

    assert isinstance(rebuilt, RuntimeAsyncEngineArgs)
    assert rebuilt.model == "/models/test"
    assert rebuilt.tensor_parallel_size == 2
    assert rebuilt.dtype == "bfloat16"
    assert not hasattr(rebuilt, "legacy_only_field")


def test_rebuild_engine_args_for_runtime_keeps_runtime_args():
    runtime_args = SimpleNamespace(model_class_overrides={})

    assert rebuild_engine_args_for_runtime(runtime_args) is runtime_args
