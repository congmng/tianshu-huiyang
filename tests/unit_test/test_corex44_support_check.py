import importlib.util
from pathlib import Path


def _load_gate():
    path = Path(__file__).parents[2] / "tools" / "corex44_support_check.py"
    spec = importlib.util.spec_from_file_location("corex44_support_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corex44_gate_accepts_supported_stack():
    gate = _load_gate()
    assert gate.validate_versions({
        "python": "3.12.13", "vllm": "0.11.2",
        "torch": "2.7.1", "ray": "2.52.1",
    }) == []


def test_corex44_gate_rejects_unsupported_stack():
    gate = _load_gate()
    errors = gate.validate_versions({
        "python": "3.11.9", "vllm": "0.6.3",
        "torch": "2.6.0", "ray": "2.10.0",
    })
    assert len(errors) == 4


def test_corex44_gate_compares_two_hosts():
    gate = _load_gate()
    local = {"python": "3.12.13", "vllm": "0.11.2", "ray": "2.52.1",
             "torch": "2.7.1", "affinity_hashes": ["a"], "supported": True}
    assert gate.compare_hosts(local, dict(local)) == []
    remote = dict(local)
    remote["affinity_hashes"] = ["b"]
    assert gate.compare_hosts(local, remote)


def test_qwen_smoke_exposes_multi_gpu_tensor_parallelism():
    script = Path(__file__).parents[2] / "tools" / "run_qwen3_14b_smoke.py"
    source = script.read_text(encoding="utf-8")
    assert 'TENSOR_PARALLEL_SIZE' in source
    assert 'tensor_parallel_size=TENSOR_PARALLEL_SIZE' in source
