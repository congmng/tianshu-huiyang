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
