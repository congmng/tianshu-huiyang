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


def test_corex44_runtime_gate_requires_vendor_device_and_sdk():
    gate = _load_gate()
    assert gate.validate_corex_runtime({
        "corex_sdk": "Iluvatar CoreX SDK 4.4.0", "cuda_available": True,
        "device_name": "Iluvatar BI-V150",
    }) == []
    errors = gate.validate_corex_runtime({
        "corex_sdk": "CUDA toolkit", "cuda_available": False,
        "device_name": "NVIDIA A100",
    })
    assert len(errors) == 3


def test_corex44_gate_compares_two_hosts():
    gate = _load_gate()
    local = {"python": "3.12.13", "vllm": "0.11.2", "ray": "2.52.1",
             "torch": "2.7.1", "affinity_hashes": ["a"],
             "source_fingerprint": "same", "supported": True}
    assert gate.compare_hosts(local, dict(local)) == []
    remote = dict(local)
    remote["affinity_hashes"] = ["b"]
    assert gate.compare_hosts(local, remote)
    remote = dict(local)
    remote["source_fingerprint"] = "different"
    assert gate.compare_hosts(local, remote)


def test_corex44_source_fingerprint_is_sha256():
    gate = _load_gate()
    fingerprint = gate.source_fingerprint()
    assert len(fingerprint) == 64
    assert int(fingerprint, 16) >= 0


def test_corex44_source_fingerprint_covers_all_v1_serving_boundaries():
    gate = _load_gate()
    expected = {
        "llumnix/backends/utils.py", "llumnix/backends/vllm/v1_engine.py",
        "llumnix/backends/vllm/v1_kv.py", "llumnix/backends/vllm/v1_kv_transfer.py",
        "llumnix/backends/vllm/corex_p2p_connector.py",
        "llumnix/global_scheduler/dispatch_scheduler.py",
        "llumnix/global_scheduler/global_scheduler.py", "llumnix/manager.py",
        "llumnix/global_scheduler/scaling_scheduler.py",
        "llumnix/global_scheduler/scaling_policy.py",
        "llumnix/launcher.py", "llumnix/llumlet/llumlet.py",
        "llumnix/entrypoints/vllm/arg_utils.py", "llumnix/entrypoints/vllm/client.py",
        "llumnix/entrypoints/vllm/v1_api_server.py", "llumnix/instance_info.py",
        "tools/run_corex44_validation.py", "tools/run_llumnix_v1_http_e2e.py",
        "tools/v1_p2p_model_probe.py",
        "configs/corex44_v1_pd.yml",
    }
    assert expected <= set(gate.SOURCE_FINGERPRINT_FILES)


def test_qwen_smoke_exposes_multi_gpu_tensor_parallelism():
    script = Path(__file__).parents[2] / "tools" / "run_qwen3_14b_smoke.py"
    source = script.read_text(encoding="utf-8")
    assert 'TENSOR_PARALLEL_SIZE' in source
    assert 'tensor_parallel_size=TENSOR_PARALLEL_SIZE' in source


def test_layered_corex_validation_runner_has_all_required_levels():
    script = Path(__file__).parents[2] / "tools" / "run_corex44_validation.py"
    source = script.read_text(encoding="utf-8")
    for level in ('"unit"', '"integration"', '"e2e"'):
        assert level in source
    assert "corex44_zmq_kv_probe.py" in source
    assert "run_qwen3_14b_smoke.py" in source
    assert "run_llumnix_v1_http_e2e.py" in source
    assert "--model-pd" in source
    assert "v1_p2p_model_probe.py" in source
    assert "--local-ip" in source
    assert "--remote-ip" in source
    assert "exited during startup" in source
