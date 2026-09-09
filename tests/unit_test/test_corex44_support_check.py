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


def test_corex45_gate_accepts_v300_stack():
    gate = _load_gate()
    assert gate.validate_versions({
        "python": "3.12.13", "vllm": "0.25.1",
        "torch": "2.10.0", "ray": "2.56.1",
    }, "45") == []
    assert gate.validate_corex_runtime({
        "corex_sdk": "Iluvatar CoreX SDK 4.5.0", "cuda_available": True,
        "device_name": "Iluvatar TG-V300 OAM",
    }, "45") == []


def test_corex45_gate_rejects_v150_device_on_v300_stack():
    gate = _load_gate()
    errors = gate.validate_corex_runtime({
        "corex_sdk": "Iluvatar CoreX SDK 4.5.0", "cuda_available": True,
        "device_name": "Iluvatar BI-V150",
    }, "45")
    assert any("device" in error for error in errors)


def test_mixed_stack_gate_compares_code_and_protocol_not_vendor_versions():
    gate = _load_gate()
    local = {"python": "3.12.13", "vllm": "0.11.2", "ray": "2.52.1",
             "torch": "2.7.1", "corex_sdk": "Iluvatar CoreX SDK 4.4.0",
             "device_name": "Iluvatar BI-V150", "affinity_hashes": ["a"],
             "source_fingerprint": "same", "migration_protocol_version": 1,
             "supported": True}
    remote = dict(local)
    remote.update({"torch": "2.10.0", "corex_sdk": "Iluvatar CoreX SDK 4.5.0",
                   "device_name": "Iluvatar TG-V300 OAM"})
    assert gate.compare_hosts(local, remote, mixed_stack=True) == []
    remote.update({
        "vllm": "0.25.1",
        "ray": "2.56.1",
        "migration_protocol_version": 0,
    })
    assert gate.compare_hosts(local, remote, mixed_stack=True) == []
    remote["source_fingerprint"] = "different"
    assert gate.compare_hosts(local, remote, mixed_stack=True)


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
        "llumnix/backends/vllm/v1_migration.py",
        "llumnix/global_scheduler/dispatch_scheduler.py",
        "llumnix/global_scheduler/global_scheduler.py", "llumnix/manager.py",
        "llumnix/global_scheduler/scaling_scheduler.py",
        "llumnix/global_scheduler/scaling_policy.py",
        "llumnix/launcher.py", "llumnix/llumlet/llumlet.py",
        "llumnix/entrypoints/vllm/arg_utils.py", "llumnix/entrypoints/vllm/client.py",
        "llumnix/entrypoints/vllm/v1_api_server.py", "llumnix/instance_info.py",
        "tools/run_corex44_validation.py", "tools/run_llumnix_v1_http_e2e.py",
        "tools/v1_p2p_model_probe.py",
        "tools/corex44_native_nccl_probe.py",
        "tools/corex_env.sh",
        "tools/corex44_env.sh",
        "tools/corex45_env.sh",
        "configs/corex44_v1_pd.yml",
        "docs/vLLM_V1_True_KV_Migration_Plan.md",
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
    assert "--native-nccl" in source
    assert 'default="nccl"' in source
    assert "--corex-transport" in source
    assert "v1_p2p_model_probe.py" in source
    assert "--local-ip" in source
    assert "--remote-ip" in source
    assert "exited during startup" in source


def test_remote_stack_is_selected_before_sourcing_corex_env():
    """Remote 45 gate must not source corex_env.sh while LLUMNIX_COREX_STACK
    is still unset; that would make the script search for a 4.4 Python env.
    """
    script = Path(__file__).parents[2] / "tools" / "corex44_support_check.py"
    source = script.read_text(encoding="utf-8")
    assert "LLUMNIX_COREX_STACK={remote_stack} " in source
    assert "source tools/corex_env.sh" in source
    assert source.index("LLUMNIX_COREX_STACK={remote_stack} ") < source.index(
        "source tools/corex_env.sh"
    )


def test_remote_validation_commands_select_stack_before_sourcing():
    script = Path(__file__).parents[2] / "tools" / "run_corex44_validation.py"
    source = script.read_text(encoding="utf-8")
    assert source.count("source tools/corex_env.sh") == 3
    for needle in ("LLUMNIX_COREX_STACK={remote_stack} ", "source tools/corex_env.sh"):
        assert needle in source


def test_corex45_docker_gate_does_not_claim_ported_migration_protocol():
    """The 4.5 Docker image runs vLLM 0.23 and has no 0.11-fork migration
    module. The runtime gate must be explicit about that boundary instead of
    silently accepting a false migration-protocol match.
    """
    script = Path(__file__).parents[2] / "tools" / "corex45_docker_support_check.py"
    source = script.read_text(encoding="utf-8")
    assert '"unported-vllm-0.23"' in source
    assert "0.23." in source
    assert "Iluvatar TG-V300" in source
    assert "Docker runtime support gate" in source


def test_corex45_docker_gate_is_used_by_mixed_stack_integration():
    script = Path(__file__).parents[2] / "tools" / "run_corex44_validation.py"
    source = script.read_text(encoding="utf-8")
    assert "corex45_docker_support_check.py" in source
    assert 'remote_stack == "45"' in source
    assert "vLLM 0.23 migration adapter" in source
