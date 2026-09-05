from tests.e2e_test import utils


def test_e2e_commands_omit_removed_worker_flag_on_v1(monkeypatch):
    import vllm

    monkeypatch.setattr(vllm, "__version__", "0.11.2")
    command = utils.generate_launch_command(model="dummy", port=38000)
    assert "--worker-use-ray" not in command


def test_e2e_commands_keep_worker_flag_for_legacy_vllm(monkeypatch):
    import vllm

    monkeypatch.setattr(vllm, "__version__", "0.6.5")
    command = utils.generate_serve_command(model="dummy", port=38001)
    assert "--worker-use-ray" in command
