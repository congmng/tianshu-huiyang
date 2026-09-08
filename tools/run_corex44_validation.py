#!/usr/bin/env python3
"""Run repeatable Python 3.12/CoreX validation by test level.

Levels are cumulative only in intent, not automatically chained:

* ``unit``: CPU/isolated-Ray V1, KV-affinity and HTTP contract tests.
* ``integration``: two-host version/hash gate, V1 KV-event affinity, then a
  real GPU BF16 KV staging transfer from this host to ``--remote-host``.
* ``e2e``: Qwen3-14B real inference plus the Llumnix V1 HTTP frontend.

The runner does not manage shared Ray clusters and never deletes model or Ray
state. Source ``tools/corex44_env.sh`` first. Use ``--dry-run`` to print
commands before operating on a deployment node.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COREX45_DOCKER_IMAGE = (
    "dev-community-acr-registry.cn-shanghai.cr.aliyuncs.com/"
    "dev-community/dev-community:vllm-py3.12-corex.4.5.0-ubuntu24.04"
)


def corex45_docker_gate_command(remote_project: str) -> str:
    """Return the remote shell command for the 4.5 Docker runtime gate."""
    script = f"{remote_project}/tools/corex45_docker_support_check.py"
    return (
        "LIBS=$(find /data/tianshu/20260720/corex/corex/corex-toolkit "
        "-mindepth 2 -maxdepth 2 -type d -name lib64 | paste -sd: -)"
        ":/usr/local/corex-4.5.0/lib64:/usr/local/openmpi/lib; "
        "docker run --rm --privileged "
        "-v /dev:/dev:ro "
        f"-v {remote_project}:{remote_project}:ro "
        "-v /data/tianshu/20260720/corex:/data/tianshu/20260720/corex:ro "
        "-v /usr/local/corex-4.5.0:/usr/local/corex-4.5.0:ro "
        '-e LD_LIBRARY_PATH="$LIBS" '
        f"{COREX45_DOCKER_IMAGE} python {script} --json"
    )


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def run(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def unit_commands() -> list[list[str]]:
    return [[sys.executable, "-m", "pytest", "-q",
             "tests/unit_test/test_corex44_support_check.py",
             "tests/unit_test/test_corex44_pd_config.py",
             "tests/unit_test/backend/test_v1_kv_transfer.py",
             "tests/unit_test/llumlet/test_v1_migration_capabilities.py",
             "tests/unit_test/global_scheduler/test_v1_migration.py",
             "tests/unit_test/global_scheduler/test_v1_kv_affinity.py",
             "tests/unit_test/global_scheduler/test_dispatch_scheduler.py",
             "tests/unit_test/global_scheduler/test_manager.py::test_manager_v1_pd_role_selection_uses_kv_affinity",
             "tests/unit_test/global_scheduler/test_manager.py::test_manager_v1_pd_role_selection_waits_when_a_role_is_missing",
             "tests/unit_test/global_scheduler/test_manager.py::test_manager_abort_removes_pd_role_waiter_without_actor_abort",
             "tests/unit_test/global_scheduler/test_manager.py::test_manager_pd_waiter_can_cancel_when_one_role_pool_is_missing",
             "tests/unit_test/global_scheduler/test_manager.py::test_pd_state_check_does_not_treat_no_constraints_as_prefill",
             "tests/unit_test/entrypoints/vllm/test_e2e_command_utils.py",
             "tests/unit_test/entrypoints/vllm/test_v1_api_server.py"]]


def _run_corex45_docker_gate(remote_host: str, remote_project: str,
                             dry_run: bool,
                             ssh_password: str | None) -> None:
    """Run the vLLM 0.23 CoreX 4.5 Docker gate and compare source hashes."""
    remote_cmd = corex45_docker_gate_command(remote_project)
    ssh_prefix = (["sshpass", "-p", ssh_password] if ssh_password else [])
    ssh_cmd = ssh_prefix + [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", remote_host, remote_cmd,
    ]
    print("+", " ".join(ssh_cmd), flush=True)
    if dry_run:
        return
    completed = subprocess.run(
        ssh_cmd, check=False, capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "remote CoreX 4.5 Docker gate failed: "
            + (completed.stderr or completed.stdout).strip()[-2000:]
        )
    import importlib.util
    import json

    gate_path = ROOT / "tools" / "corex44_support_check.py"
    spec = importlib.util.spec_from_file_location("corex44_support_check", gate_path)
    gate44 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate44)
    local_fingerprint = gate44.source_fingerprint()
    remote_result = json.loads(completed.stdout.strip().splitlines()[-1])
    if not remote_result.get("supported", False):
        raise RuntimeError(
            "remote CoreX 4.5 Docker gate is unsupported: "
            + repr(remote_result.get("errors"))
        )
    remote_fingerprint = remote_result.get("source_fingerprint")
    if remote_fingerprint != local_fingerprint:
        raise RuntimeError(
            "CoreX 4.5 remote source fingerprint differs: "
            f"local={local_fingerprint} remote={remote_fingerprint}"
        )
    print(json.dumps(remote_result, sort_keys=True), flush=True)


def run_integration(local_ip: str, remote_ip: str, remote_host: str,
                    remote_project: str, dry_run: bool, model_pd: bool = False,
                    model: str = "/data1/congmng/llumnix/.models/Qwen3-14B",
                    corex_transport: str = "nccl", corex_stack: str = "44",
                    remote_stack: str | None = None,
                    ssh_password: str | None = None) -> None:
    remote_stack = remote_stack or corex_stack
    if remote_stack == "45":
        _run_corex45_docker_gate(
            remote_host, remote_project, dry_run, ssh_password
        )
        print(
            "INFO remote CoreX 4.5 runtime gate passed; skipping 0.11-fork "
            "KV event/model probes until the vLLM 0.23 migration adapter "
            "is ported",
            flush=True,
        )
        return
    support_cmd = [sys.executable, "tools/corex44_support_check.py",
                   "--remote-host", remote_host, "--remote-project", remote_project,
                   "--corex-stack", corex_stack, "--remote-stack", remote_stack,
                   "--mixed-stack"]
    if ssh_password:
        support_cmd.extend(["--ssh-password", ssh_password])
    run(support_cmd, dry_run)
    event_port = free_port()
    # Some CoreX deployments firewall arbitrary ZMQ ports between nodes even
    # though SSH is allowed.  Use an SSH reverse tunnel for the event control
    # plane; the publisher/subscriber and msgspec payload remain real vLLM
    # ZMQ traffic, while the validation is not coupled to firewall policy.
    event_remote_cmd = (
        f"cd {remote_project} && LLUMNIX_COREX_STACK={remote_stack} "
        "source tools/corex_env.sh && PYTHONPATH=. "
        "python tools/corex44_kv_event_probe.py --role consumer "
        f"--host 127.0.0.1 --port {event_port} --timeout 15"
    )
    event_local_cmd = [sys.executable, "tools/corex44_kv_event_probe.py", "--role", "publisher",
                       "--host", "127.0.0.1", "--port", str(event_port), "--timeout", "4"]
    ssh_prefix = (["sshpass", "-p", ssh_password] if ssh_password else [])
    print("+ ssh", remote_host, event_remote_cmd, flush=True)
    if dry_run:
        print("+", " ".join(event_local_cmd), flush=True)
    else:
        # Bind the local PUB endpoint before the reverse forward accepts the
        # remote subscriber. Otherwise SSH eagerly opens the forwarded local
        # socket and a one-shot ZMQ subscriber can observe connection-refused.
        event_publisher = subprocess.Popen(event_local_cmd, cwd=ROOT)
        time.sleep(0.5)
        event_remote = subprocess.Popen(ssh_prefix + [
            "ssh", "-o", "ExitOnForwardFailure=yes", "-R",
            f"{event_port}:127.0.0.1:{event_port}", remote_host, event_remote_cmd,
        ])
        try:
            time.sleep(1)
            if event_remote.poll() is not None:
                raise RuntimeError(f"remote KV-event consumer exited during startup (exit_code={event_remote.returncode})")
            if event_publisher.wait(timeout=20) != 0:
                raise RuntimeError("local KV-event publisher failed")
            if event_remote.wait(timeout=20) != 0:
                raise RuntimeError("remote KV-event consumer failed")
        finally:
            if event_publisher.poll() is None:
                event_publisher.terminate()
                event_publisher.wait(timeout=5)
            if event_remote.poll() is None:
                event_remote.terminate()
                event_remote.wait(timeout=5)
    consumer_port = free_port()
    producer_port = free_port()
    remote_cmd = (
        f"cd {remote_project} && LLUMNIX_COREX_STACK={remote_stack} "
        "source tools/corex_env.sh && CUDA_VISIBLE_DEVICES=0 "
        "python tools/corex44_zmq_kv_probe.py "
        f"--role consumer --host {remote_ip} --port {consumer_port} --timeout 30"
    )
    local_cmd = [sys.executable, "tools/corex44_zmq_kv_probe.py", "--role", "producer",
                 "--host", local_ip, "--port", str(producer_port),
                 "--peer", f"{remote_ip}:{consumer_port}", "--timeout", "30"]
    print("+ ssh", remote_host, remote_cmd, flush=True)
    if dry_run:
        print("+", " ".join(local_cmd), flush=True)
    else:
        remote = subprocess.Popen(ssh_prefix + ["ssh", remote_host, remote_cmd])
        try:
            time.sleep(2)
            if remote.poll() is not None:
                raise RuntimeError(
                    f"remote CoreX KV consumer exited during startup "
                    f"(exit_code={remote.returncode})"
                )
            run(local_cmd, False)
            if remote.wait(timeout=40) != 0:
                raise RuntimeError("remote CoreX KV consumer failed")
        finally:
            if remote.poll() is None:
                remote.terminate()
                remote.wait(timeout=5)
    if not model_pd:
        return

    # Optional expensive stage: two real V1 engines perform a connector-driven
    # Prefill/Decode handoff. Keep it opt-in because it loads two 14B models,
    # while the default integration stage already validates the transport.
    pd_port = free_port()
    request_id = "corex-pd-model-validation"
    transport_arg = f" --corex-transport {corex_transport}"
    remote_pd_cmd = (
        f"cd {remote_project} && LLUMNIX_COREX_STACK={remote_stack} "
        "source tools/corex_env.sh && CUDA_VISIBLE_DEVICES=0 "
        "PYTHONHASHSEED=0 python tools/v1_p2p_model_probe.py "
        f"--role consumer --model {model} --host {remote_ip} "
        f"--peer {local_ip}:{pd_port} --port {pd_port} --request-id {request_id} "
        f"--max-model-len 256 --max-tokens 4{transport_arg}"
    )
    local_pd_cmd = ["env", "CUDA_VISIBLE_DEVICES=0", "PYTHONHASHSEED=0", sys.executable,
                    "tools/v1_p2p_model_probe.py", "--role", "producer",
                    "--model", model, "--host", local_ip, "--peer",
                    f"{remote_ip}:{pd_port}", "--port", str(pd_port),
                    "--request-id", request_id, "--max-model-len", "256", "--max-tokens", "4"]
    local_pd_cmd.extend(["--corex-transport", corex_transport])
    print("+ ssh", remote_host, remote_pd_cmd, flush=True)
    if dry_run:
        print("+", " ".join(local_pd_cmd), flush=True)
        return
    remote_pd = subprocess.Popen(ssh_prefix + ["ssh", remote_host, remote_pd_cmd])
    try:
        time.sleep(2)
        if remote_pd.poll() is not None:
            raise RuntimeError(f"remote model P/D consumer exited during startup (exit_code={remote_pd.returncode})")
        run(local_pd_cmd, False)
        if remote_pd.wait(timeout=180) != 0:
            raise RuntimeError("remote model P/D consumer failed")
    finally:
        if remote_pd.poll() is None:
            remote_pd.terminate()
            remote_pd.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", choices=("unit", "integration", "e2e"))
    parser.add_argument("--remote-host", default="congmng@10.31.10.210")
    parser.add_argument("--local-ip", default="10.31.10.62")
    parser.add_argument("--remote-ip", default="10.31.10.210")
    parser.add_argument("--remote-project", default="/data1/congmng/llumnix")
    parser.add_argument("--corex-stack", default=os.getenv("LLUMNIX_COREX_STACK", "44"),
                        choices=("44", "45"))
    parser.add_argument("--remote-stack", default=None, choices=("44", "45"),
                        help="remote stack; defaults to --corex-stack")
    parser.add_argument("--ssh-password", default=None,
                        help="optional password for sshpass-based remote login")
    parser.add_argument("--tp", type=int, default=1, choices=(1, 2))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-pd", action="store_true",
                        help="also run the expensive two-host Qwen3 V1 P/D handoff")
    parser.add_argument("--model", default="/data1/congmng/llumnix/.models/Qwen3-14B",
                        help="model path for --model-pd")
    parser.add_argument("--corex-transport", choices=("nccl", "zmq_cpu"),
                        default="nccl", help=("transport for --model-pd; native NCCL is "
                                               "the default, zmq_cpu is an explicit fallback"))
    parser.add_argument("--native-nccl", action="store_const", const="nccl",
                        dest="corex_transport",
                        help="deprecated compatibility alias for --corex-transport nccl")
    args = parser.parse_args()
    if args.level == "unit":
        for command in unit_commands():
            run(command, args.dry_run)
    elif args.level == "integration":
        run_integration(args.local_ip, args.remote_ip, args.remote_host,
                        args.remote_project, args.dry_run, args.model_pd, args.model,
                        args.corex_transport, args.corex_stack, args.remote_stack,
                        args.ssh_password)
    else:
        visible = "0" if args.tp == 1 else "0,1"
        command = ["env", f"CUDA_VISIBLE_DEVICES={visible}",
                   f"TENSOR_PARALLEL_SIZE={args.tp}", "MAX_MODEL_LEN=256",
                   sys.executable, "tools/run_qwen3_14b_smoke.py"]
        run(command, args.dry_run)
        # HTTP uses one GPU; run after direct TP inference releases its worker
        # processes. This validates the actual Llumnix V1 serving boundary.
        run(["env", "CUDA_VISIBLE_DEVICES=0", sys.executable,
             "tools/run_llumnix_v1_http_e2e.py"], args.dry_run)


if __name__ == "__main__":
    main()
