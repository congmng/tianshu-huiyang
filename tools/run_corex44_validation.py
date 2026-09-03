#!/usr/bin/env python3
"""Run repeatable Python 3.12/CoreX validation by test level.

Levels are cumulative only in intent, not automatically chained:

* ``unit``: CPU/isolated-Ray V1, KV-affinity and HTTP contract tests.
* ``integration``: two-host version/hash gate plus a real GPU BF16 KV staging
  transfer from this host to ``--remote-host``.
* ``e2e``: Qwen3-14B real inference using one or more local CoreX GPUs.

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
             "tests/unit_test/backend/test_v1_kv_transfer.py",
             "tests/unit_test/global_scheduler/test_v1_kv_affinity.py",
             "tests/unit_test/entrypoints/vllm/test_v1_api_server.py"]]


def run_integration(local_ip: str, remote_ip: str, remote_host: str,
                    remote_project: str, dry_run: bool) -> None:
    run([sys.executable, "tools/corex44_support_check.py", "--remote-host", remote_host,
         "--remote-project", remote_project], dry_run)
    consumer_port = free_port()
    producer_port = free_port()
    remote_cmd = (
        f"cd {remote_project} && source tools/corex44_env.sh && "
        f"CUDA_VISIBLE_DEVICES=0 python tools/corex44_zmq_kv_probe.py "
        f"--role consumer --host {remote_ip} --port {consumer_port} --timeout 30"
    )
    local_cmd = [sys.executable, "tools/corex44_zmq_kv_probe.py", "--role", "producer",
                 "--host", local_ip, "--port", str(producer_port),
                 "--peer", f"{remote_ip}:{consumer_port}", "--timeout", "30"]
    print("+ ssh", remote_host, remote_cmd, flush=True)
    if dry_run:
        print("+", " ".join(local_cmd), flush=True)
        return
    remote = subprocess.Popen(["ssh", remote_host, remote_cmd])
    try:
        time.sleep(2)
        run(local_cmd, False)
        if remote.wait(timeout=40) != 0:
            raise RuntimeError("remote CoreX KV consumer failed")
    finally:
        if remote.poll() is None:
            remote.terminate()
            remote.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", choices=("unit", "integration", "e2e"))
    parser.add_argument("--remote-host", default="congmng@10.31.10.210")
    parser.add_argument("--local-ip", default="10.31.10.62")
    parser.add_argument("--remote-ip", default="10.31.10.210")
    parser.add_argument("--remote-project", default="/data1/congmng/llumnix")
    parser.add_argument("--tp", type=int, default=1, choices=(1, 2))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.level == "unit":
        for command in unit_commands():
            run(command, args.dry_run)
    elif args.level == "integration":
        run_integration(args.local_ip, args.remote_ip, args.remote_host,
                        args.remote_project, args.dry_run)
    else:
        visible = "0" if args.tp == 1 else "0,1"
        command = ["env", f"CUDA_VISIBLE_DEVICES={visible}",
                   f"TENSOR_PARALLEL_SIZE={args.tp}", "MAX_MODEL_LEN=256",
                   sys.executable, "tools/run_qwen3_14b_smoke.py"]
        run(command, args.dry_run)


if __name__ == "__main__":
    main()
