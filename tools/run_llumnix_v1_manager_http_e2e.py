#!/usr/bin/env python3
"""Exercise the Llumnix V1 Manager/Llumlet HTTP frontend on one GPU.

Unlike ``run_llumnix_v1_http_e2e.py``, this starts the real
``llumnix.entrypoints.vllm.api_server`` entrypoint, which creates a Ray
cluster, one Manager and one V1 Llumlet.  It is the smallest service-level
proof that the production dispatch path works on both CoreX 4.4 and 4.5
without requiring two GPUs or true-KV migration.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def request(url: str, payload: dict | None = None) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.status, response.read().decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / ".models/Qwen3-14B"))
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=256)
    args = parser.parse_args()
    model = Path(args.model).resolve()
    if not (model / "config.json").is_file():
        raise SystemExit(f"model is not a complete Hugging Face directory: {model}")

    port = args.port or free_port()
    environment = os.environ.copy()
    environment.update({
        "RAY_DEDUP_LOGS": "0",
        "HEAD_NODE_IP": "127.0.0.1",
        "HEAD_NODE": "1",
        "PYTHONHASHSEED": "0",
        "VLLM_FORCE_NCCL_COMM": "1",
    })
    command = [
        sys.executable, "-m", "llumnix.entrypoints.vllm.api_server",
        "--host", "127.0.0.1", "--port", str(port),
        "--initial-instances", "1",
        "--model", str(model), "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager", "--tensor-parallel-size", "1",
        "--launch-ray-cluster", "--dispatch-policy", "load",
    ]
    process = subprocess.Popen(command, cwd=ROOT, env=environment)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + args.timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"V1 Manager HTTP server exited early ({process.returncode})"
                )
            try:
                status, _ = request(base + "/health")
                if status == 200:
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Llumnix Manager HTTP server")
            time.sleep(1)

        status, ready = request(base + "/is_ready")
        assert status == 200 and ready == "true", (status, ready)
        status, topology = request(base + "/instance_list")
        topology_data = json.loads(topology)["data"]
        assert status == 200 and len(topology_data) == 1, topology_data
        status, generated = request(base + "/generate", {
            "prompt": "请用一句话说明 KV cache 的作用。",
            "request_id": "corex-v1-manager-http-e2e",
            "max_tokens": 12,
            "temperature": 0,
        })
        text = json.loads(generated)["text"][0]
        assert status == 200 and text.strip(), generated
        print(f"port={port}")
        print(f"instance_count={len(topology_data)}")
        print(f"completion={text}")
        print("llumnix_v1_manager_http_corex: PASS")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
