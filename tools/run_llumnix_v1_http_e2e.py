#!/usr/bin/env python3
"""Exercise the Llumnix V1 HTTP frontend with a real local model and CoreX GPU.

This is deliberately separate from the direct-vLLM smoke: it starts
``llumnix.entrypoints.vllm.v1_api_server`` and verifies its health, topology,
and generation contracts. It only terminates the process group it creates.
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
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    parser.add_argument("--max-model-len", type=int, default=256)
    args = parser.parse_args()
    model = Path(args.model).resolve()
    if not (model / "config.json").is_file():
        raise SystemExit(f"model is not a complete Hugging Face directory: {model}")
    port = args.port or free_port()
    command = [
        sys.executable, "-m", "llumnix.entrypoints.vllm.v1_api_server",
        "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
        "--max-model-len", str(args.max_model_len), "--max-num-seqs", "1",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--enforce-eager",
    ]
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    process = subprocess.Popen(command, cwd=ROOT, env=environment)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + args.timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"V1 HTTP server exited early ({process.returncode})")
            try:
                status, _ = request(base + "/health")
                if status == 200:
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Llumnix V1 HTTP server")
            time.sleep(1)
        status, ready = request(base + "/is_ready")
        assert status == 200 and ready == "true", (status, ready)
        status, topology = request(base + "/instance_list")
        topology_data = json.loads(topology)["data"]
        assert status == 200 and topology_data and topology_data[0]["gpu_count"] >= 1
        status, generated = request(base + "/generate", {
            "prompt": "请用一句话说明 KV cache 的作用。",
            "request_id": "corex-v1-http-e2e",
            "max_tokens": 12,
            "temperature": 0,
        })
        text = json.loads(generated)["text"][0]
        assert status == 200 and text.strip(), generated
        print(f"port={port}")
        print(f"gpu_count={topology_data[0]['gpu_count']}")
        print(f"completion={text}")
        print("llumnix_v1_http_corex: PASS")
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
