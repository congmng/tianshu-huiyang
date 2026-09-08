#!/usr/bin/env python3
"""Run a full service-level V1 true-KV migration E2E.

This starts the real Llumnix V1 HTTP frontend (``api_server``), which in turn
creates a Ray cluster, a Manager, and two homogeneous V1 Llumlets.  A
streaming request is pinned to one instance long enough for Manager's
capability-gated V1 migration loop to move it to the other instance, then the
harness verifies that the Manager reported the migration and that the response
completed normally.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def request_json(url: str, payload: dict | None = None) -> tuple[int, object]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read().decode()
    try:
        return response.status, json.loads(body)
    except json.JSONDecodeError:
        return response.status, body


def wait_ready(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            status, _ = request_json(base + "/health")
            if status == 200:
                status, ready = request_json(base + "/is_ready")
                if status == 200 and ready is True:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for Llumnix service")
        time.sleep(1)


def log_contains(path: Path, needle: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return needle in text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / ".models/Qwen3-14B"))
    parser.add_argument("--api-port", type=int, default=0)
    parser.add_argument("--base-p2p-port", type=int, default=29500)
    parser.add_argument("--request-output-queue-port", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--log-file", default="v1_true_kv_migration_service.log")
    args = parser.parse_args()

    model = Path(args.model).resolve()
    if not (model / "config.json").is_file():
        raise SystemExit(f"model is not a complete Hugging Face directory: {model}")

    api_port = args.api_port or free_port()
    queue_port = args.request_output_queue_port or free_port()
    log_path = ROOT / args.log_file
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0,1",
        "RAY_DEDUP_LOGS": "0",
        "HEAD_NODE_IP": "127.0.0.1",
        "HEAD_NODE": "1",
        "LLUMNIX_KV_IP": "127.0.0.1",
        "LLUMNIX_KV_PORT": str(args.base_p2p_port),
        "LLUMNIX_TRUE_KV_MIGRATION_ONLY": "1",
        "LLUMNIX_COREX_TRANSPORT": "nccl",
        "PYTHONHASHSEED": "0",
        "VLLM_FORCE_NCCL_COMM": "1",
    })
    command = [
        sys.executable, "-m", "llumnix.entrypoints.vllm.api_server",
        "--host", "127.0.0.1",
        "--port", str(api_port),
        "--initial-instances", "2",
        "--enable-migration",
        "--pair-migration-frequency", "1",
        "--polling-interval", "0.05",
        "--dispatch-policy", "load",
        "--migration-backend", "kvtransfer",
        "--migration-backend-transfer-type", "CoreXP2pNcclConnector",
        "--model", str(model),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.96",
        "--enforce-eager",
        "--trust-remote-code",
        "--tensor-parallel-size", "1",
        "--request-output-queue-port", str(queue_port),
        "--launch-ray-cluster",
        "--log-request-timestamps",
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
    base = f"http://127.0.0.1:{api_port}"
    response = None
    try:
        wait_ready(base, args.timeout)
        status, instance_payload = request_json(base + "/instance_list")
        assert status == 200, (status, instance_payload)
        instances = instance_payload.get("data", [])
        assert len(instances) >= 2, instances
        print(f"api_port={api_port}", flush=True)
        print(f"instance_count={len(instances)}", flush=True)

        payload = {
            "prompt": args.prompt,
            "request_id": "service-v1-migration",
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
        }
        response = requests.post(
            base + "/generate", json=payload, stream=True, timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"generate request failed with status {response.status_code}: "
                f"{response.text[:200]}"
            )

        stream_error: list[BaseException] = []
        stream_done = threading.Event()

        def consume_stream() -> None:
            try:
                for _ in response.iter_content(chunk_size=None, decode_unicode=True):
                    pass
            except BaseException as exc:  # noqa: BLE001 - captured for the test
                stream_error.append(exc)
            finally:
                stream_done.set()

        consumer = threading.Thread(target=consume_stream, daemon=True)
        consumer.start()

        deadline = time.monotonic() + args.timeout
        migration_seen = False
        while time.monotonic() < deadline:
            if log_contains(log_path, "migrated request"):
                migration_seen = True
                break
            time.sleep(0.2)
        if not migration_seen:
            raise AssertionError(
                "Manager did not complete a V1 migration; log excerpt:\n"
                + log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            )
        # The service-level contract is not satisfied by a Manager log line
        # alone: the migrated request must finish normally on the target.
        if not stream_done.wait(timeout=5.0):
            raise AssertionError(
                "migrated request did not finish streaming after migration"
            )
        if stream_error:
            raise AssertionError(
                f"migrated request stream failed: {stream_error[0]}"
            )
        print("PASS service_v1_true_kv_migration", flush=True)
    finally:
        if response is not None:
            response.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        subprocess.run(
            ["ray", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        )


if __name__ == "__main__":
    main()
