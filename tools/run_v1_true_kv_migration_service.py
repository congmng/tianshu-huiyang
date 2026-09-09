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
import re
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
MIGRATION_EVENT_RE = re.compile(
    r"Instance\s+([0-9a-fA-F-]+)\s*->\s*([0-9a-fA-F-]+)\s+migrated request\s+(\S+)"
)
PERMANENT_FAILURE_RE = re.compile(r"V1 migration .* permanently failed")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def migration_gpu_ids(tensor_parallel_size: int, gpu_ids: str) -> str:
    """Resolve the visible GPU list for two ``tensor_parallel_size`` Llumlets."""
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    if gpu_ids:
        return gpu_ids
    return ",".join(str(index) for index in range(2 * tensor_parallel_size))


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


def log_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def migration_events(path: Path) -> list[tuple[str, str, str]]:
    """Return successful Manager migration events from the service log."""
    return [
        (match.group(1), match.group(2), match.group(3))
        for match in MIGRATION_EVENT_RE.finditer(log_text(path))
    ]


def decode_stream_records(chunks: list[bytes]) -> list[dict]:
    """Decode the NUL-delimited JSON stream returned by Llumnix."""
    payload = b"".join(chunks).decode("utf-8", errors="replace")
    records: list[dict] = []
    for part in payload.split("\0"):
        part = part.strip()
        if part:
            records.append(json.loads(part))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / ".models/Qwen3-14B"))
    parser.add_argument("--api-port", type=int, default=0)
    parser.add_argument("--base-p2p-port", type=int, default=29500)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--request-output-queue-port", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--log-file", default="v1_true_kv_migration_service.log")
    parser.add_argument(
        "--vllm-migration-fork",
        default=os.environ.get(
            "LLUMNIX_VLLM_MIGRATION_FORK",
            "/data1/congmng/vllm-corex44-v1-migration",
        ),
        help="CoreX vLLM fork worktree implementing the V1 true-KV API",
    )
    parser.add_argument(
        "--gpu-ids", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        help="comma-separated visible GPUs for the two Llumlets",
    )
    args = parser.parse_args()
    try:
        args.gpu_ids = migration_gpu_ids(
            args.tensor_parallel_size, args.gpu_ids
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    model = Path(args.model).resolve()
    if not (model / "config.json").is_file():
        raise SystemExit(f"model is not a complete Hugging Face directory: {model}")
    migration_fork = Path(args.vllm_migration_fork).resolve()
    if not (migration_fork / "vllm" / "v1" / "migration.py").is_file():
        raise SystemExit(
            "V1 true-KV migration fork not found at "
            f"{migration_fork}; pass --vllm-migration-fork or set "
            "LLUMNIX_VLLM_MIGRATION_FORK"
        )

    api_port = args.api_port or free_port()
    queue_port = args.request_output_queue_port or free_port()
    log_path = ROOT / args.log_file
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (
            str(migration_fork),
            str(ROOT),
            environment.get("PYTHONPATH", ""),
        ) if path
    )
    environment.update({
        "CUDA_VISIBLE_DEVICES": args.gpu_ids,
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
        "--tensor-parallel-size", str(args.tensor_parallel_size),
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
        stream_chunks: list[bytes] = []

        def consume_stream() -> None:
            try:
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        stream_chunks.append(chunk)
            except BaseException as exc:  # noqa: BLE001 - captured for the test
                stream_error.append(exc)
            finally:
                stream_done.set()

        consumer = threading.Thread(target=consume_stream, daemon=True)
        consumer.start()

        deadline = time.monotonic() + args.timeout
        migration_events_seen: list[tuple[str, str, str]] = []
        while time.monotonic() < deadline:
            migration_events_seen = migration_events(log_path)
            if migration_events_seen:
                break
            time.sleep(0.2)
        if not migration_events_seen:
            raise AssertionError(
                "Manager did not complete a V1 migration; log excerpt:\n"
                + log_text(log_path)[-4000:]
            )
        matched_migrations = [
            event for event in migration_events_seen
            if event[2] == "service-v1-migration" and event[0] != event[1]
        ]
        if not matched_migrations:
            raise AssertionError(
                "no valid source->target migration event for "
                f"service-v1-migration: {migration_events_seen}"
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
        records = decode_stream_records(stream_chunks)
        if not records:
            raise AssertionError("migrated request produced no stream records")
        completion = records[-1]["text"][0]
        # The migrated target's first post-import output can omit the original
        # prompt echo, so the service-level contract here is non-empty
        # continuation rather than a full prompt+completion reconstruction.
        if not completion.strip() or completion == args.prompt:
            raise AssertionError(
                f"migrated request produced no continuation: {completion!r}"
            )
        if PERMANENT_FAILURE_RE.search(log_text(log_path)):
            raise AssertionError(
                "service E2E log contains a permanent V1 migration failure"
            )
        source_id, target_id, request_id = matched_migrations[0]
        print(f"migration={source_id}->{target_id}", flush=True)
        print(f"request_id={request_id}", flush=True)
        print(f"completion={completion}", flush=True)
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
