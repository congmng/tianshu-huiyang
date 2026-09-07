#!/usr/bin/env python3
"""Launch two CUDA-scoped workers and run a single real V1 KV migration.

The script is intentionally opt-in and uses only local loopback control
ports.  It does not download models or alter installed packages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "v1_true_kv_migration_worker.py"


async def rpc(port: int, command: dict) -> dict:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(json.dumps(command).encode() + b"\n")
    await writer.drain()
    response = json.loads((await reader.readline()).decode())
    writer.close()
    await writer.wait_closed()
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "worker command failed"))
    return response["result"]


async def wait_ready(port: int, process: asyncio.subprocess.Process) -> None:
    for _ in range(600):
        if process.returncode is not None:
            raise RuntimeError(f"worker exited before READY (code={process.returncode})")
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise TimeoutError(f"worker did not open control port {port}")


async def run(args: argparse.Namespace) -> None:
    common = [sys.executable, "-u", str(WORKER), "--model", args.model,
              "--transport", args.transport, "--max-model-len", str(args.max_model_len),
              "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    env = os.environ.copy()
    fork = "/data1/congmng/vllm-corex44-v1-migration"
    env["PYTHONPATH"] = f"{fork}:{ROOT}:{env.get('PYTHONPATH', '')}"
    source = await asyncio.create_subprocess_exec(
        *common, "--role", "source", "--control-port", str(args.source_control),
        "--p2p-port", str(args.source_p2p), "--peer-p2p", str(args.target_p2p),
        env={**env, "CUDA_VISIBLE_DEVICES": str(args.source_gpu)},
    )
    target = await asyncio.create_subprocess_exec(
        *common, "--role", "target", "--control-port", str(args.target_control),
        "--p2p-port", str(args.target_p2p), "--peer-p2p", str(args.source_p2p),
        env={**env, "CUDA_VISIBLE_DEVICES": str(args.target_gpu)},
    )
    try:
        await asyncio.gather(wait_ready(args.source_control, source),
                             wait_ready(args.target_control, target))
        request_id, epoch = args.request_id, args.epoch
        # The source baseline is the authoritative greedy sequence. A target
        # baseline is retained as diagnostic evidence because different CoreX
        # devices may have non-bitwise-identical FP16 reductions.
        source_baseline = await rpc(args.source_control, {
            "op": "baseline", "request_id": f"{request_id}-baseline",
            "prompt": args.prompt, "max_tokens": 12,
        })
        target_baseline = await rpc(args.target_control, {
            "op": "baseline", "request_id": f"{request_id}-target-baseline",
            "prompt": args.prompt, "max_tokens": 12,
        })
        generated = await rpc(args.source_control, {"op": "generate", "request_id": request_id,
                                                    "prompt": args.prompt})
        request_id = generated["request_id"]
        out = await rpc(args.source_control, {"op": "prepare_out", "request_id": request_id,
                                              "epoch": epoch})
        snapshot = out["snapshot"]
        target_blocks = await rpc(args.target_control, {"op": "prepare_in", "snapshot": snapshot})
        source_blocks = await rpc(args.source_control, {"op": "blocks", "request_id": request_id,
                                                        "epoch": epoch})
        layers = await rpc(args.source_control, {"op": "layers"})
        for group_src, group_dst in zip(source_blocks["blocks"], target_blocks["blocks"]):
            for layer in layers["layers"]:
                manifest = await rpc(args.source_control, {
                    "op": "send", "request_id": request_id, "epoch": epoch,
                    "layer": layer, "source_blocks": group_src,
                    "target_blocks": group_dst,
                    "peer": f"127.0.0.1:{args.target_p2p}",
                })
                await rpc(args.target_control, {
                    "op": "receive", "request_id": request_id, "epoch": epoch,
                    "manifest": manifest["manifest"],
                    "peer": f"127.0.0.1:{args.source_p2p}",
                })
        await rpc(args.target_control, {"op": "commit", "request_id": request_id,
                                        "epoch": epoch, "incoming": True})
        await rpc(args.source_control, {"op": "commit", "request_id": request_id,
                                        "epoch": epoch, "incoming": False})
        resumed = await rpc(args.target_control, {"op": "resume", "request_id": request_id,
                                                   "epoch": epoch, "prompt": args.prompt,
                                                   "tokens": args.verify_tokens})
        # EngineCore may execute one pending decode input before the control
        # command reaches its boundary.  The snapshot's serialized output
        # history, rather than the frontend's observation timing, defines the
        # exact continuation point.
        continuation = len(out["output_token_ids"])
        expected = source_baseline["token_ids"][continuation:
                                                 continuation + args.verify_tokens]
        if resumed["token_ids"] != expected:
            raise AssertionError(
                f"post-migration token mismatch: source={generated['token_ids']}, "
                f"source_baseline={source_baseline['token_ids']}, "
                f"target_baseline={target_baseline['token_ids']}, expected={expected}, "
                f"got={resumed['token_ids']}")
        print("PASS phase2 migration control+KV transfer+two-phase-commit+decode-equivalence", flush=True)
    finally:
        for port, process in ((args.source_control, source), (args.target_control, target)):
            if process.returncode is None:
                try:
                    await rpc(port, {"op": "shutdown"})
                except Exception:
                    process.send_signal(signal.SIGTERM)
        await asyncio.gather(source.wait(), target.wait())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / ".models" / "Qwen3-14B"))
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    parser.add_argument("--source-control", type=int, default=19201)
    parser.add_argument("--target-control", type=int, default=19202)
    parser.add_argument("--source-p2p", type=int, default=19211)
    parser.add_argument("--target-p2p", type=int, default=19212)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--request-id", default="phase2-real-migration")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.96)
    parser.add_argument("--transport", choices=("nccl", "zmq_cpu"), default="nccl")
    parser.add_argument("--verify-tokens", type=int, default=2)
    parser.add_argument("--prompt", default="Explain KV cache migration in one sentence.")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
