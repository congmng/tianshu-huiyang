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
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "v1_true_kv_migration_worker.py"


async def rpc(host: str, port: int, command: dict) -> dict:
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(json.dumps(command).encode() + b"\n")
    await writer.drain()
    response = json.loads((await reader.readline()).decode())
    writer.close()
    await writer.wait_closed()
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "worker command failed"))
    return response["result"]


async def checked_rpc(host: str, port: int, command: dict, timeout_s: float = 30.0) -> dict:
    """Bound every probe control call and identify its failed phase."""
    try:
        return await asyncio.wait_for(rpc(host, port, command), timeout=timeout_s)
    except TimeoutError as exc:
        raise TimeoutError(f"timeout op={command['op']} port={port}") from exc


async def wait_ready(host: str, port: int, process: asyncio.subprocess.Process) -> None:
    for _ in range(600):
        if process.returncode is not None:
            raise RuntimeError(f"worker exited before READY (code={process.returncode})")
        try:
            reader, writer = await asyncio.open_connection(host, port)
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
    # CoreX's optional ixformer communicator is not ABI-stable across hosts;
    # standard torch NCCL is sufficient for TP=1 and the migration P2P engine.
    env["VLLM_FORCE_NCCL_COMM"] = "1"
    # Freeze/import is performed by the fork's explicit V1 migration API.
    # Do not activate the unrelated legacy P/D attention lifecycle.
    env["LLUMNIX_TRUE_KV_MIGRATION_ONLY"] = "1"
    fork = "/data1/congmng/vllm-corex44-v1-migration"
    env["PYTHONPATH"] = f"{fork}:{ROOT}:{env.get('PYTHONPATH', '')}"
    source = await asyncio.create_subprocess_exec(
        *common, "--role", "source", "--control-port", str(args.source_control),
        "--control-host", args.source_host, "--p2p-host", args.source_p2p_host,
        "--p2p-port", str(args.source_p2p), "--peer-p2p", str(args.target_p2p),
        "--peer-p2p-host", args.target_p2p_host,
        env={**env, "CUDA_VISIBLE_DEVICES": str(args.source_gpu)},
    )
    target_args = ["python", "-u", str(WORKER), "--model", args.model,
                   "--transport", args.transport, "--max-model-len", str(args.max_model_len),
                   "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                   "--role", "target", "--control-port", str(args.target_control),
                   "--control-host", args.target_host, "--p2p-host", args.target_p2p_host,
                   "--p2p-port", str(args.target_p2p), "--peer-p2p", str(args.source_p2p),
                   "--peer-p2p-host", args.source_p2p_host]
    if args.inject_target_capacity:
        env["LLUMNIX_INJECT_TARGET_CAPACITY"] = "1"
    if args.target_ssh:
        remote = " ".join([f"'{part}'" for part in target_args])
        remote = (
            "source /data1/congmng/llumnix/tools/corex44_env.sh; "
            "export VLLM_FORCE_NCCL_COMM=1; "
            "export LLUMNIX_TRUE_KV_MIGRATION_ONLY=1; "
            + ("export LLUMNIX_INJECT_TARGET_CAPACITY=1; "
               if args.inject_target_capacity else "")
            + f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={fork}:{ROOT} "
            + f"CUDA_VISIBLE_DEVICES={args.target_gpu} exec {remote}"
        )
        target = await asyncio.create_subprocess_exec("ssh", args.target_ssh, remote)
    else:
        target = await asyncio.create_subprocess_exec(
            *target_args, env={**env, "CUDA_VISIBLE_DEVICES": str(args.target_gpu)})
    active_request_id: str | None = None
    active_epoch: int | None = None
    source_prepared = False
    target_prepared = False
    try:
        await asyncio.gather(wait_ready(args.source_host, args.source_control, source),
                             wait_ready(args.target_host, args.target_control, target))
        for iteration in range(args.iterations):
            request_id = args.request_id if args.iterations == 1 else f"{args.request_id}-{iteration}"
            epoch = args.epoch + iteration
            active_request_id, active_epoch = request_id, epoch
            # The source baseline is the authoritative greedy sequence. A
            # target baseline is diagnostic evidence for device differences.
            print(f"START iteration {iteration + 1}/{args.iterations} baseline-source", flush=True)
            source_baseline = await checked_rpc(args.source_host, args.source_control, {
                "op": "baseline", "request_id": f"{request_id}-baseline",
                "prompt": args.prompt, "max_tokens": 12,
            })
            print(f"START iteration {iteration + 1}/{args.iterations} baseline-target", flush=True)
            target_baseline = await checked_rpc(args.target_host, args.target_control, {
                "op": "baseline", "request_id": f"{request_id}-target-baseline",
                "prompt": args.prompt, "max_tokens": 12,
            })
            print(f"START iteration {iteration + 1}/{args.iterations} source-generate", flush=True)
            generated = await checked_rpc(args.source_host, args.source_control, {"op": "generate", "request_id": request_id,
                                                        "prompt": args.prompt}, args.rpc_timeout)
            request_id = generated["request_id"]
            out = await rpc(args.source_host, args.source_control, {"op": "prepare_out", "request_id": request_id,
                                              "epoch": epoch})
            source_prepared = True
            snapshot = out["snapshot"]
            target_blocks = await rpc(args.target_host, args.target_control, {"op": "prepare_in", "snapshot": snapshot})
            target_prepared = True
            source_blocks = await rpc(args.source_host, args.source_control, {"op": "blocks", "request_id": request_id,
                                                        "epoch": epoch})
            layers = await rpc(args.source_host, args.source_control, {"op": "layers"})
            for group_src, group_dst in zip(source_blocks["blocks"], target_blocks["blocks"]):
                for layer in layers["layers"]:
                    if args.inject_latency_ms:
                        await asyncio.sleep(args.inject_latency_ms / 1000.0)
                    manifest = await rpc(args.source_host, args.source_control, {
                    "op": "send", "request_id": request_id, "epoch": epoch,
                    "layer": layer, "source_blocks": group_src,
                    "target_blocks": group_dst,
                    "peer": f"{args.target_p2p_host}:{args.target_p2p}",
                })
                    await rpc(args.target_host, args.target_control, {
                    "op": "receive", "request_id": request_id, "epoch": epoch,
                    "manifest": manifest["manifest"],
                    "peer": f"{args.source_p2p_host}:{args.source_p2p}",
                })
            await rpc(args.target_host, args.target_control, {"op": "commit", "request_id": request_id,
                                        "epoch": epoch, "incoming": True})
            await rpc(args.source_host, args.source_control, {"op": "commit", "request_id": request_id,
                                        "epoch": epoch, "incoming": False})
            source_prepared = target_prepared = False
            resumed = await rpc(args.target_host, args.target_control, {"op": "resume", "request_id": request_id,
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
            if iteration == 0 or (iteration + 1) % 10 == 0:
                print(f"PASS iteration {iteration + 1}/{args.iterations}", flush=True)
        print("PASS phase2 migration control+KV transfer+two-phase-commit+decode-equivalence", flush=True)
    except BaseException:
        # A rejected reservation, interrupted peer, or bounded RPC timeout
        # must restore the source before shutdown.  Abort calls are best
        # effort here because the original failure is the authoritative one.
        if active_request_id is not None and active_epoch is not None:
            if target_prepared:
                try:
                    await asyncio.wait_for(rpc(args.target_host, args.target_control, {
                        "op": "abort", "request_id": active_request_id,
                        "epoch": active_epoch, "incoming": True}), timeout=10)
                except Exception:
                    pass
            if source_prepared:
                try:
                    await asyncio.wait_for(rpc(args.source_host, args.source_control, {
                        "op": "abort", "request_id": active_request_id,
                        "epoch": active_epoch, "incoming": False}), timeout=10)
                except Exception:
                    pass
        raise
    finally:
        for port, process in ((args.source_control, source), (args.target_control, target)):
            if process.returncode is None:
                try:
                    await rpc(args.source_host if port == args.source_control else args.target_host, port, {"op": "shutdown"})
                except Exception:
                    process.send_signal(signal.SIGTERM)
        await asyncio.gather(source.wait(), target.wait())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / ".models" / "Qwen3-14B"))
    parser.add_argument("--source-host", default="127.0.0.1")
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-ssh", default="", help="SSH destination for a remote target worker")
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    parser.add_argument("--source-control", type=int, default=19201)
    parser.add_argument("--target-control", type=int, default=19202)
    parser.add_argument("--source-p2p", type=int, default=19211)
    parser.add_argument("--source-p2p-host", default="127.0.0.1")
    parser.add_argument("--target-p2p", type=int, default=19212)
    parser.add_argument("--target-p2p-host", default="127.0.0.1")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--request-id", default="phase2-real-migration")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.96)
    parser.add_argument("--transport", choices=("nccl", "zmq_cpu"), default="nccl")
    parser.add_argument("--verify-tokens", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rpc-timeout", type=float, default=30.0,
                        help="timeout for bounded control operations")
    parser.add_argument("--inject-latency-ms", type=float, default=0.0,
                        help="delay before each layer transfer (fault injection)")
    parser.add_argument("--inject-target-capacity", action="store_true",
                        help="make target reservation fail deterministically")
    parser.add_argument("--prompt", default="Explain KV cache migration in one sentence.")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
