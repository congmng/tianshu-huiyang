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
import shlex
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


async def cleanup_remote_target(ssh_destination: str, control_port: int) -> None:
    """Remove only the remote worker belonging to this launcher invocation.

    A killed local SSH channel can leave the remote Python wrapper alive after
    its EngineCore has exited.  The control port is unique per invocation, so
    select the exact worker command rather than using a broad process match.
    """
    if not ssh_destination:
        return
    script = (
        "pids=$(ps -eo pid=,args= | awk -v port="
        + shlex.quote(str(control_port))
        + " '$0 ~ /v1_true_kv_migration_worker.py/ "
          "&& index($0, \"--control-port \" port) {print $1}'); "
          "for pid in $pids; do kill -TERM \"$pid\" 2>/dev/null || true; done; "
          "sleep 1; for pid in $pids; do kill -KILL \"$pid\" 2>/dev/null || true; done"
    )
    process = await asyncio.create_subprocess_exec("ssh", ssh_destination, script)
    await asyncio.wait_for(process.wait(), timeout=15)


async def run(args: argparse.Namespace) -> None:
    common = [sys.executable, "-u", str(WORKER), "--model", args.model,
              "--transport", args.transport, "--max-model-len", str(args.max_model_len),
              "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    common += ["--temperature", str(args.temperature)]
    if args.seed is not None:
        common += ["--seed", str(args.seed)]
    env = os.environ.copy()
    # CoreX's optional ixformer communicator is not ABI-stable across hosts;
    # standard torch NCCL is sufficient for TP=1 and the migration P2P engine.
    env["VLLM_FORCE_NCCL_COMM"] = "1"
    # Freeze/import is performed by the fork's explicit V1 migration API.
    # Do not activate the unrelated legacy P/D attention lifecycle.
    env["LLUMNIX_TRUE_KV_MIGRATION_ONLY"] = "1"
    fork = "/data1/congmng/vllm-corex44-v1-migration"
    env["PYTHONPATH"] = f"{fork}:{ROOT}:{env.get('PYTHONPATH', '')}"
    log_dir = Path(args.worker_log_dir) if args.worker_log_dir else None
    source_log = target_log = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        source_log = (log_dir / f"source-{args.source_control}.log").open("w")
        target_log = (log_dir / f"target-{args.target_control}.log").open("w")
        print(f"WORKER_LOG source={source_log.name} target={target_log.name}", flush=True)
    source = await asyncio.create_subprocess_exec(
        *common, "--role", "source", "--control-port", str(args.source_control),
        "--control-host", args.source_host, "--p2p-host", args.source_p2p_host,
        "--p2p-port", str(args.source_p2p), "--peer-p2p", str(args.target_p2p),
        "--peer-p2p-host", args.target_p2p_host,
        env={**env, "CUDA_VISIBLE_DEVICES": str(args.source_gpu)},
        stdout=source_log, stderr=asyncio.subprocess.STDOUT,
    )
    target_args = ["python", "-u", str(WORKER), "--model", args.model,
                   "--transport", args.transport, "--max-model-len", str(args.max_model_len),
                   "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                   "--role", "target", "--control-port", str(args.target_control),
                   "--control-host", args.target_host, "--p2p-host", args.target_p2p_host,
                   "--p2p-port", str(args.target_p2p), "--peer-p2p", str(args.source_p2p),
                   "--peer-p2p-host", args.source_p2p_host]
    target_args += ["--temperature", str(args.temperature)]
    if args.seed is not None:
        target_args += ["--seed", str(args.seed)]
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
            *target_args, env={**env, "CUDA_VISIBLE_DEVICES": str(args.target_gpu)},
            stdout=target_log, stderr=asyncio.subprocess.STDOUT)
    active_request_id: str | None = None
    active_epoch: int | None = None
    source_prepared = False
    target_prepared = False
    incremental_source = False
    incremental_target = False
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
                "prompt": args.prompt, "max_tokens": args.max_model_len,
            })
            print(f"START iteration {iteration + 1}/{args.iterations} baseline-target", flush=True)
            target_baseline = await checked_rpc(args.target_host, args.target_control, {
                "op": "baseline", "request_id": f"{request_id}-target-baseline",
                "prompt": args.prompt, "max_tokens": args.max_model_len,
            })
            print(f"START iteration {iteration + 1}/{args.iterations} source-generate", flush=True)
            generated = await checked_rpc(args.source_host, args.source_control, {"op": "generate", "request_id": request_id,
                                                        "prompt": args.prompt}, args.rpc_timeout)
            request_id = generated["request_id"]
            if args.incremental_precopy:
                # Pre-copy only immutable full blocks while source remains
                # runnable. This is deliberately explicit, one round at a
                # time; final cutover below still transfers the mutable tail.
                source_session = await rpc(args.source_host, args.source_control, {
                    "op": "incremental_begin", "request_id": request_id, "epoch": epoch})
                incremental_source = True
                immutable = await rpc(args.source_host, args.source_control, {
                    "op": "incremental_immutable_blocks", "request_id": request_id})
                if len(immutable["blocks"]) != 1:
                    raise RuntimeError("incremental precopy requires one KV cache group")
                target_session = await rpc(args.target_host, args.target_control, {
                    "op": "incremental_prepare_in", "session": source_session["session"],
                    "counts": immutable["counts"]})
                incremental_target = True
                source_ids, target_ids = immutable["blocks"][0], target_session["blocks"][0]
                if source_ids:
                    candidate = await rpc(args.source_host, args.source_control, {
                        "op": "incremental_preview", "request_id": request_id, "epoch": epoch + 1,
                        "pairs": list(zip(source_ids, target_ids))})
                    layers = await rpc(args.source_host, args.source_control, {"op": "layers"})
                    for layer in layers["layers"]:
                        manifest = await rpc(args.source_host, args.source_control, {
                            "op": "incremental_send", "session": candidate["session"],
                            "layer": layer, "source_blocks": source_ids,
                            "target_blocks": target_ids,
                            "peer": f"{args.target_p2p_host}:{args.target_p2p}"})
                        await rpc(args.target_host, args.target_control, {
                            "op": "incremental_receive", "session": candidate["session"],
                            "manifest": manifest["manifest"],
                            "peer": f"{args.source_p2p_host}:{args.source_p2p}"})
                    await rpc(args.target_host, args.target_control, {
                        "op": "incremental_commit_in", "session": candidate["session"]})
                    await rpc(args.source_host, args.source_control, {
                        "op": "incremental_append", "request_id": request_id, "epoch": epoch + 1,
                        "pairs": list(zip(source_ids, target_ids))})
            out = await rpc(args.source_host, args.source_control, {"op": "prepare_out", "request_id": request_id,
                                              "epoch": epoch})
            source_prepared = True
            snapshot = out["snapshot"]
            target_blocks = await rpc(args.target_host, args.target_control, {"op": "prepare_in", "snapshot": snapshot})
            target_prepared = True
            source_blocks = await rpc(args.source_host, args.source_control, {"op": "blocks", "request_id": request_id,
                                                        "epoch": epoch})
            synced_pairs = []
            if args.incremental_precopy:
                # Reuse target's committed pre-copy mapping.  The final
                # snapshot may have grown, so only the old ordinal prefix is
                # skipped; the mutable/new suffix is transferred now.
                synced_pairs = list(zip(source_ids, target_ids)) if source_ids else []
            layers = await rpc(args.source_host, args.source_control, {"op": "layers"})
            for group_src, group_dst in zip(source_blocks["blocks"], target_blocks["blocks"]):
                skip = len(synced_pairs)
                final_source = group_src[skip:] if skip else group_src
                final_target = group_dst[skip:] if skip else group_dst
                if not final_source:
                    continue
                for layer_index, layer in enumerate(layers["layers"], start=1):
                    if args.inject_latency_ms:
                        await asyncio.sleep(args.inject_latency_ms / 1000.0)
                    if (args.inject_transfer_timeout_after_layers
                            and layer_index > args.inject_transfer_timeout_after_layers):
                        raise TimeoutError(
                            "injected migration transfer timeout after "
                            f"{args.inject_transfer_timeout_after_layers} layers")
                    if (args.inject_source_restart_after_layers
                            and layer_index > args.inject_source_restart_after_layers):
                        # This is intentionally the exact source process
                        # launched above, never a broad process match.
                        source.send_signal(signal.SIGKILL)
                        raise RuntimeError(
                            "injected source actor restart after "
                            f"{args.inject_source_restart_after_layers} layers")
                    manifest = await rpc(args.source_host, args.source_control, {
                    "op": "send", "request_id": request_id, "epoch": epoch,
                    "layer": layer, "source_blocks": group_src,
                    "target_blocks": group_dst,
                    "synced_prefix_block_count": skip,
                    "peer": f"{args.target_p2p_host}:{args.target_p2p}",
                })
                    await rpc(args.target_host, args.target_control, {
                    "op": "receive", "request_id": request_id, "epoch": epoch,
                    "manifest": manifest["manifest"],
                    "synced_prefix_block_pairs": synced_pairs,
                    "peer": f"{args.source_p2p_host}:{args.source_p2p}",
                })
            await rpc(args.target_host, args.target_control, {"op": "commit", "request_id": request_id,
                                        "epoch": epoch, "incoming": True})
            await rpc(args.source_host, args.source_control, {"op": "commit", "request_id": request_id,
                                        "epoch": epoch, "incoming": False})
            source_prepared = target_prepared = False
            if incremental_source:
                await rpc(args.source_host, args.source_control, {"op": "incremental_abort", "request_id": request_id})
                incremental_source = False
            if incremental_target:
                await rpc(args.target_host, args.target_control, {"op": "incremental_abort", "request_id": request_id})
                incremental_target = False
            resumed = await rpc(args.target_host, args.target_control, {"op": "resume", "request_id": request_id,
                                                   "epoch": epoch, "prompt": args.prompt,
                                                   "tokens": args.verify_tokens})
        # EngineCore may execute one pending decode input before the control
        # command reaches its boundary.  The snapshot's serialized output
        # history, rather than the frontend's observation timing, defines the
        # exact continuation point.
            continuation = len(out["output_token_ids"])
            observed = resumed["token_ids"]
            # The source may complete one decode between the frontend's
            # output observation and EngineCore's token-boundary freeze. Use
            # the immutable snapshot history to locate the only valid baseline
            # window, while still rejecting arbitrary token mismatches.
            candidates = []
            for offset in range(0, 3):
                expected = source_baseline["token_ids"][continuation + offset:
                                                     continuation + offset + len(observed)]
                if observed == expected:
                    candidates.append((offset, expected))
            if len(candidates) != 1:
                raise AssertionError(
                f"post-migration token mismatch: source={generated['token_ids']}, "
                f"source_baseline={source_baseline['token_ids']}, "
                f"target_baseline={target_baseline['token_ids']}, expected={expected}, "
                f"got={resumed['token_ids']}")
            alignment_offset, expected = candidates[0]
            print(f"INFO iteration {iteration + 1}: continuation_alignment_offset={alignment_offset}",
                  flush=True)
            if iteration == 0 or (iteration + 1) % 10 == 0:
                print(f"PASS iteration {iteration + 1}/{args.iterations}", flush=True)
        print("PASS phase2 migration control+KV transfer+two-phase-commit+decode-equivalence", flush=True)
    except BaseException:
        # A rejected reservation, interrupted peer, or bounded RPC timeout
        # must restore the source before shutdown.  Abort calls are best
        # effort here because the original failure is the authoritative one.
        if active_request_id is not None and active_epoch is not None:
            if incremental_target:
                try:
                    await asyncio.wait_for(rpc(args.target_host, args.target_control, {
                        "op": "incremental_abort", "request_id": active_request_id}), timeout=10)
                except Exception:
                    pass
            if incremental_source:
                try:
                    await asyncio.wait_for(rpc(args.source_host, args.source_control, {
                        "op": "incremental_abort", "request_id": active_request_id}), timeout=10)
                except Exception:
                    pass
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
                    await asyncio.wait_for(
                        rpc(args.source_host if port == args.source_control
                            else args.target_host, port, {"op": "shutdown"}),
                        timeout=10,
                    )
                except Exception:
                    process.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.gather(source.wait(), target.wait()), timeout=15)
        except TimeoutError:
            # The PID is owned by this launcher: kill only this exact local
            # child (an SSH child terminates its remote exec session too).
            for process in (source, target):
                if process.returncode is None:
                    process.kill()
            await asyncio.gather(source.wait(), target.wait())
        # ``ssh`` does not guarantee that a remote Python child receives the
        # local channel's termination signal. Clean the exact per-run target
        # wrapper after its local SSH process is no longer needed.
        try:
            await cleanup_remote_target(args.target_ssh, args.target_control)
        except Exception:
            # The test's original result must remain authoritative. A later
            # preflight on the same unique port will reveal any stale worker.
            pass
        for handle in (source_log, target_log):
            if handle is not None:
                handle.close()


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
    parser.add_argument("--incremental-precopy", action="store_true",
                        help="run one explicit immutable-prefix pre-copy round before cutover")
    parser.add_argument("--worker-log-dir", default="",
                        help="write local worker stdout/stderr to this directory")
    parser.add_argument("--rpc-timeout", type=float, default=30.0,
                        help="timeout for bounded control operations")
    parser.add_argument("--inject-latency-ms", type=float, default=0.0,
                        help="delay before each layer transfer (fault injection)")
    parser.add_argument("--inject-target-capacity", action="store_true",
                        help="make target reservation fail deterministically")
    parser.add_argument("--inject-transfer-timeout-after-layers", type=int, default=0,
                        help="raise a deterministic timeout after this many KV layers")
    parser.add_argument("--inject-source-restart-after-layers", type=int, default=0,
                        help="kill only this launcher's source worker after this many layers")
    parser.add_argument("--expect-failure", action="store_true",
                        help="treat a deterministic injected failure as a passing test")
    parser.add_argument("--prompt", default="Explain KV cache migration in one sentence.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sampling temperature; nonzero requires --seed")
    parser.add_argument("--seed", type=int, default=None,
                        help="explicit per-request seed for RNG migration")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        if not args.expect_failure:
            raise
        print(f"EXPECTED_FAILURE {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
