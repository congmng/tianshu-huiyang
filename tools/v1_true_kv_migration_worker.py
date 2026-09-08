#!/usr/bin/env python3
"""One CUDA-scoped worker for the V1 Phase-2 true-KV migration probe.

Run this program once per GPU with distinct ``CUDA_VISIBLE_DEVICES``, P2P,
and control ports. It is deliberately a diagnostic control plane, not a
serving API. Commands are newline-delimited JSON over loopback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from vllm import AsyncEngineArgs, SamplingParams
from vllm.config import KVTransferConfig
from vllm.v1.migration import RequestMigrationSnapshot
from vllm.sampling_params import RequestOutputKind

from llumnix.backends.vllm.v1_engine import V1EngineAdapter


class Phase2Worker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        config = KVTransferConfig(
            kv_connector="CoreXP2pNcclConnector",
            kv_connector_module_path="llumnix.backends.vllm.corex_p2p_connector",
            kv_role="kv_producer" if args.role == "source" else "kv_consumer",
            kv_rank=0, kv_parallel_size=2, kv_ip=args.p2p_host, kv_port=args.p2p_port,
            kv_connector_extra_config={
                "corex_transport": args.transport,
                "send_type": "PUT",
                # The connector still owns the P2P engine/endpoints, but
                # normal P/D hooks must not transfer an unfrozen request.
                "true_kv_migration_only": True,
            },
        )
        engine_args = AsyncEngineArgs(
            model=args.model, dtype="float16", gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len, max_num_seqs=1, enforce_eager=True,
            enable_prefix_caching=True, prefix_caching_hash_algo="sha256_cbor",
            kv_transfer_config=config,
        )
        self.adapter = V1EngineAdapter(engine_args, f"phase2-{args.role}")
        self.generator = None
        self.token_count = 0
        self.generated_token_ids: list[int] = []
        self.migration_snapshot = None
        self.migration_queue = None
        self.active_request_id: str | None = None

    async def generate(self, request_id: str, prompt: str) -> dict:
        self.token_count = 0
        self.generated_token_ids = []
        # The explicit V1 migration commands below carry peer addresses as
        # RPC arguments.  Do not encode legacy P/D routing markers into this
        # request ID: even with empty connector metadata, CoreX-specific P/D
        # paths can inspect those markers while scheduling the source decode.
        # Keeping the EngineCore ID equal to the public migration ID is also
        # what makes snapshot/commit identity unambiguous across hosts.
        internal_request_id = request_id
        self.active_request_id = internal_request_id
        # Register the normal frontend request, then hold its stream open
        # until EngineCore has emitted the migration boundary.  Unlike a task
        # that blocks inside ``async for``, this leaves no orphaned generator
        # to race the next iteration after source commit.
        stream = self.adapter.engine.generate(
            prompt, SamplingParams(temperature=self.args.temperature, seed=self.args.seed,
                                   max_tokens=self.args.max_model_len,
                                   ignore_eos=True,
                                   output_kind=RequestOutputKind.DELTA), internal_request_id)
        for _ in range(2):
            output = await anext(stream)
            token_ids = output.outputs[0].token_ids
            self.token_count += len(token_ids)
            self.generated_token_ids.extend(token_ids)
            if self.token_count >= 2:
                break
        self.generator = stream
        for _ in range(600):
            if self.token_count >= 2:
                return {"tokens": self.token_count, "token_ids": self.generated_token_ids,
                        "request_id": internal_request_id}
            await asyncio.sleep(0.05)
        raise TimeoutError("source did not reach migration token boundary")

    async def cleanup_source_generator(self) -> None:
        """Discard a stream whose EngineCore request was committed away."""
        if self.generator is None:
            return
        # commit_migration_out has already removed the request in EngineCore.
        # Calling AsyncLLM.abort/Generator.aclose here can wait forever for an
        # output that can no longer arrive.  The stream owns no running task
        # (we consumed it synchronously through anext), so dropping it is the
        # correct teardown for this migration-only probe.
        self.generator = None
        self.active_request_id = None

    async def command(self, value: dict) -> dict:
        op = value["op"]
        if op == "generate":
            return await self.generate(value["request_id"], value["prompt"])
        if op == "baseline":
            token_ids = []
            async for output in self.adapter.engine.generate(
                value["prompt"], SamplingParams(temperature=self.args.temperature,
                                                  seed=self.args.seed,
                                                  max_tokens=value.get("max_tokens", 12),
                                                  output_kind=RequestOutputKind.DELTA),
                value.get("request_id", "phase2-baseline"),
            ):
                token_ids.extend(output.outputs[0].token_ids)
            return {"token_ids": token_ids}
        if op == "prepare_out":
            snapshot = await self.adapter.migration_prepare_out(value["request_id"], value["epoch"])
            return {"snapshot": snapshot.to_wire().decode(),
                    "output_token_ids": list(snapshot.output_token_ids),
                    "all_token_ids": list(snapshot.all_token_ids),
                    "num_computed_tokens": snapshot.num_computed_tokens}
        if op == "prepare_in":
            if os.getenv("LLUMNIX_INJECT_TARGET_CAPACITY", "0") in {"1", "true", "TRUE"}:
                raise RuntimeError("injected target capacity exhaustion")
            snapshot = RequestMigrationSnapshot.from_wire(value["snapshot"].encode())
            self.migration_snapshot = snapshot
            blocks = await self.adapter.migration_prepare_in(snapshot)
            return {"blocks": blocks}
        if op == "blocks":
            return {"blocks": await self.adapter.migration_source_blocks(value["request_id"], value["epoch"])}
        if op == "layers":
            return {"layers": await self.adapter.migration_layer_names()}
        if op == "incremental_begin":
            wire = await self.adapter.begin_incremental_migration(
                value["request_id"], value["epoch"])
            return {"session": wire.decode()}
        if op == "incremental_immutable_blocks":
            blocks, counts = await self.adapter.incremental_migration_immutable_blocks(
                value["request_id"])
            return {"blocks": blocks, "counts": counts}
        if op == "incremental_prepare_in":
            blocks = await self.adapter.prepare_incremental_migration_in(
                value["session"].encode(), tuple(value["counts"]))
            return {"blocks": blocks}
        if op == "incremental_preview":
            wire = await self.adapter.preview_incremental_migration(
                value["request_id"], value["epoch"],
                tuple(tuple(pair) for pair in value["pairs"]))
            return {"session": wire.decode()}
        if op == "incremental_send":
            wire = await self.adapter.send_incremental_migration_kv_layer(
                value["session"].encode(), value["layer"],
                value["source_blocks"], value["target_blocks"], value["peer"])
            return {"manifest": wire.decode()}
        if op == "incremental_receive":
            await self.adapter.receive_incremental_migration_kv_layer(
                value["session"].encode(), value["manifest"].encode(), value["peer"])
            return {}
        if op == "incremental_commit_in":
            wire = await self.adapter.commit_incremental_migration_in(
                value["session"].encode())
            return {"session": wire.decode()}
        if op == "incremental_append":
            wire = await self.adapter.append_incremental_migration(
                value["request_id"], value["epoch"],
                tuple(tuple(pair) for pair in value["pairs"]))
            return {"session": wire.decode()}
        if op == "incremental_abort":
            await self.adapter.abort_incremental_migration(value["request_id"])
            return {}
        if op == "send":
            wire = await self.adapter.migration_send_layer(
                value["request_id"], value["epoch"], value["layer"], value["source_blocks"],
                value["target_blocks"], value["peer"], value.get("synced_prefix_block_count", 0),
            )
            return {"manifest": wire.decode()}
        if op == "receive":
            await self.adapter.migration_receive_layer(value["request_id"], value["epoch"],
                                                       value["manifest"].encode(), value["peer"],
                                                       tuple(tuple(pair) for pair in value.get("synced_prefix_block_pairs", ())))
            return {}
        if op == "commit":
            await self.adapter.migration_commit(value["request_id"], value["epoch"], value["incoming"])
            if not value["incoming"] and self.generator is not None:
                await self.cleanup_source_generator()
            return {}
        if op == "resume":
            if self.migration_snapshot is None:
                raise RuntimeError("no prepared migration snapshot")
            snap = self.migration_snapshot
            stream = self.adapter.add_migrated_request(snap, None)
            token_ids = []
            try:
                for _ in range(int(value.get("tokens", 1))):
                    out = await anext(stream)
                    token_ids.extend(out.outputs[0].token_ids)
                    if out.finished:
                        break
            finally:
                await stream.aclose()
                # The probe intentionally stops after a short continuation.
                # Clean up both frontend and EngineCore state so repeated
                # iterations do not accumulate live requests or KV blocks.
                await self.adapter.engine.abort(snap.request_id)
                self.adapter.release_request(snap.request_id)
            return {"token_ids": token_ids}
        if op == "abort":
            await self.adapter.migration_abort(value["request_id"], value["epoch"], value["incoming"])
            return {}
        if op == "shutdown":
            await self.cleanup_source_generator()
            self.adapter.shutdown()
            return {"shutdown": True}
        raise ValueError(f"unknown op: {op}")


async def serve(worker: Phase2Worker, host: str, port: int) -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            raw = await reader.readline()
            try:
                result = await worker.command(json.loads(raw))
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
    server = await asyncio.start_server(handler, host, port)
    print(f"READY role={worker.args.role} control={host}:{port}", flush=True)
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("source", "target"), required=True)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--p2p-port", type=int, required=True)
    parser.add_argument("--p2p-host", default="127.0.0.1",
                        help="advertised/bound routable P2P address")
    parser.add_argument("--peer-p2p", type=int, required=True)
    parser.add_argument("--peer-p2p-host", default="127.0.0.1",
                        help="peer's advertised routable P2P address")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / ".models/Qwen3-14B"))
    parser.add_argument("--transport", choices=("nccl", "zmq_cpu"), default="nccl")
    # Qwen3-14B FP16 weights occupy about 27.5GiB on a 32GiB BI-V150;
    # leave the remaining ~5GiB for the short Phase-2 KV cache.
    parser.add_argument("--gpu-memory-utilization", type=float, default=.96)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    os.environ.setdefault("PYTHONHASHSEED", "0")
    asyncio.run(serve(Phase2Worker(args), args.control_host, args.control_port))


if __name__ == "__main__":
    main()
