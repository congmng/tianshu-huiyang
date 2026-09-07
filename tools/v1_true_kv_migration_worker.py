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
from vllm.v1.migration import RequestMigrationSnapshot, deserialize_greedy_sampling_params
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.output_processor import RequestOutputCollector
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
            prompt, SamplingParams(temperature=0, max_tokens=12,
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
                value["prompt"], SamplingParams(temperature=0, max_tokens=value.get("max_tokens", 12),
                                                  output_kind=RequestOutputKind.DELTA),
                value.get("request_id", "phase2-baseline"),
            ):
                token_ids.extend(output.outputs[0].token_ids)
            return {"token_ids": token_ids}
        if op == "prepare_out":
            snapshot = await self.adapter.migration_prepare_out(value["request_id"], value["epoch"])
            # EngineCore utility responses cross a msgspec boundary.  Frozen
            # dataclasses are decoded there as plain dictionaries, whereas
            # in-process/unit callers receive RequestMigrationSnapshot.
            # Normalize both forms before choosing the versioned JSON wire.
            if isinstance(snapshot, dict):
                snapshot = RequestMigrationSnapshot(
                    **{
                        **snapshot,
                        "prompt_token_ids": tuple(snapshot["prompt_token_ids"]),
                        "all_token_ids": tuple(snapshot["all_token_ids"]),
                        "output_token_ids": tuple(snapshot["output_token_ids"]),
                        "kv_group_block_counts": tuple(snapshot["kv_group_block_counts"]),
                        "feature_flags": tuple(snapshot["feature_flags"]),
                    }
                )
            return {"snapshot": snapshot.to_wire().decode(),
                    "output_token_ids": list(snapshot.output_token_ids),
                    "all_token_ids": list(snapshot.all_token_ids),
                    "num_computed_tokens": snapshot.num_computed_tokens}
        if op == "prepare_in":
            snapshot = RequestMigrationSnapshot.from_wire(value["snapshot"].encode())
            self.migration_snapshot = snapshot
            blocks = await self.adapter.migration_prepare_in(snapshot)
            return {"blocks": blocks}
        if op == "blocks":
            return {"blocks": await self.adapter.migration_source_blocks(value["request_id"], value["epoch"])}
        if op == "layers":
            return {"layers": await self.adapter.migration_layer_names()}
        if op == "send":
            wire = await self.adapter.migration_send_layer(
                value["request_id"], value["epoch"], value["layer"], value["source_blocks"],
                value["target_blocks"], value["peer"],
            )
            return {"manifest": wire.decode()}
        if op == "receive":
            await self.adapter.migration_receive_layer(value["request_id"], value["epoch"],
                                                       value["manifest"].encode(), value["peer"])
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
            params = deserialize_greedy_sampling_params(snap.sampling_params)
            req = EngineCoreRequest(
                request_id=snap.request_id,
                prompt_token_ids=list(snap.prompt_token_ids), mm_features=None,
                sampling_params=params, pooling_params=None,
                eos_token_id=snap.eos_token_id, arrival_time=0.0,
                lora_request=None, cache_salt=None, data_parallel_rank=None,
            )
            req.sampling_params.output_kind = RequestOutputKind.DELTA
            self.migration_queue = RequestOutputCollector(RequestOutputKind.DELTA)
            self.adapter.engine._run_output_handler()
            self.adapter.engine.output_processor.add_request(
                req, value.get("prompt", ""), None, 0, self.migration_queue)
            token_ids = []
            for _ in range(int(value.get("tokens", 1))):
                out = await self.migration_queue.get()
                token_ids.extend(out.outputs[0].token_ids)
                if out.finished:
                    break
            # The probe intentionally stops after a short continuation. Clean
            # up both frontend and EngineCore state so repeated iterations do
            # not accumulate live requests or KV blocks.
            await self.adapter.engine.abort(snap.request_id)
            self.migration_queue = None
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
    args = parser.parse_args()
    os.environ.setdefault("PYTHONHASHSEED", "0")
    asyncio.run(serve(Phase2Worker(args), args.control_host, args.control_port))


if __name__ == "__main__":
    main()
