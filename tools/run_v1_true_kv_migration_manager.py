#!/usr/bin/env python3
"""Run one real Manager-orchestrated V1 true-KV migration on two GPUs.

This is intentionally a local smoke test rather than a serving endpoint.  It
uses the same ``V1EngineAdapter`` and Llumlet-compatible remote methods that
``Manager._migrate_v1_request`` calls, so it exercises the production control
plane against a real Qwen3-14B / CoreX NCCL data plane.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import ray
from vllm import AsyncEngineArgs, SamplingParams
from vllm.config import KVTransferConfig
from vllm.sampling_params import RequestOutputKind

from llumnix.backends.vllm.v1_engine import V1EngineAdapter
from llumnix.manager import Manager

ROOT = Path(__file__).resolve().parents[1]


@ray.remote(num_gpus=1)
class MigrationInstance:
    def __init__(self, model: str, p2p_host: str, p2p_port: int,
                 role: str, transport: str):
        # EngineCore is spawned by vLLM inside this actor.  The driver-side
        # ``run()`` variables are not inherited across the Ray actor boundary
        # in this CoreX build, so pin the explicit-migration flags here before
        # the adapter constructs its EngineCore subprocess.
        os.environ["VLLM_FORCE_NCCL_COMM"] = "1"
        os.environ["LLUMNIX_TRUE_KV_MIGRATION_ONLY"] = "1"
        os.environ.setdefault("PYTHONHASHSEED", "0")
        self.role = role
        config = KVTransferConfig(
            kv_connector="CoreXP2pNcclConnector",
            kv_connector_module_path="llumnix.backends.vllm.corex_p2p_connector",
            kv_role="kv_producer" if role == "source" else "kv_consumer",
            kv_rank=0,
            kv_parallel_size=2,
            kv_ip=p2p_host,
            kv_port=p2p_port,
            kv_connector_extra_config={
                "corex_transport": transport,
                "send_type": "PUT",
                "true_kv_migration_only": True,
            },
        )
        engine_args = AsyncEngineArgs(
            model=model,
            dtype="float16",
            gpu_memory_utilization=0.96,
            max_model_len=128,
            max_num_seqs=1,
            enforce_eager=True,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="sha256_cbor",
            kv_transfer_config=config,
        )
        self.adapter = V1EngineAdapter(engine_args, f"manager-{role}")
        self.generator = None
        self.active_request_id = None
        self.generated_token_ids: list[int] = []
        self.migration_snapshot = None
        self.migrated_stream = None

    def is_ready(self):
        return True

    def get_kv_endpoint(self):
        return self.adapter.get_kv_endpoint()

    def get_all_request_ids(self):
        return self.adapter.get_all_request_ids()

    def migration_snapshot_output_len(self):
        if self.migration_snapshot is None:
            raise RuntimeError("no prepared migration snapshot")
        return len(self.migration_snapshot.output_token_ids)

    async def baseline(self, request_id: str, prompt: str, temperature: float,
                       seed: int | None) -> list[int]:
        stream = self.adapter.engine.generate(
            prompt,
            SamplingParams(temperature=temperature, seed=seed,
                           max_tokens=64, ignore_eos=True,
                           output_kind=RequestOutputKind.DELTA),
            request_id,
        )
        token_ids = []
        async for output in stream:
            token_ids.extend(output.outputs[0].token_ids)
        return token_ids

    async def generate(self, request_id: str, prompt: str, temperature: float,
                       seed: int | None) -> dict:
        self.active_request_id = request_id
        self.generated_token_ids = []
        stream = self.adapter.engine.generate(
            prompt,
            SamplingParams(temperature=temperature, seed=seed,
                           max_tokens=64, ignore_eos=True,
                           output_kind=RequestOutputKind.DELTA),
            request_id,
        )
        for _ in range(2):
            output = await anext(stream)
            self.generated_token_ids.extend(output.outputs[0].token_ids)
        self.generator = stream
        for _ in range(600):
            if len(self.generated_token_ids) >= 2:
                return {"tokens": len(self.generated_token_ids),
                        "token_ids": self.generated_token_ids}
            await asyncio.sleep(0.05)
        raise TimeoutError("source did not reach migration token boundary")

    async def migration_prepare_out_wire(self, request_id: str,
                                          migration_epoch: int) -> str:
        snapshot = await self.adapter.migration_prepare_out(
            request_id, migration_epoch
        )
        self.migration_snapshot = snapshot
        return self.adapter.encode_migration_snapshot(snapshot)

    async def migration_source_blocks(self, request_id: str,
                                       migration_epoch: int):
        return await self.adapter.migration_source_blocks(
            request_id, migration_epoch
        )

    async def migration_prepare_in_wire(self, snapshot_wire: str):
        snapshot = self.adapter.decode_migration_snapshot(snapshot_wire)
        self.migration_snapshot = snapshot
        return await self.adapter.migration_prepare_in(snapshot)

    async def migration_layer_names(self):
        return await self.adapter.migration_layer_names()

    async def migration_send_layer(self, request_id: str, migration_epoch: int,
                                    layer_name: str, source_block_ids,
                                    target_block_ids, target_endpoint: str,
                                    synced_prefix_block_count: int = 0) -> str:
        manifest = await self.adapter.migration_send_layer(
            request_id, migration_epoch, layer_name, source_block_ids,
            target_block_ids, target_endpoint, synced_prefix_block_count,
        )
        return manifest.decode()

    async def migration_receive_layer(self, request_id: str,
                                       migration_epoch: int,
                                       manifest_wire: str,
                                       source_endpoint: str,
                                       synced_prefix_block_pairs=(),
                                       ) -> None:
        await self.adapter.migration_receive_layer(
            request_id, migration_epoch, manifest_wire.encode(),
            source_endpoint, synced_prefix_block_pairs,
        )

    async def migration_commit(self, request_id: str, migration_epoch: int,
                               incoming: bool = False) -> None:
        await self.adapter.migration_commit(
            request_id, migration_epoch, incoming=incoming
        )

    async def migration_abort(self, request_id: str, migration_epoch: int,
                              incoming: bool = False) -> None:
        await self.adapter.migration_abort(
            request_id, migration_epoch, incoming=incoming
        )

    async def register_migrated_request(self, request_id: str,
                                         snapshot_wire: str,
                                         server_info) -> None:
        snapshot = self.adapter.decode_migration_snapshot(snapshot_wire)
        self.migration_snapshot = snapshot
        self.migrated_stream = self.adapter.add_migrated_request(
            snapshot, server_info
        )

    async def finish_migrated_out(self, request_id: str) -> None:
        if self.generator is not None:
            self.generator = None
        self.adapter.release_request(request_id)

    async def abort_migrated_request(self, request_id: str) -> None:
        try:
            await self.adapter.abort(request_id)
        except Exception:
            pass
        self.adapter.release_request(request_id)

    async def drain_migrated(self, tokens: int) -> list[int]:
        if self.migrated_stream is None:
            raise RuntimeError("no migrated stream registered")
        token_ids = []
        try:
            for _ in range(tokens):
                output = await anext(self.migrated_stream)
                token_ids.extend(output.outputs[0].token_ids)
                if output.finished:
                    break
        finally:
            await self.migrated_stream.aclose()
            await self.adapter.engine.abort(self.migration_snapshot.request_id)
            self.adapter.release_request(self.migration_snapshot.request_id)
        return token_ids

    def shutdown(self):
        self.adapter.shutdown()


async def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_FORCE_NCCL_COMM", "1")
    os.environ["LLUMNIX_TRUE_KV_MIGRATION_ONLY"] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "0")
    source = MigrationInstance.remote(
        args.model, args.source_p2p_host, args.source_p2p, "source",
        args.transport,
    )
    target = MigrationInstance.remote(
        args.model, args.target_p2p_host, args.target_p2p, "target",
        args.transport,
    )
    ray.get([source.is_ready.remote(), target.is_ready.remote()])
    source_endpoint = ray.get(source.get_kv_endpoint.remote())
    target_endpoint = ray.get(target.get_kv_endpoint.remote())
    print(f"ENDPOINTS source={source_endpoint} target={target_endpoint}",
          flush=True)

    source_baseline = ray.get(source.baseline.remote(
        f"{args.request_id}-source-baseline", args.prompt,
        args.temperature, args.seed,
    ))
    target_baseline = ray.get(target.baseline.remote(
        f"{args.request_id}-target-baseline", args.prompt,
        args.temperature, args.seed,
    ))
    generated = ray.get(source.generate.remote(
        args.request_id, args.prompt, args.temperature, args.seed,
    ))

    manager = object.__new__(Manager)
    manager.instances = {"src": source, "dst": target}
    manager.request_server_info = {}
    manager.v1_migrating_requests = set()
    manager.v1_migration_epochs = {}
    manager.v1_migration_retries = {}
    manager.request_instance = {args.request_id: "src"}
    manager.request_instances = {args.request_id: {"src"}}
    await manager._migrate_v1_request(
        "src", "dst", args.request_id, args.epoch,
        source_endpoint, target_endpoint, None,
    )
    continuation = ray.get(source.migration_snapshot_output_len.remote())
    observed = ray.get(target.drain_migrated.remote(args.verify_tokens))

    candidates = []
    for offset in range(0, 3):
        expected = source_baseline[
            continuation + offset: continuation + offset + len(observed)
        ]
        if observed == expected:
            candidates.append((offset, expected))
    if len(candidates) != 1:
        raise AssertionError(
            f"post-migration token mismatch: generated={generated}, "
            f"continuation={continuation}, source_baseline={source_baseline}, "
            f"target_baseline={target_baseline}, observed={observed}"
        )
    print(f"INFO continuation_alignment_offset={candidates[0][0]}", flush=True)
    print("PASS manager_v1_true_kv_migration", flush=True)

    ray.get(source.shutdown.remote())
    ray.get(target.shutdown.remote())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / ".models/Qwen3-14B"))
    parser.add_argument("--transport", choices=("nccl", "zmq_cpu"),
                        default="nccl")
    parser.add_argument("--source-p2p", type=int, default=19001)
    parser.add_argument("--target-p2p", type=int, default=19002)
    parser.add_argument("--source-p2p-host", default="127.0.0.1")
    parser.add_argument("--target-p2p-host", default="127.0.0.1")
    parser.add_argument("--request-id", default="manager-v1-migration")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verify-tokens", type=int, default=4)
    args = parser.parse_args()
    ray.init(num_cpus=2, num_gpus=2, include_dashboard=False,
             ignore_reinit_error=True)
    try:
        asyncio.run(run(args))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
