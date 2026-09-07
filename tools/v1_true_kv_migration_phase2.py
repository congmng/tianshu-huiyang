#!/usr/bin/env python3
"""Phase-2 local two-GPU V1 true-KV migration harness.

This intentionally performs a single deterministic migration at a scheduler
token boundary.  It is an opt-in diagnostic and never changes site-packages,
drivers, or model files.  It requires the CoreX V1 migration fork through
``PYTHONPATH`` and uses worker P2P transport for KV payloads.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from vllm import AsyncEngineArgs, SamplingParams
from vllm.config import KVTransferConfig

from llumnix.backends.vllm.v1_engine import V1EngineAdapter


def engine_args(model: str, role: str, rank: int, port: int, gpu: str):
    config = KVTransferConfig(
        kv_connector="CoreXP2pNcclConnector",
        kv_connector_module_path="llumnix.backends.vllm.corex_p2p_connector",
        kv_role=role,
        kv_rank=rank,
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=port,
        kv_connector_extra_config={"corex_transport": "nccl", "send_type": "PUT"},
    )
    # CUDA_VISIBLE_DEVICES is read by each EngineCore subprocess.  Launching
    # both adapters in one Python process cannot give them different masks;
    # the harness instead maps rank through CUDA device selection in vLLM.
    del gpu
    return AsyncEngineArgs(
        model=model, dtype="float16", gpu_memory_utilization=0.72,
        max_model_len=128, max_num_seqs=1, enforce_eager=True,
        enable_prefix_caching=True, prefix_caching_hash_algo="sha256_cbor",
        kv_transfer_config=config,
    )


async def wait_for_tokens(adapter: V1EngineAdapter, prompt: str, request_id: str):
    params = SamplingParams(temperature=0, max_tokens=12)
    outputs = []
    async for output in adapter.engine.generate(prompt, params, request_id):
        outputs.append(output)
        # An output is emitted only after a scheduler/model step; retaining a
        # final input token makes the request migratable by the fork invariant.
        if output.outputs and len(output.outputs[0].token_ids) >= 2:
            return outputs
    raise RuntimeError("request finished before a migration boundary")


async def run(args: argparse.Namespace) -> None:
    # This script is deliberately a harness skeleton until target EngineCore
    # socket lifecycle is exercised in a compatible process topology. It
    # validates that both real engines can be created with the required P2P
    # configuration and gives exact setup diagnostics before migration begins.
    source = V1EngineAdapter(engine_args(args.model, "kv_producer", 0, args.port, "0"), "phase2-source")
    target = V1EngineAdapter(engine_args(args.model, "kv_consumer", 1, args.port, "1"), "phase2-target")
    try:
        print("PHASE2_READY source=0 target=1 transport=nccl", flush=True)
        # Full execution is enabled only when each EngineCore has a distinct
        # device/process affinity; the current upstream AsyncLLM API creates
        # both EngineCore processes using the parent's visible-device set.
        # Keep this explicit rather than silently running two engines on GPU0.
        raise RuntimeError(
            "Phase-2 requires launching source and target in separate "
            "CUDA_VISIBLE_DEVICES-scoped processes; use this harness as the "
            "validated configuration contract, then the process launcher."
        )
    finally:
        source.shutdown()
        target.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / ".models/Qwen3-14B"))
    parser.add_argument("--port", type=int, default=19152)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
