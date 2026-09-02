#!/usr/bin/env python3
"""Minimal two-host vLLM V1 P2P KV-transfer probe.

Run one producer and one consumer with the same model. The producer performs
only the prefill handoff; the consumer imports that KV and produces the public
text. This tool does not download models or modify drivers/system packages.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from vllm import AsyncEngineArgs, SamplingParams
from vllm.config import KVTransferConfig

from llumnix.backends.vllm.v1_kv_transfer import (
    decorate_p2p_pd_request_id,
    producer_sampling_params,
)


async def run(args: argparse.Namespace) -> None:
    config = KVTransferConfig(
        kv_connector="CoreXP2pNcclConnector",
        kv_connector_module_path="llumnix.backends.vllm.corex_p2p_connector",
        kv_role="kv_producer" if args.role == "producer" else "kv_consumer",
        kv_rank=0 if args.role == "producer" else 1,
        kv_parallel_size=2,
        kv_ip=args.host,
        kv_port=args.port,
        # The probe exits after the producer's first output. Use synchronous
        # transport so the worker cannot be torn down before its async queue
        # hands the KV tensors to the consumer.
        kv_connector_extra_config={"send_type": "PUT"},
    )
    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=True,
        prefix_caching_hash_algo="sha256_cbor",
        kv_transfer_config=config,
    )
    from llumnix.backends.vllm.v1_engine import V1EngineAdapter

    engine = V1EngineAdapter(engine_args, instance_id=f"probe-{args.role}")
    public_id = args.request_id
    local_endpoint = f"{args.host}:{args.port}"
    if args.role == "producer":
        request_id = decorate_p2p_pd_request_id(public_id, args.peer, local_endpoint)
        params = producer_sampling_params(SamplingParams(max_tokens=args.max_tokens, temperature=0.0))
    else:
        request_id = decorate_p2p_pd_request_id(public_id, local_endpoint, args.peer)
        params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    print(f"START role={args.role} endpoint={args.host}:{args.port} request_id={request_id}", flush=True)
    try:
        async for output in engine.engine.generate(args.prompt, params, request_id):
            text = output.outputs[0].text if output.outputs else ""
            print(f"OUTPUT role={args.role} finished={output.finished} text={text!r}", flush=True)
    finally:
        engine.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("producer", "consumer"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", required=True, help="This host's routable IP")
    parser.add_argument("--peer", required=True, help="Peer host:base-port")
    parser.add_argument("--port", type=int, default=19052)
    parser.add_argument(
        "--prompt",
        default=(
            "KV cache stores attention keys and values for already processed "
            "tokens so distributed inference can reuse the prompt state. "
        ) * 12,
        help="Use a prompt longer than one KV block to exercise P2P transfer.",
    )
    parser.add_argument("--request-id", default="llumnix-p2p-model-probe")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        asyncio.run(run(parse_args()))
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
