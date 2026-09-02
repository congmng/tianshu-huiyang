"""Minimal Llumnix-compatible HTTP frontend for vLLM 0.11 (V1).

The frontend uses Llumnix's V1 adapter around vLLM's supported ``AsyncLLM``
API and keeps the same ``/health`` and ``/generate`` wire format as the legacy
frontend. This ensures CoreX connector configuration and request-ID handling
are applied consistently with the Manager/Llumlet path when supplied through
``AsyncEngineArgs``; multi-instance orchestration itself remains the Manager's
responsibility.
"""

import argparse
import asyncio
import json
import time
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from llumnix.backends.vllm.v1_engine import V1EngineAdapter


def build_app(engine: V1EngineAdapter) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200)

    @app.post("/generate")
    async def generate(request: Request) -> Response:
        body = await request.json()
        prompt = body.pop("prompt")
        stream = body.pop("stream", False)
        request_id = body.pop("request_id", f"v1-{time.time_ns()}")
        params = SamplingParams(**body)
        results = engine.generate(prompt, params, request_id)

        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for output in results:
                yield (json.dumps({"text": [prompt + x.text for x in output.outputs]}) + "\0").encode()

        if stream:
            return StreamingResponse(stream_results(), media_type="application/octet-stream")

        final = None
        async for output in results:
            if await request.is_disconnected():
                await engine.abort(request_id)
                return Response(status_code=499)
            final = output
        if final is None:
            return JSONResponse({"error": "engine returned no output"}, status_code=500)
        return JSONResponse({"text": [prompt + x.text for x in final.outputs]})

    @app.on_event("shutdown")
    async def shutdown() -> None:
        engine.shutdown()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="Maximum concurrent sequences; 4 avoids sampler warmup OOM on 32 GiB CoreX cards.",
    )
    args, unknown = parser.parse_known_args()
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
    )
    # Keep unknown options available for future V1 argument forwarding rather
    # than silently claiming that legacy Llumnix options are supported.
    if unknown:
        print("Ignoring unsupported legacy/V1 options:", " ".join(unknown))
    engine = V1EngineAdapter(engine_args, instance_id=f"v1-api-{args.port}")
    uvicorn.run(build_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
