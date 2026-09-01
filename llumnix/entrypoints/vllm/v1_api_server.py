"""Minimal Llumnix-compatible HTTP frontend for vLLM 0.11 (V1).

This is the first migration seam: request handling uses vLLM's supported V1
``AsyncLLM`` API and keeps the same ``/health`` and ``/generate`` wire format
as the legacy Llumnix frontend. KV-cache migration and multi-instance routing
remain intentionally disabled until their V1 equivalents are implemented.
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
from vllm.v1.engine.async_llm import AsyncLLM


def build_app(engine: AsyncLLM) -> FastAPI:
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
    args, unknown = parser.parse_known_args()
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    # Keep unknown options available for future V1 argument forwarding rather
    # than silently claiming that legacy Llumnix options are supported.
    if unknown:
        print("Ignoring unsupported legacy/V1 options:", " ".join(unknown))
    engine = AsyncLLM.from_engine_args(engine_args)
    uvicorn.run(build_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
