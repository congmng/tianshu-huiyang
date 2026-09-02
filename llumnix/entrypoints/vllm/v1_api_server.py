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
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from llumnix.backends.vllm.v1_engine import V1EngineAdapter


def build_app(engine: V1EngineAdapter) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        engine.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200)

    @app.post("/generate")
    async def generate(request: Request) -> Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            prompt = body.pop("prompt")
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            stream = body.pop("stream", False)
            if not isinstance(stream, bool):
                raise ValueError("stream must be a boolean")
            request_id = body.pop("request_id", f"v1-{time.time_ns()}")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request_id must be a non-empty string")
            params = SamplingParams(**body)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid generate request: {exc}") from exc
        results = engine.generate(prompt, params, request_id)

        async def stream_results() -> AsyncGenerator[bytes, None]:
            completed = False
            try:
                async for output in results:
                    yield (json.dumps({"text": [prompt + x.text for x in output.outputs]}) + "\0").encode()
                completed = True
            finally:
                # Starlette closes the iterator when a client disconnects.
                # Abort the V1 request in that path so EngineCore does not
                # retain a producer or consumer stream indefinitely.
                if not completed:
                    await engine.abort(request_id)
                else:
                    release = getattr(engine, "release_request", None)
                    if release is not None:
                        release(request_id)

        if stream:
            return StreamingResponse(stream_results(), media_type="application/octet-stream")

        final = None
        async for output in results:
            if await request.is_disconnected():
                await engine.abort(request_id)
                return Response(status_code=499)
            final = output
        if final is None:
            release = getattr(engine, "release_request", None)
            if release is not None:
                release(request_id)
            return JSONResponse({"error": "engine returned no output"}, status_code=500)
        release = getattr(engine, "release_request", None)
        if release is not None:
            release(request_id)
        return JSONResponse({"text": [prompt + x.text for x in final.outputs]})

    return app


def build_arg_parser():
    # Register all vLLM V1 options.  Keeping only a small hand-written subset
    # silently discarded deployment settings such as quantization and KV IO.
    parser = FlexibleArgumentParser(description=__doc__)
    AsyncEngineArgs.add_cli_args(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.set_defaults(dtype="float16", gpu_memory_utilization=0.70,
                        max_model_len=4096, max_num_seqs=4, enforce_eager=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = V1EngineAdapter(engine_args, instance_id=f"v1-api-{args.port}")
    uvicorn.run(build_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
