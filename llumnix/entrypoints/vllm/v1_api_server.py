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
import socket
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

    @app.get("/is_ready")
    async def is_ready() -> bool:
        """Keep the main Llumnix API's readiness contract for V1 deployments."""
        from llumnix.backends.backend_interface import EngineState

        return getattr(engine, "state", None) != EngineState.STOPPED

    @app.get("/instance_list")
    async def instance_list() -> JSONResponse:
        """Expose the same CoreX topology/load fields as the managed API.

        The standalone V1 server has one logical instance, but it can own a
        TP/PP group.  Build the response from the adapter's normal
        ``update_instance_info`` path so memory, GPU count and cached-prefix
        state stay consistent with Manager/Llumlet deployments.
        """
        from llumnix.instance_info import InstanceInfo

        info = InstanceInfo(instance_id=getattr(engine, "instance_id", "v1-api"))
        engine.update_instance_info(info)
        # Standalone V1 uses no Ray actor, so it cannot populate topology from
        # Ray's runtime context. Publish the local node identity explicitly;
        # this keeps independent per-host V1 servers observable by the same
        # cross-domain tooling as Manager/Llumlet deployments.
        info.node_id = socket.gethostname()
        try:
            info.node_ip = socket.gethostbyname(info.node_id)
        except OSError:
            info.node_ip = ""
        return JSONResponse({"data": [{
            "instance_id": info.instance_id,
            "node_id": getattr(info, "node_id", ""),
            "node_ip": getattr(info, "node_ip", ""),
            "gpu_count": max(int(getattr(info, "gpu_count", 1)), 1),
            "request_count": info.num_running_requests + info.num_waiting_requests,
            "running_request_count": info.num_running_requests,
            "waiting_request_count": info.num_waiting_requests,
            "total_gpu_blocks_count": info.num_total_gpu_blocks,
            "used_gpu_blocks_count": info.num_used_gpu_blocks,
            "waiting_gpu_blocks_count": info.num_blocks_all_waiting_requests,
            "gpu_memory_total_bytes": info.gpu_memory_total_bytes,
            "gpu_memory_free_bytes": info.gpu_memory_free_bytes,
            "compute_capacity": info.compute_capacity,
            "kv_cache_affinity_blocks": len(info.kv_cache_block_hashes),
        }]})

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

    @app.post("/generate_benchmark")
    async def generate_benchmark(request: Request) -> Response:
        """Return generation text plus per-token timing for benchmark clients."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            prompt = body.pop("prompt")
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            body.pop("stream", False)
            request_id = body.pop("request_id", f"bench-{time.time_ns()}")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request_id must be a non-empty string")
            params = SamplingParams(**body)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid benchmark request: {exc}") from exc

        started = time.monotonic()
        timestamps = []
        final = None
        results = engine.generate(prompt, params, request_id)
        try:
            async for output in results:
                now = time.monotonic()
                timestamps.append([now, (now - started) * 1000.0])
                final = output
        except Exception:
            await engine.abort(request_id)
            raise
        finally:
            release = getattr(engine, "release_request", None)
            if release is not None:
                release(request_id)
        if final is None or not final.outputs:
            return JSONResponse({"error": "engine returned no output"}, status_code=500)
        completion = final.outputs[0]
        token_ids = getattr(completion, "token_ids", ()) or ()
        return JSONResponse({
            "request_id": request_id,
            "generated_text": completion.text,
            "num_output_tokens_cf": len(token_ids),
            "num_input_tokens": 0,
            "per_token_latency": timestamps,
        })

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
