# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import AsyncGenerator
from contextlib import asynccontextmanager
import time
import asyncio
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import subprocess
import os
import math

from vllm.sampling_params import SamplingParams

from llumnix.arg_utils import LlumnixArgumentParser, LaunchArgs
from llumnix.entrypoints.setup import setup_ray_cluster, setup_llumnix
from llumnix.entrypoints.vllm.arg_utils import add_cli_args, get_args
from llumnix.entrypoints.vllm.client import LlumnixClientVLLM
from llumnix.logging.logger import init_logger
from llumnix.utils import random_uuid
from llumnix.config import get_llumnix_config
from llumnix.backends.backend_interface import BackendType
from llumnix.entrypoints.utils import LaunchMode, is_gpu_available
from llumnix.constants import SERVER_TIMEOUT_KEEP_ALIVE
from llumnix.metrics.timestamps import set_timestamp
from llumnix.entrypoints.vllm.web.types import (
    InferenceInstanceInfo,
    APIResponse,
    BenchmarkRequest,
)
from llumnix.entrypoints.vllm.web.utils import generate_bench_command

# Code file with __main__ should set the logger name to inherit the llumnix logger configuration.
logger = init_logger("llumnix.entrypoints.vllm.api_server")

llumnix_client: LlumnixClientVLLM = None


# pylint: disable=unused-argument
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    asyncio.create_task(llumnix_client.request_output_queue.run_server_loop())
    asyncio.create_task(llumnix_client.get_request_outputs_loop())
    yield
    llumnix_client.request_output_queue.cleanup()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    try:
        request_dict = await request.json()
        if not isinstance(request_dict, dict):
            raise ValueError("request body must be a JSON object")
        prompt = request_dict.pop("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        stream = request_dict.pop("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean")
        request_id = request_dict.pop("request_id", random_uuid())
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        sampling_params = SamplingParams(**request_dict)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid generate request: {exc}") from exc

    # Use LlumnixClientVLLM's generate and abort api to replace with vLLM AsyncLLMEngine's generate and abort api.
    results_generator = await llumnix_client.generate(
        prompt, sampling_params, request_id
    )

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        completed = False
        try:
            async for request_output in results_generator.generator():
                prompt = request_output.prompt
                text_outputs = [prompt + output.text for output in request_output.outputs]
                ret = {"text": text_outputs}
                yield (json.dumps(ret) + "\0").encode("utf-8")
            completed = True
        finally:
            # A disconnected streaming client must not leave a V1 producer or
            # P/D consumer active behind the Manager.
            if not completed:
                await llumnix_client.abort(request_id)

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator.generator():
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await llumnix_client.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


@app.post("/generate_benchmark")
async def generate_benchmark(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    # Add some benchmark-related codes comparing to the generate API.
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    _ = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    start = time.time()

    results_generator = await llumnix_client.generate(
        prompt, sampling_params, request_id
    )

    # Non-streaming case
    final_output = None
    per_token_latency = []
    per_token_latency_breakdown_list = []
    async for request_output in results_generator.generator():
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await llumnix_client.abort(request_id)
            return Response(status_code=499)
        now = time.time()
        per_token_latency.append([now, (now - start) * 1000])
        start = now
        final_output = request_output
        set_timestamp(request_output, "api_server_generate_timestamp_end", now)
        if hasattr(request_output, "request_timestamps"):
            per_token_latency_breakdown_list.append(
                request_output.request_timestamps.to_latency_breakdown_dict()
            )
    assert final_output is not None

    if llumnix_client.log_requests:
        llumnix_client.num_finished_requests += 1
        logger.info("entrypoints finished request {}".format(request_id))
        logger.info(
            "num_finished_requests {}".format(llumnix_client.num_finished_requests)
        )

    generation = final_output.outputs[0].text
    num_output_tokens = len(final_output.outputs[0].token_ids)
    num_input_tokens = len(final_output.prompt_token_ids)
    expected_resp_len = request_dict["max_tokens"]
    if not max(expected_resp_len, 1) == max(num_output_tokens, 1):
        "request_id={}, expected_resp_len={}, num_output_tokens={}, num_input_tokens={}".format(
            request_id, expected_resp_len, num_output_tokens, num_input_tokens
        )
    ret = {
        "request_id": request_id,
        "generated_text": generation,
        "num_output_tokens_cf": num_output_tokens,
        "per_token_latency": per_token_latency,
    }
    if per_token_latency_breakdown_list:
        ret["per_token_latency_breakdown_list"] = per_token_latency_breakdown_list
    return JSONResponse(ret)


@app.get("/is_ready")
async def is_ready() -> bool:
    return await llumnix_client.is_ready()


@app.get("/instance_list")
async def get_instance_list() -> Response:
    """Get the list of the instance."""
    # 使用manager的get_all_instances_info API获取所有instance信息
    instance_infos = await llumnix_client.get_all_instances_info()

    # 转换为InferenceInstanceInfo格式
    inference_infos = []
    for instance_info in instance_infos:
        inference_info = InferenceInstanceInfo(
            instance_id=instance_info.instance_id,
            node_id=getattr(instance_info, "node_id", ""),
            node_ip=getattr(instance_info, "node_ip", ""),
            gpu_count=max(int(getattr(instance_info, "gpu_count", 1)), 1),
            request_count=instance_info.num_running_requests
            + instance_info.num_waiting_requests,
            running_request_count=instance_info.num_running_requests,
            waiting_request_count=instance_info.num_waiting_requests,
            total_gpu_blocks_count=instance_info.num_total_gpu_blocks,
            used_gpu_blocks_count=instance_info.num_used_gpu_blocks,
            waiting_gpu_blocks_count=instance_info.num_blocks_all_waiting_requests,
            gpu_memory_total_bytes=instance_info.gpu_memory_total_bytes,
            gpu_memory_free_bytes=instance_info.gpu_memory_free_bytes,
            compute_capacity=instance_info.compute_capacity,
            # ``-inf`` is the internal sentinel for an idle instance (and is
            # intentionally the best value for the load dispatcher). JSON
            # does not permit non-finite floats, so expose it as ``null``
            # without changing the scheduler's comparison semantics.
            dispatch_load_metric=(
                instance_info.dispatch_load_metric
                if math.isfinite(instance_info.dispatch_load_metric)
                else None
            ),
            kv_cache_affinity_blocks=len(instance_info.kv_cache_block_hashes),
        )
        inference_infos.append(inference_info)

    return JSONResponse(
        APIResponse(code=0, message="success", data=inference_infos).model_dump()
    )


@app.get("/instance_log/{instance_id}")
async def get_instance_log(instance_id: str) -> Response:
    """Get the log of the instance."""
    LOG_FILE = f"/home/l1hy/project/llumnix/vllm.serve.vu.log"
    with open(LOG_FILE, "r") as f:
        instance_log = f.read()
    return JSONResponse(
        APIResponse(code=0, message="success", data=instance_log).model_dump()
    )


@app.post("/benchmark")
async def benchmark_start(request: Request) -> Response:
    """Benchmark the instance."""
    request_dict = await request.json()
    qps = request_dict.pop("qps")
    num_prompts = request_dict.pop("num_prompts")

    # 使用时间戳和随机字符串生成benchmark_id
    timestamp = int(time.time())
    random_str = random_uuid()[:8]  # 取随机字符串的前8位
    benchmark_id = f"{timestamp}_{random_str}"

    bench_command = generate_bench_command(
        ip_ports=f"10.212.70.38:37000",
        model="/data/model/Qwen2.5-3B",
        num_prompts=num_prompts,
        dataset_type="sharegpt",
        dataset_path="/data/dataset/sharegpt4/sharegpt_gpt4.jsonl",
        qps=qps,
        results_filename=f"{benchmark_id}.out",
        verbose=False,
    )

    try:
        # 直接启动进程，将输出重定向到DEVNULL
        process = subprocess.Popen(
            bench_command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 立即返回结果
        return JSONResponse(
            APIResponse(
                code=0,
                message="success",
                data={
                    "benchmark_id": benchmark_id,
                    "command": bench_command,
                    "pid": process.pid,
                },
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            APIResponse(
                code=1,
                message=f"Failed to start benchmark: {str(e)}",
                data={
                    "benchmark_id": benchmark_id,
                    "command": bench_command,
                    "pid": 0,
                },
            ).model_dump()
        )


@app.get("/benchmark_result/{benchmark_id}")
async def benchmark_result(benchmark_id: str) -> Response:
    """Get the benchmark result for the given benchmark_id."""
    # Get the llumnix root directory
    llumnix_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    result_file = os.path.join(llumnix_root, f"bench_{benchmark_id}.out")
    try:
        with open(result_file, "r") as f:
            result_content = f.read()
        return JSONResponse(
            APIResponse(code=0, message="success", data=result_content).model_dump()
        )
    except FileNotFoundError:
        return JSONResponse(
            APIResponse(
                code=1,
                message=f"Benchmark result file {result_file} not found",
                data=None,
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            APIResponse(
                code=1,
                message=f"Error reading benchmark result: {str(e)}",
                data=None,
            ).model_dump()
        )


if __name__ == "__main__":
    parser: LlumnixArgumentParser = LlumnixArgumentParser()

    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--ssl-keyfile", type=str)
    parser.add_argument("--ssl-certfile", type=str)
    parser.add_argument(
        "--log-level", type=str, choices=["debug", "info", "warning", "error"]
    )

    cli_args = add_cli_args(parser)
    cfg = get_llumnix_config(cli_args.config_file, cli_args)

    entrypoints_args, manager_args, instance_args, engine_args = get_args(
        cfg, LaunchMode.LOCAL, parser, cli_args
    )
    backend_type = (
        BackendType.VLLM if not instance_args.simulator_mode else BackendType.SIM_VLLM
    )
    launch_args = LaunchArgs(launch_mode=LaunchMode.LOCAL, backend_type=backend_type)

    # Launch or connect to the ray cluster for multi-node serving.
    setup_ray_cluster(entrypoints_args)

    # if gpu is not available, it means that this node is head pod without any llumnix components
    if is_gpu_available():
        entrypoints_context = setup_llumnix(
            entrypoints_args, manager_args, instance_args, engine_args, launch_args
        )
        llumnix_client = LlumnixClientVLLM(entrypoints_context)

        # Start the api server after all the components of llumnix are ready.
        logger.info(
            "Start api server on '{}:{}'.".format(
                entrypoints_args.host, entrypoints_args.port
            )
        )
        uvicorn.run(
            app,
            host=entrypoints_args.host,
            port=entrypoints_args.port,
            log_level=entrypoints_args.log_level,
            timeout_keep_alive=SERVER_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=entrypoints_args.ssl_keyfile,
            ssl_certfile=entrypoints_args.ssl_certfile,
        )
