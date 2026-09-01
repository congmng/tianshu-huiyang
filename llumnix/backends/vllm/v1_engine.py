"""Small vLLM V1 adapter used while the legacy KV-migration backend is ported.

vLLM 0.11 moved its serving engine to ``vllm.v1.engine.AsyncLLM``.  This
adapter deliberately exposes the subset needed by Llumnix's request router,
without importing removed 0.6.x private classes.
"""

from typing import Deque, Iterable, Union
from collections import deque
import asyncio
import math
import time

from vllm import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from llumnix.backends.backend_interface import EngineState


class V1EngineAdapter:
    def __init__(self, engine_args):
        self.engine_args = engine_args
        self.engine = AsyncLLM.from_engine_args(engine_args)
        self.requests = {}
        self.waiting = deque()
        self.running = deque()
        self.state = EngineState.RUNNING

    def generate(self, prompt, sampling_params: SamplingParams, request_id: str):
        return self.engine.generate(prompt, sampling_params, request_id)

    def add_request(self, request_id, server_info, expected_steps, prompt,
                    sampling_params, *args, **kwargs):
        self.requests[request_id] = (server_info, time.time())
        self.running.append(request_id)
        return self.generate(prompt, sampling_params, request_id)

    def get_all_request_ids(self):
        return list(self.requests)

    def abort_request(self, request_id):
        ids = (request_id,) if isinstance(request_id, str) else tuple(request_id)
        for rid in ids:
            self.requests.pop(rid, None)
        return asyncio.create_task(self.abort(ids))

    def get_running_queue(self) -> Deque:
        return self.running

    def get_waiting_queue(self) -> Deque:
        return self.waiting

    def update_instance_info(self, info):
        info.num_running_requests = len(self.running)
        info.num_waiting_requests = len(self.waiting)
        info.num_seqs = info.num_running_requests
        info.num_total_gpu_blocks = 0
        info.num_used_gpu_blocks = 0
        info.num_free_gpu_blocks = 0
        info.gpu_cache_usage = 0.0

    # The V1 engine owns scheduling and does not expose Llumnix's legacy
    # request/block-manager mutation hooks.  Keep these methods explicit so
    # callers cannot accidentally enter a partially-compatible migration path.
    def free_dst_pre_alloc_cache(self):
        raise NotImplementedError("KV-cache migration is unavailable for vLLM V1")

    def pop_migrating_out_requests_last_stage(self):
        return []

    def add_running_request(self, request):
        raise NotImplementedError("request migration is unavailable for vLLM V1")

    def add_waiting_request(self, request):
        raise NotImplementedError("request migration is unavailable for vLLM V1")

    # KV-cache migration is not safe to emulate against V1's redesigned
    # scheduler. Keep the contract explicit until a V1 block-manager adapter
    # is implemented.
    def __getattr__(self, name):
        if name in {"pre_alloc", "send_blocks", "commit_dst_request"}:
            raise NotImplementedError(f"{name} requires a vLLM V1 KV-cache adapter")
        raise AttributeError(name)

    async def abort(self, request_id: Union[str, Iterable[str]]):
        await self.engine.abort(request_id)

    def shutdown(self):
        if self.state == EngineState.STOPPED:
            return
        self.state = EngineState.STOPPED
        self.engine.shutdown()

    @property
    def model_executor(self):
        return None
