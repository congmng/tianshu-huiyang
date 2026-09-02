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
import os

from vllm import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from llumnix.backends.backend_interface import EngineState
from llumnix.backends.vllm.v1_kv import KVCacheAffinityIndex, KVEventSubscriber
from llumnix.backends.vllm.v1_kv_transfer import (
    decorate_p2p_request_id,
    p2p_connector_enabled,
    strip_p2p_request_id,
)


class V1EngineAdapter:
    def __init__(self, engine_args, instance_id: str = "local"):
        self.engine_args = engine_args
        self.engine = AsyncLLM.from_engine_args(engine_args)
        self.instance_id = instance_id
        self.kv_affinity = KVCacheAffinityIndex()
        self.kv_event_subscriber = None
        events_config = getattr(engine_args, "kv_events_config", None)
        if events_config is not None and getattr(events_config, "enable_kv_cache_events", False):
            endpoint = getattr(events_config, "endpoint", "")
            topic = getattr(events_config, "topic", "")
            replay_endpoint = getattr(events_config, "replay_endpoint", None)
            if endpoint:
                self.kv_event_subscriber = KVEventSubscriber(
                    endpoint,
                    self._apply_kv_events,
                    topic=topic,
                    replay_endpoint=replay_endpoint,
                )
        self.requests = {}
        self._request_id_aliases = {}
        self.waiting = deque()
        self.running = deque()
        self.state = EngineState.RUNNING

    def _apply_kv_events(self, events) -> None:
        """Update the instance-local affinity index from decoded V1 events."""
        self.kv_affinity.apply(self.instance_id, events)

    @staticmethod
    def public_request_id(request_id: str) -> str:
        return strip_p2p_request_id(request_id)

    def get_kv_affinity(self, block_hashes):
        """Return this instance's cache-hit ratio for a requested prefix."""
        return self.kv_affinity.affinity(self.instance_id, block_hashes)

    def get_prompt_block_hashes(self, prompt: str):
        """Tokenize a text prompt and produce EngineCore-compatible hashes.

        This is called by Manager before dispatch, so ordinary HTTP requests
        can benefit from cache affinity without exposing an extra client API.
        It is intentionally best-effort: unsupported multimodal/prompt-embed
        inputs return no hashes and use the existing dispatch policy.
        """
        tokenizer = self.engine.tokenizer
        if tokenizer is None or not isinstance(prompt, str):
            return ()
        # Keep tokenization aligned with vLLM's InputPreprocessor, which only
        # overrides ``add_special_tokens`` for model-specific cases (e.g.
        # Whisper). Passing False unconditionally would miss cache blocks for
        # text models whose tokenizer adds a BOS token by default.
        token_ids = tokenizer.encode(prompt)
        cache_config = self.engine.vllm_config.cache_config
        block_size = (
            cache_config.block_size
            * self.engine.vllm_config.parallel_config.decode_context_parallel_size
        )
        return self.kv_affinity.prefix_hashes(
            token_ids,
            block_size,
            cache_config.prefix_caching_hash_algo,
        )

    def generate(self, prompt, sampling_params: SamplingParams, request_id: str,
                 decode_address: str | None = None):
        if p2p_connector_enabled(self.engine_args):
            role = getattr(self.engine_args.kv_transfer_config, "kv_role", None)
            if role in ("kv_producer", "kv_both"):
                request_id = decorate_p2p_request_id(
                    request_id, decode_address or os.getenv("LLUMNIX_KV_DECODE_ADDRESS")
                )
        return self.engine.generate(prompt, sampling_params, request_id)

    def add_request(self, request_id, server_info, expected_steps, prompt,
                    sampling_params, *args, **kwargs):
        decode_address = kwargs.pop("llumnix_kv_decode_address", None)
        internal_request_id = request_id
        if p2p_connector_enabled(self.engine_args):
            role = getattr(self.engine_args.kv_transfer_config, "kv_role", None)
            if role in ("kv_producer", "kv_both") and decode_address:
                internal_request_id = decorate_p2p_request_id(request_id, decode_address)
        self._request_id_aliases[request_id] = internal_request_id
        self.requests[request_id] = (server_info, time.time())
        self.running.append(request_id)
        return self.engine.generate(prompt, sampling_params, internal_request_id)

    def get_all_request_ids(self):
        return list(self.requests)

    def abort_request(self, request_id):
        ids = (request_id,) if isinstance(request_id, str) else tuple(request_id)
        internal_ids = tuple(self._request_id_aliases.get(rid, rid) for rid in ids)
        for rid in ids:
            self.requests.pop(rid, None)
            self._request_id_aliases.pop(rid, None)
        return asyncio.create_task(self.abort(internal_ids))

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
        info.kv_cache_block_hashes = self.kv_affinity.block_hashes(self.instance_id)

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
        if self.kv_event_subscriber is not None:
            self.kv_event_subscriber.close()
        self.engine.shutdown()

    @property
    def model_executor(self):
        return None
