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

from typing import Dict, List
import asyncio
import time
import os
import copy
import importlib.util
import inspect

import ray
from ray.util.placement_group import PlacementGroup

from llumnix.backends.backend_interface import BackendInterface, BackendType
from llumnix.queue.queue_type import QueueType
from llumnix.queue.queue_client_base import QueueClientBase
from llumnix.queue.utils import init_request_output_queue_client
from llumnix.server_info import ServerInfo
from llumnix.logging.logger import init_logger
from llumnix.utils import get_instance_name
from llumnix.internal_config import MigrationConfig
from llumnix.metrics.timestamps import set_timestamp

logger = init_logger(__name__)


_CROSS_VLLM_ENGINE_ARG_NAMES = (
    "model", "tokenizer", "served_model_name", "hf_config_path", "runner",
    "convert", "skip_tokenizer_init", "enable_prompt_embeds", "tokenizer_mode",
    "trust_remote_code", "allowed_local_media_path", "download_dir", "load_format",
    "config_format", "dtype", "kv_cache_dtype", "seed", "max_model_len",
    "cudagraph_capture_sizes", "max_cudagraph_capture_size",
    "distributed_executor_backend", "pipeline_parallel_size", "master_addr",
    "master_port", "nnodes", "node_rank", "tensor_parallel_size",
    "max_num_batched_tokens", "max_num_scheduled_tokens", "block_size",
    "enable_prefix_caching", "prefix_caching_hash_algo", "disable_sliding_window",
    "disable_cascade_attn", "gpu_memory_utilization", "kv_cache_memory_bytes",
    "max_num_partial_prefills", "max_long_partial_prefills",
    "long_prefill_token_threshold", "max_num_seqs", "max_logprobs", "logprobs_mode",
    "disable_log_stats", "revision", "code_revision", "tokenizer_revision",
    "quantization", "enforce_eager", "disable_custom_all_reduce",
    "limit_mm_per_prompt", "enable_mm_embeds", "enable_lora", "max_loras",
    "max_lora_rank", "default_mm_loras", "num_gpu_blocks_override",
    "model_loader_extra_config", "ignore_patterns", "enable_chunked_prefill",
    "disable_chunked_mm_input", "max_parallel_loading_workers", "worker_cls",
    "worker_extension_cls", "generation_config", "model_impl", "additional_config",
    "async_scheduling", "stream_interval", "tokens_only", "enable_log_requests",
)


def rebuild_engine_args_for_runtime(engine_args):
    """Rebuild serialized engine args with the node's installed vLLM class.

    A mixed Ray cluster can deserialize a 0.11 ``AsyncEngineArgs`` instance
    in a 0.25 worker.  The two classes share a module path but not a field
    contract, so ``isinstance`` cannot detect this boundary.  Reconstructing
    from stable CLI values avoids passing version-specific config objects.
    """
    from vllm.engine.arg_utils import AsyncEngineArgs

    signature = inspect.signature(AsyncEngineArgs)
    if not hasattr(engine_args, "model_class_overrides"):
        values = {}
        for name in _CROSS_VLLM_ENGINE_ARG_NAMES:
            if name not in signature.parameters or not hasattr(engine_args, name):
                continue
            value = getattr(engine_args, name)
            if value is not None:
                values[name] = value
        values["model"] = getattr(engine_args, "model")
        return AsyncEngineArgs(**values)
    return engine_args


class AsyncPutQueueActor:
    def __init__(self, instance_id: str, request_output_queue_type: QueueType):
        self.job_id = ray.get_runtime_context().get_job_id()
        self.worker_id = ray.get_runtime_context().get_worker_id()
        self.actor_id = ray.get_runtime_context().get_actor_id()
        self.node_id = ray.get_runtime_context().get_node_id()
        self.instance_id = instance_id
        logger.info("AsyncPutQueueActor(job_id={}, worker_id={}, actor_id={}, node_id={}, instance_id={})".format(
                        self.job_id, self.worker_id, self.actor_id, self.node_id, self.instance_id))
        self.request_output_queue_type = request_output_queue_type
        self.request_output_queue_client: QueueClientBase = init_request_output_queue_client(request_output_queue_type)
        self.engine_actor_handle = None

    def __repr__(self):
        return f"{self.__class__.__name__}(iid={self.instance_id[:5]})"

    async def put_nowait_to_servers(self,
                                    server_request_outputs: Dict[str, List],
                                    server_info_dict: Dict[str, ServerInfo]) -> None:
        if self.engine_actor_handle is None:
            self.engine_actor_handle = ray.get_actor(get_instance_name(self.instance_id), namespace="llumnix")
        tasks = []
        for server_id, req_outputs in server_request_outputs.items():
            server_info = server_info_dict[server_id]
            set_timestamp(req_outputs, 'engine_actor_put_queue_timestamp', time.time())
            tasks.append(asyncio.create_task(self.request_output_queue_client.put_nowait(req_outputs, server_info)))
        rets = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, ret in enumerate(rets):
            if isinstance(ret, Exception):
                server_id = list(server_request_outputs.keys())[idx]
                server_info = server_info_dict[server_id]
                logger.error("Server {} is dead, exception: {}".format(server_id, ret))
                if self.request_output_queue_type == QueueType.ZMQ:
                    logger.warning("request output queue ip: {}, port: {}".format(server_info.request_output_queue_ip,
                                                                                  server_info.request_output_queue_port))
                req_outputs = list(server_request_outputs.values())[idx]
                request_ids = [req_output.request_id for req_output in req_outputs]
                self.engine_actor_handle.abort.remote(request_ids)

def init_backend_engine(instance_id: str,
                        placement_group: PlacementGroup,
                        request_output_queue_type: QueueType,
                        migration_config: MigrationConfig,
                        backend_type: BackendType,
                        engine_args,
                        profiling_result_file_path: str = None) -> BackendInterface:
    if backend_type == BackendType.VLLM:
        # vLLM 0.11 uses V1 and no longer exposes the private 0.6.x engine,
        # scheduler, or executor APIs used by BackendVLLM.  Select the V1
        # request-serving adapter on such installations.  Its KV migration
        # methods are intentionally unavailable until separately ported.
        import vllm
        if importlib.util.find_spec("vllm.v1.engine") is not None:
            # AsyncEngineArgs is mutable and may be shared by several Ray
            # Llumlets during global launch. Clone it before injecting the
            # instance-specific connector rank/ports so one instance cannot
            # inherit another instance's KV endpoint.
            engine_args = rebuild_engine_args_for_runtime(engine_args)
            engine_args = copy.deepcopy(engine_args)
            # Llumnix already reserves the TP GPUs in the parent actor's Ray
            # placement group.  vLLM's Ray executor would try to create a
            # second Ray client and pass resource counts, which fails when
            # another Ray head is present on the host.  Use local multiprocess
            # workers for TP inside that reserved bundle.
            if getattr(engine_args, "tensor_parallel_size", 1) > 1:
                engine_args.distributed_executor_backend = "mp"
            # Translate Llumnix's opt-in ``kvtransfer`` backend to vLLM V1's
            # connector config before AsyncLLM snapshots engine arguments.
            from llumnix.backends.vllm.v1_kv_transfer import configure_v1_kv_transfer
            from llumnix.backends.vllm.v1_kv_transfer import validate_p2p_environment
            configure_v1_kv_transfer(
                engine_args,
                migration_config,
                instance_id,
                getattr(migration_config, "instance_type", None),
            )
            validate_p2p_environment(engine_args)
            from llumnix.backends.vllm.v1_engine import V1EngineAdapter
            backend_engine = V1EngineAdapter(engine_args, instance_id=instance_id)
        else:
            # pylint: disable=import-outside-toplevel
            from llumnix.backends.vllm.llm_engine import BackendVLLM
            backend_engine = BackendVLLM(instance_id,
                                            placement_group,
                                            request_output_queue_type,
                                            migration_config,
                                            engine_args)
    elif backend_type == BackendType.BLADELLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.bladellm.llm_engine import BackendBladeLLM
        backend_engine = BackendBladeLLM(instance_id,
                                         placement_group,
                                         request_output_queue_type,
                                         migration_config,
                                         engine_args)
    elif backend_type == BackendType.SIM_VLLM:
        # pylint: disable=import-outside-toplevel
        from llumnix.backends.vllm.sim_llm_engine import BackendSimVLLM
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        backend_engine = BackendSimVLLM(instance_id,
                                        placement_group,
                                        request_output_queue_type,
                                        migration_config,
                                        engine_args,
                                        profiling_result_file_path)
    else:
        raise ValueError(f'Unsupported backend: {backend_type}')
    return backend_engine

def get_engine_world_size(engine_args, backend_type: BackendType):
    if backend_type == BackendType.VLLM:
        # vLLM V1 validates ParallelConfig against the *driver*'s visible
        # accelerator count.  Llumnix computes placement groups in its CPU
        # Manager actor, where that count is zero, so constructing an engine
        # config here incorrectly rejects valid TP deployments.  World size
        # is purely the product of TP and PP and can be derived without
        # triggering accelerator validation.
        tensor_parallel_size = getattr(engine_args, "tensor_parallel_size", 1)
        pipeline_parallel_size = getattr(engine_args, "pipeline_parallel_size", 1)
        world_size = tensor_parallel_size * pipeline_parallel_size
    else: # BLADE_LLM
        world_size = engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size
    return world_size
