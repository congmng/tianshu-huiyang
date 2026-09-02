from dataclasses import dataclass
from pydantic import BaseModel
from typing import Any, Optional


class APIResponse(BaseModel):
    code: int
    message: str
    data: Any


class InferenceInstanceInfo(BaseModel):
    instance_id: str
    node_id: str = ""
    node_ip: str = ""
    gpu_count: int
    request_count: int
    running_request_count: int
    waiting_request_count: int
    total_gpu_blocks_count: int
    used_gpu_blocks_count: int
    waiting_gpu_blocks_count: int
    # V1/CoreX heterogeneous-load signals. Defaults preserve the legacy API
    # response shape for callers constructing this model directly.
    gpu_memory_total_bytes: int = 0
    gpu_memory_free_bytes: int = 0
    compute_capacity: float = 1.0
    # ``None`` represents the scheduler's non-finite idle sentinel in the
    # JSON API. It is deliberately distinct from an actual numeric load.
    dispatch_load_metric: Optional[float] = 0.0
    kv_cache_affinity_blocks: int = 0


class BenchmarkRequest(BaseModel):
    qps: float
    num_prompts: int
