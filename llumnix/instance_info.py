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

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
import numpy as np
import time

from llumnix.logging.logger import init_logger
from llumnix.llumlet.request import RequestInferenceType

logger = init_logger(__name__)


class InstanceType(str, Enum):
    NO_CONSTRAINTS = "no_constraints"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class InstanceInfo:
    instance_id: str = ""
    instance_type: InstanceType = None

    step_id: int = None
    timestamp: float = None
    num_batched_tokens: int = None
    num_seqs = None
    running_seq_lens: List[int] = field(default_factory=list)
    waiting_seq_lens: List[int] = field(default_factory=list)
    last_inference_latency: float = None
    inference_type: RequestInferenceType = None

    num_total_gpu_blocks: int = 0
    num_watermark_blocks: int = 0
    num_used_gpu_blocks: int = 0
    num_free_gpu_blocks: int = 0
    gpu_cache_usage: float = 0.0
    num_running_requests: int = 0
    num_waiting_requests: int = 0
    num_killed_requests: int = 0
    num_blocks_first_waiting_request: int = 0
    waiting_time_first_waiting_request: int = 0
    num_blocks_all_waiting_requests: int = 0
    num_blocks_last_running_request: int = 0

    # on-demand init infos
    dispatch_load_metric: float = -np.inf
    migration_load_metric: float = np.inf
    migration_load_metric_after_migrate_in: float = -np.inf
    migration_load_metric_after_migrate_out: float = np.inf

    # lazy init infos
    num_available_gpu_blocks: int = 0
    num_available_gpu_blocks_waiting: int = 0

    # manual init infos
    profiling_data: Tuple[str, int, int, float] = None

    def __post_init__(self) -> None:
        self.num_available_gpu_blocks = (
            self.num_free_gpu_blocks - self.num_watermark_blocks
        )
        self.num_available_gpu_blocks_waiting = (
            self.num_available_gpu_blocks - self.num_blocks_all_waiting_requests
        )


class InstanceLoadCalculator:
    def __init__(
        self, dispatch_load_metric: str, migration_load_metric: str, enable_defrag: bool
    ) -> None:
        logger.info(
            f"dispatch_load_metric: {dispatch_load_metric}, migration_load_metric: {migration_load_metric}, enable_defrag: {enable_defrag}"
        )
        self.dispatch_load_calculator = DispatchLoadComputation(dispatch_load_metric)
        self.migration_load_calculator = MigrationLoadComputation(
            migration_load_metric, enable_defrag
        )

    def compute_instance_load(self, instance_info: InstanceInfo):
        instance_info.dispatch_load_metric = (
            self.dispatch_load_calculator.compute_instance_load(instance_info)
        )
        instance_info.migration_load_metric = (
            self.migration_load_calculator.compute_instance_load(instance_info)
        )
        instance_info.migration_load_metric_after_migrate_out = (
            self.migration_load_calculator.compute_instance_load_after_migrate(
                instance_info, is_migrate_in=False
            )
        )
        instance_info.migration_load_metric_after_migrate_in = (
            self.migration_load_calculator.compute_instance_load_after_migrate(
                instance_info, is_migrate_in=True
            )
        )


class LoadComputationStrategy(ABC):
    def __init__(self, load_metric: str, enable_defrag: bool = False) -> None:
        self.load_metric = load_metric
        self.enable_defrag = enable_defrag

    @abstractmethod
    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        pass


class DispatchLoadComputation(LoadComputationStrategy):
    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        instance_load = -np.inf
        if self.load_metric == "usage_ratio":
            if instance_info.num_total_gpu_blocks > 0:
                instance_load = (
                    instance_info.num_used_gpu_blocks
                    + instance_info.num_blocks_all_waiting_requests
                ) / instance_info.num_total_gpu_blocks
        elif self.load_metric == "remaining_steps":
            num_requests = (
                instance_info.num_running_requests + instance_info.num_waiting_requests
            )
            num_available_gpu_blocks = (
                instance_info.num_available_gpu_blocks
                - instance_info.num_blocks_all_waiting_requests
            )
            if num_requests == 0:
                return -np.inf
            logger.info(
                f"instance_load: {instance_load} "
                f"num_available_gpu_blocks: {instance_info.num_available_gpu_blocks} "
                f"num_blocks_all_waiting_requests: {instance_info.num_blocks_all_waiting_requests} "
                f"num_running_requests: {instance_info.num_running_requests} "
                f"num_waiting_requests: {instance_info.num_waiting_requests} "
            )
            instance_load = (num_available_gpu_blocks / num_requests) * (-1)
        elif self.load_metric == "virtual_usage":
            if instance_info.num_total_gpu_blocks > 20000:
                throughput = 1.5
            else:
                throughput = 1.0

            num_requests = (
                instance_info.num_running_requests + instance_info.num_waiting_requests
            )
            if num_requests == 0:
                return -np.inf

            compute_load = num_requests / 256
            compute_weight = 1 / throughput

            # The V1 adapter does not expose legacy physical block counts.
            # An absent counter must not make Manager's polling task divide by
            # zero and evict an otherwise healthy inference instance.
            if instance_info.num_total_gpu_blocks > 0:
                memory_use_ratio = (
                    instance_info.num_used_gpu_blocks
                    + instance_info.num_blocks_all_waiting_requests
                ) / instance_info.num_total_gpu_blocks
            else:
                memory_use_ratio = 0.0
            memory_load = memory_use_ratio
            memory_weight = 1 / (1 + np.exp(-5 * (memory_use_ratio - 0.4)))

            instance_load = (compute_load * compute_weight) + (
                memory_weight * memory_load
            )
            logger.info(
                f"Instance Load Calculation:\n"
                f"  Throughput: {throughput}\n"
                f"  Num Requests: {num_requests}\n"
                f"  Compute Load: {compute_load} (requests/256)\n"
                f"  Compute Weight: {compute_weight}\n"
                f"  Memory Use Ratio: {memory_use_ratio:.3f}\n"
                f"  Memory Load: {memory_load:.3f}\n"
                f"  Memory Weight: {memory_weight:.3f}\n"
                f"  Final Instance Load: {instance_load:.3f}\n"
                f"  Components:\n"
                f"    - Compute Component: {compute_load * compute_weight:.3f}\n"
                f"    - Memory Component: {memory_weight * memory_load:.3f}"
            )
        return instance_load


class MigrationLoadComputation(LoadComputationStrategy):
    def compute_instance_load_after_migrate(
        self, instance_info: InstanceInfo, is_migrate_in: bool
    ) -> float:
        instance_info_after_migrate = copy.deepcopy(instance_info)
        num_blocks_last_running_request = (
            instance_info_after_migrate.num_blocks_last_running_request
        )

        if is_migrate_in:
            instance_info_after_migrate.num_running_requests += 1
            instance_info_after_migrate.num_available_gpu_blocks -= (
                num_blocks_last_running_request
            )
        else:
            instance_info_after_migrate.num_running_requests -= 1
            instance_info_after_migrate.num_available_gpu_blocks += (
                num_blocks_last_running_request
            )

        return self.compute_instance_load(instance_info_after_migrate)

    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        instance_load = -np.inf
        if self.load_metric == "usage_ratio":
            if instance_info.num_total_gpu_blocks > 0:
                instance_load = (
                    instance_info.num_used_gpu_blocks
                    + instance_info.num_blocks_first_waiting_request
                ) / instance_info.num_total_gpu_blocks
        elif self.load_metric == "remaining_steps":
            if not self.enable_defrag:
                num_requests = instance_info.num_running_requests
                num_available_gpu_blocks = instance_info.num_available_gpu_blocks
            else:
                num_requests = instance_info.num_running_requests
                if instance_info.num_waiting_requests != 0:
                    num_requests += 1
                num_available_gpu_blocks = (
                    instance_info.num_available_gpu_blocks
                    - instance_info.num_blocks_first_waiting_request
                )
            if num_requests == 0:
                return -np.inf
            instance_load = (num_available_gpu_blocks / num_requests) * (-1)
        elif self.load_metric == "remaining_tokens":
            instance_load = instance_info.num_available_gpu_blocks
        return instance_load


# TODO(KuilongCui): currently scaling and dispatch use the same load calculator, leave
# it in the future to refine
class ScalingLoadComputation(LoadComputationStrategy):
    def __init__(self, load_metric):
        super().__init__(load_metric)
        self.load_calculator = DispatchLoadComputation(load_metric)

    def compute_instance_load(self, instance_info: InstanceInfo) -> float:
        return self.load_calculator.compute_instance_load(instance_info)
