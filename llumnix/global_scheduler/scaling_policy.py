from typing import List
from abc import ABC, abstractmethod

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, ScalingLoadComputation

logger = init_logger(__name__)


class ScalePolicy(ABC):
    def __init__(self, scaling_load_metric: str) -> None:
        self.scaling_load_calculator = ScalingLoadComputation(scaling_load_metric)

    @abstractmethod
    def compute_load_metric_up(self, instance_infos: List[InstanceInfo]) -> float:
        pass

    @abstractmethod
    def compute_load_metric_down(self, instance_infos: List[InstanceInfo]) -> float:
        pass

    def compute_load_metric_avg(self, instance_infos: List[InstanceInfo]) -> float:
        if not instance_infos:
            return float("-inf")
        return sum(
            self.scaling_load_calculator.compute_instance_load(info)
            for info in instance_infos
        ) / len(instance_infos)


class MaxLoad(ScalePolicy):
    def compute_load_metric_up(self, instance_infos: List[InstanceInfo]) -> float:
        return max(self.scaling_load_calculator.compute_instance_load(i)
                   for i in instance_infos)

    def compute_load_metric_down(self, instance_infos: List[InstanceInfo]) -> float:
        return max(self.scaling_load_calculator.compute_instance_load(i)
                   for i in instance_infos)


class MinLoad(ScalePolicy):
    def compute_load_metric_up(self, instance_infos: List[InstanceInfo]) -> float:
        return min(self.scaling_load_calculator.compute_instance_load(i)
                   for i in instance_infos)

    def compute_load_metric_down(self, instance_infos: List[InstanceInfo]) -> float:
        return min(self.scaling_load_calculator.compute_instance_load(i)
                   for i in instance_infos)


class AvgLoad(ScalePolicy):
    def compute_load_metric_up(self, instance_infos: List[InstanceInfo]) -> float:
        return self.compute_load_metric_avg(instance_infos)

    def compute_load_metric_down(self, instance_infos: List[InstanceInfo]) -> float:
        if len(instance_infos) <= 1:
            return float("inf")
        # Predict the average after removing the least-loaded instance. This
        # works for V1's memory headroom signal and preserves the old policy's
        # intent without relying on invalid numeric instance IDs.
        loads = [(self.scaling_load_calculator.compute_instance_load(info), info)
                 for info in instance_infos]
        _, least_loaded = min(loads, key=lambda item: item[0])
        return self.compute_load_metric_avg(
            [info for info in instance_infos if info is not least_loaded]
        )


class ScalePolicyFactory:
    _POLICY_REGISTRY = {
        'max_load': MaxLoad,
        'min_load': MinLoad,
        'avg_load': AvgLoad,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> ScalePolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
