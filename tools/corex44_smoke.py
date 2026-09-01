#!/usr/bin/env python3
"""Minimal non-destructive CoreX 4.4.0 smoke tests for Llumnix.

Run with the project-local CoreX environment:

    CUDA_VISIBLE_DEVICES=0 .conda-corex44/bin/python tools/corex44_smoke.py

The script only allocates one visible accelerator temporarily and starts a
local Ray process.  It neither installs packages nor changes the driver.
"""

import math
import pathlib
import sys

# Make the script runnable directly from any working directory without
# requiring an editable package install (the project metadata targets Python
# <3.11 while the supplied CoreX environment is Python 3.12).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ray
import torch

from llumnix.arg_utils import InstanceArgs
from llumnix.global_scheduler.global_scheduler import GlobalScheduler
from llumnix.instance_info import InstanceInfo
from llumnix.internal_config import GlobalSchedulerConfig


def check_torch() -> None:
    assert torch.cuda.is_available(), "CoreX accelerator is not available to PyTorch"
    tensor = torch.arange(1024, device="cuda", dtype=torch.float32)
    assert tensor.sum().item() == 523776.0
    print(
        "torch: PASS "
        f"version={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(0)} sum={tensor.sum().item()}"
    )


def check_ray() -> None:
    ray.init(num_cpus=2, num_gpus=1, include_dashboard=False)

    @ray.remote(num_gpus=1)
    def accelerator_task() -> dict:
        import torch as remote_torch

        tensor = remote_torch.arange(1024, device="cuda", dtype=remote_torch.float32)
        return {
            "available": remote_torch.cuda.is_available(),
            "device": remote_torch.cuda.get_device_name(0),
            "sum": tensor.sum().item(),
        }

    result = ray.get(accelerator_task.remote())
    assert result["available"] and result["sum"] == 523776.0
    print(f"ray: PASS resources={ray.cluster_resources()} task={result}")
    ray.shutdown()


def check_global_scheduler() -> None:
    config = GlobalSchedulerConfig(
        initial_instances=0,
        dispatch_policy="load",
        topk_random_dispatch=1,
        pair_migration_policy="defrag",
        migrate_out_threshold=3.0,
        scaling_policy="avg_load",
        scaling_load_metric="remaining_steps",
        scale_up_threshold=10,
        scale_down_threshold=60,
        enable_pd_disagg=False,
        is_group_kind_migration_backend=False,
    )
    scheduler = GlobalScheduler(config)
    instance_ids = ["corex-i0", "corex-i1", "corex-i2"]
    scheduler.scale_up(
        instance_ids,
        [InstanceArgs(instance_type="no_constraints") for _ in instance_ids],
    )
    infos = []
    for instance_id, load in zip(instance_ids, [-10.0, -5.0, -1.0]):
        info = InstanceInfo(
            instance_id=instance_id,
            num_total_gpu_blocks=100,
            num_free_gpu_blocks=50,
            num_used_gpu_blocks=50,
            num_running_requests=1,
        )
        info.dispatch_load_metric = load
        infos.append(info)
    scheduler.update_instance_infos(infos)
    selected, expected_steps = scheduler.dispatch()
    assert selected == "corex-i0" and math.isinf(expected_steps)
    print(
        "global_scheduler: PASS "
        f"instances={scheduler.num_instances} selected={selected}"
    )


if __name__ == "__main__":
    check_torch()
    check_ray()
    check_global_scheduler()
    print("corex44_smoke: PASS")
