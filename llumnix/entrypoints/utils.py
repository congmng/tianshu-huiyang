import socket
from enum import Enum
from typing import Dict
import subprocess
import asyncio
import time
import ray
import psutil

from llumnix.logging.logger import init_logger
from llumnix.constants import MAX_MANAGER_RETRY_TIMES, RETRY_MANAGER_INTERVAL

logger = init_logger(__name__)


# Put it in utils.py to avoid circular import.
class LaunchMode(str, Enum):
    LOCAL = "LOCAL"
    GLOBAL = "GLOBAL"


# Use "" type hint to avoid circular import.
class EntrypointsContext:
    def __init__(
        self,
        manager: "Manager",
        instances: Dict[str, "Llumlet"],
        request_output_queue: "QueueServerBase",
        server_info: "ServerInfo",
        log_requests: bool,
        log_request_timestamps: bool,
    ):
        self.manager = manager
        self.instances = instances
        self.request_output_queue = request_output_queue
        self.server_info = server_info
        self.log_requests = log_requests
        self.log_request_timestamps = log_request_timestamps


def get_ip_address():
    interfaces = psutil.net_if_addrs()
    for interface, addrs in interfaces.items():
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                return addr.address


def is_gpu_available() -> bool:
    # CoreX deployments expose Iluvatar devices through ``ixsmi`` rather than
    # NVIDIA's ``nvidia-smi``. Prefer the vendor probe, then fall back to the
    # already-installed PyTorch runtime so API startup is not skipped merely
    # because the host has no NVIDIA utility.
    for probe in ("ixsmi", "nvidia-smi"):
        try:
            subprocess.check_output([probe], stderr=subprocess.STDOUT)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except (ImportError, RuntimeError):
        return False


def retry_manager_method_sync(ray_call, method_name, *args, **kwargs):
    for attempt in range(MAX_MANAGER_RETRY_TIMES):
        try:
            ret = ray.get(ray_call(*args, **kwargs))
            break
        except ray.exceptions.RayActorError:
            if attempt < MAX_MANAGER_RETRY_TIMES - 1:
                logger.warning(
                    "Manager is unavailable, sleep {}s, and retry {} again.".format(
                        RETRY_MANAGER_INTERVAL, method_name
                    )
                )
                time.sleep(RETRY_MANAGER_INTERVAL)
            else:
                logger.error(
                    "Manager is still unavailable after {} times retries.".format(
                        MAX_MANAGER_RETRY_TIMES
                    )
                )
                raise
    return ret


async def retry_manager_method_async(ray_call, method_name, *args, **kwargs):
    for attempt in range(MAX_MANAGER_RETRY_TIMES):
        try:
            ret = await ray_call(*args, **kwargs)
            break
        except ray.exceptions.RayActorError:
            if attempt < MAX_MANAGER_RETRY_TIMES - 1:
                logger.warning(
                    "Manager is unavailable, sleep {}s, and retry {} again.".format(
                        RETRY_MANAGER_INTERVAL, method_name
                    )
                )
                await asyncio.sleep(RETRY_MANAGER_INTERVAL)
            else:
                logger.error(
                    "Manager is still unavailable after {} times retries.".format(
                        MAX_MANAGER_RETRY_TIMES
                    )
                )
                raise
    return ret
