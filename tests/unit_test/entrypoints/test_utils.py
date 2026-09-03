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

import os
import subprocess
import sys
import pytest
import ray
import socket

from llumnix.arg_utils import (ManagerArgs, EntrypointsArgs, InstanceArgs, LaunchArgs,
                               LlumnixArgumentParser)
from llumnix.entrypoints.setup import launch_ray_cluster, init_manager, init_llumnix_components
from llumnix.entrypoints.utils import get_ip_address, retry_manager_method_sync, retry_manager_method_async
from llumnix.entrypoints.utils import LaunchMode
from llumnix.backends.backend_interface import BackendType
from llumnix.queue.utils import init_request_output_queue_server
from llumnix.utils import get_manager_name

# pylint: disable=unused-import
from tests.conftest import ray_env


def test_launch_ray_cluster(monkeypatch):
    ip_address = get_ip_address()
    os.environ['HEAD_NODE'] = '1'
    os.environ['HEAD_NODE_IP'] = ip_address
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="started", stderr="")

    monkeypatch.setattr("llumnix.entrypoints.setup.subprocess.run", run)
    result = launch_ray_cluster(18079)
    assert result.returncode == 0
    assert ["ray", "stop"] in calls
    assert ["ray", "start", "--head", f"--node-ip-address={ip_address}",
            "--port=18079"] in calls


def test_launch_ray_cluster_accepts_corex_resource_overrides(monkeypatch):
    ip_address = get_ip_address()
    os.environ['HEAD_NODE'] = '1'
    os.environ['HEAD_NODE_IP'] = ip_address
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['LLUMNIX_RAY_NUM_CPUS'] = '4'
    os.environ['LLUMNIX_RAY_OBJECT_STORE_MEMORY'] = '2147483648'
    os.environ['LLUMNIX_RAY_TEMP_DIR'] = '/data1/congmng/llumnix/.ray-test'
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="started", stderr="")

    monkeypatch.setattr("llumnix.entrypoints.setup.subprocess.run", run)
    try:
        launch_ray_cluster(18080)
        start = next(call for call in calls if call[:3] == ["ray", "start", "--head"])
        assert "--num-gpus" in start and "--num-cpus" in start
        assert "--object-store-memory" in start and "--temp-dir" in start
    finally:
        for name in ("HEAD_NODE", "HEAD_NODE_IP", "CUDA_VISIBLE_DEVICES",
                     "LLUMNIX_RAY_NUM_CPUS", "LLUMNIX_RAY_OBJECT_STORE_MEMORY",
                     "LLUMNIX_RAY_TEMP_DIR"):
            os.environ.pop(name, None)


def test_no_launch_ray_cluster_overrides_config_default(monkeypatch):
    parser = LlumnixArgumentParser()
    EntrypointsArgs.add_cli_args(parser)
    monkeypatch.setattr(sys, "argv", ["test", "--no-launch-ray-cluster"])
    cli_args = parser.parse_args()
    args = EntrypointsArgs(launch_ray_cluster=cli_args.launch_ray_cluster)
    assert args.launch_ray_cluster is False

def test_init_manager(ray_env):
    manager = init_manager(ManagerArgs())
    assert manager is not None
    manager_actor_handle = ray.get_actor(get_manager_name(), namespace='llumnix')
    assert manager_actor_handle is not None
    assert manager == manager_actor_handle

def test_init_zmq(ray_env):
    ip = '127.0.0.1'
    with socket.socket() as sock:
        sock.bind((ip, 0))
        port = sock.getsockname()[1]
    request_output_queue = init_request_output_queue_server(ip, port, 'zmq')
    assert request_output_queue is not None
    request_output_queue.cleanup()


def test_local_setup_forwards_complete_manager_context(monkeypatch):
    captured = {}

    class _RemoteCall:
        def remote(self, *args):
            return args

    class _Remote:
        init_instances = _RemoteCall()

    manager = _Remote()
    entrypoints = EntrypointsArgs(host="127.0.0.1", port=8000)
    manager_args = ManagerArgs(initial_instances=0)
    instance_args = InstanceArgs()
    engine_args = object()
    launch_args = LaunchArgs(LaunchMode.LOCAL, BackendType.VLLM)

    def fake_init_manager(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return manager

    monkeypatch.setattr("llumnix.entrypoints.setup.init_manager", fake_init_manager)
    monkeypatch.setattr("llumnix.entrypoints.setup.retry_manager_method_sync",
                        lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr("llumnix.entrypoints.setup.init_request_output_queue_server",
                        lambda *_args, **_kwargs: object())
    init_llumnix_components(entrypoints, manager_args, instance_args, engine_args, launch_args)
    assert captured["args"] == (manager_args,)
    assert captured["kwargs"] == {
        "instance_args": instance_args,
        "entrypoints_args": entrypoints,
        "engine_args": engine_args,
        "launch_args": launch_args,
    }

def test_retry_manager_method_sync(ray_env):
    manager = init_manager(ManagerArgs())
    ret = retry_manager_method_sync(manager.is_ready.remote, 'is_ready')
    assert ret is True

@pytest.mark.asyncio
async def test_retry_manager_method_async(ray_env):
    manager = init_manager(ManagerArgs())
    ret = await retry_manager_method_async(manager.is_ready.remote, 'is_ready')
    assert ret is True
