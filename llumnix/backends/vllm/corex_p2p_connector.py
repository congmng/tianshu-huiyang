"""CoreX-compatible vLLM V1 P2P connector.

CoreX 4.4 ships an NCCL-compatible library whose public API predates the
optional symmetric-memory window functions that vLLM's generic ctypes wrapper
tries to bind.  The V1 P2P connector does not use those optional functions, so
filter them before constructing the upstream connector.  No driver or shared
library is modified.
"""

from __future__ import annotations

import re
import os
from contextlib import contextmanager
import threading
from collections import defaultdict

import msgpack
import torch
import zmq

from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p import (
    p2p_nccl_connector as upstream_p2p_connector,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _enable_v1_kv_attention_hooks() -> None:
    """Enable the upstream V1 attention transfer wrapper for this connector.

    CoreX's vLLM build guards the direct-call attention wrapper with
    ``VLLM_SUPPORT_IXSERVER``.  That wrapper is also the sole place that
    calls ``save_kv_layer`` / ``wait_for_layer_load`` for V1 connectors.
    P2P metadata therefore reaches the worker but no KV tensors are ever
    exported when the guard is left at its CoreX default (false).  The wrapper
    is generic vLLM connector code; enabling it in the process that explicitly
    loads the CoreX P2P connector restores the normal V1 connector lifecycle
    without changing the installed CoreX or NCCL libraries.
    """
    os.environ.setdefault("VLLM_SUPPORT_IXSERVER", "1")
    # ``vllm.envs`` may already have been imported by the worker before the
    # connector module is loaded, so changing the environment alone would not
    # update its cached boolean setting.
    try:
        import vllm.envs as vllm_envs

        vllm_envs.VLLM_SUPPORT_IXSERVER = True
    except (ImportError, AttributeError):
        pass


def _disable_corex_cumem_for_p2p() -> None:
    """Avoid CoreX NCCL's unstable cuMem path during communicator setup.

    vLLM's generic P2P helper unconditionally sets ``NCCL_CUMEM_ENABLE=1``
    while calling ``ncclCommInitRank``.  CoreX 4.4's NCCL-compatible 2.24
    library can abort the consumer process with ``double free or corruption``
    in that mode.  The regular allocator path is supported by CoreX and is
    sufficient for this connector.  Patch only the imported Python context
    manager; no environment or shared library outside this worker is changed.
    """
    try:
        # Set this before any worker-side communicator is constructed.  The
        # context-manager shim below also re-applies it around each InitRank
        # call because vLLM restores the environment on context exit.
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p import (
            p2p_nccl_engine,
        )

        @contextmanager
        def corex_context(num_channels):
            names = (
                "NCCL_MAX_NCHANNELS",
                "NCCL_MIN_NCHANNELS",
                "NCCL_CUMEM_ENABLE",
            )
            old = {name: os.environ.get(name) for name in names}
            os.environ["NCCL_MAX_NCHANNELS"] = str(num_channels)
            os.environ["NCCL_MIN_NCHANNELS"] = str(num_channels)
            os.environ["NCCL_CUMEM_ENABLE"] = "0"
            try:
                yield
            finally:
                for name, value in old.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

        p2p_nccl_engine.set_p2p_nccl_context = corex_context
    except (ImportError, AttributeError):
        pass


def _drop_unavailable_optional_nccl_symbols() -> None:
    optional = {"ncclCommWindowRegister", "ncclCommWindowDeregister"}
    # Mutate only the in-process descriptor used by vLLM's ctypes wrapper.
    # The CoreX library itself remains untouched.
    available = []
    import ctypes

    try:
        library = ctypes.CDLL(
            __import__(
                "vllm.distributed.device_communicators.pynccl_wrapper",
                fromlist=["find_nccl_library"],
            ).find_nccl_library()
        )
    except Exception:
        return
    for function in NCCLLibrary.exported_functions:
        if function.name in optional:
            try:
                getattr(library, function.name)
            except AttributeError:
                continue
        available.append(function)
    NCCLLibrary.exported_functions = available


_drop_unavailable_optional_nccl_symbols()
_enable_v1_kv_attention_hooks()
_disable_corex_cumem_for_p2p()


class CoreXZmqP2pEngine:
    """CPU-staged transport for CoreX when V1 NCCL worker init is unstable.

    The CoreX 4.4 NCCL library successfully transfers standalone tensors, but
    its rank-1 communicator can abort a vLLM EngineCore process.  This engine
    retains the upstream P2P connector protocol (routing IDs, per-layer KV
    ownership and blocking load semantics) while moving each KV tensor through
    ZMQ as a CPU buffer.  It is intentionally selected only by the CoreX
    connector; native NCCL remains available with ``corex_transport=nccl``.
    """

    def __init__(self, local_rank, config, hostname="", port_offset=0, **_):
        del hostname
        self.config = config
        self.rank = port_offset
        self.device = torch.device(f"cuda:{local_rank}")
        host = getattr(config, "kv_ip", None)
        if not host:
            raise ValueError("CoreX ZMQ P2P requires kv_ip")
        self.zmq_address = f"{host}:{int(config.kv_port) + port_offset}"
        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.setsockopt(zmq.LINGER, 0)
        self.router_socket.bind(f"tcp://{self.zmq_address}")
        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)
        self.recv_store = {}
        self.recv_store_cv = threading.Condition()
        self.recv_request_id_to_tensor_ids = defaultdict(set)
        self.send_request_id_to_tensor_ids = defaultdict(set)
        self._sockets = {}
        self._closed = threading.Event()
        self._recv_timeout_s = float(
            config.get_from_extra_config("zmq_recv_timeout_s", 120.0)
            if hasattr(config, "get_from_extra_config")
            else 120.0
        )
        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()
        logger.info("CoreX P2P using ZMQ CPU staging at %s", self.zmq_address)

    def _listen(self):
        while not self._closed.is_set():
            try:
                events = dict(self.poller.poll(timeout=250))
                if self.router_socket not in events:
                    continue
                frames = self.router_socket.recv_multipart()
                if len(frames) != 3:
                    continue
                peer, packed, payload = frames
                meta = msgpack.loads(packed)
                if meta.get("cmd") != "PUT_CPU":
                    continue
                dtype = getattr(torch, meta["dtype"])
                # ``Tensor.numpy()`` does not support bfloat16 on the
                # Python 3.12/CoreX torch build.  The wire format is raw
                # bytes, so reconstruct through uint8 and reinterpret the
                # storage as the declared dtype (works for all scalar types).
                storage = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                itemsize = torch.empty((), dtype=dtype).element_size()
                expected_bytes = itemsize
                for dim in tuple(meta["shape"]):
                    expected_bytes *= int(dim)
                if storage.numel() != expected_bytes:
                    raise ValueError(
                        f"payload size {storage.numel()} != expected {expected_bytes}"
                    )
                tensor = storage.view(dtype).reshape(tuple(meta["shape"]))
                tensor_id = meta["tensor_id"]
                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.recv_request_id_to_tensor_ids[tensor_id.split("#")[0]].add(tensor_id)
                    self.recv_store_cv.notify_all()
                self.router_socket.send_multipart([peer, b"0"])
            except zmq.ZMQError:
                return
            except Exception as exc:  # malformed peer payload must not kill listener
                logger.warning("CoreX P2P rejected malformed ZMQ payload: %s", exc)
                try:
                    self.router_socket.send_multipart([peer, b"1"])
                except (NameError, zmq.ZMQError):
                    pass

    def _socket(self, remote_address):
        sock = self._sockets.get(remote_address)
        if sock is None:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, int(self._recv_timeout_s * 1000))
            sock.connect(f"tcp://{remote_address}")
            self._sockets[remote_address] = sock
        return sock

    def send_tensor(self, tensor_id, tensor, remote_address=None):
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor.detach().cpu()
                self.recv_store_cv.notify_all()
            return True
        cpu = tensor.detach().contiguous().cpu()
        meta = {
            "cmd": "PUT_CPU",
            "tensor_id": tensor_id,
            "shape": tuple(cpu.shape),
            "dtype": str(cpu.dtype).replace("torch.", ""),
        }
        sock = self._socket(remote_address)
        # Use a byte view instead of ``cpu.numpy()`` so bfloat16 tensors are
        # supported by torch builds whose NumPy bridge lacks bfloat16.
        payload = cpu.view(torch.uint8).numpy().tobytes()
        sock.send_multipart([msgpack.dumps(meta), payload])
        try:
            response = sock.recv()
        except zmq.Again as exc:
            raise TimeoutError(
                f"timed out waiting for CoreX P2P peer {remote_address}"
            ) from exc
        if response != b"0":
            return False
        self.send_request_id_to_tensor_ids[tensor_id.split("#")[0]].add(tensor_id)
        return True

    def recv_tensor(self, tensor_id, remote_address=None):
        del remote_address
        with self.recv_store_cv:
            while tensor_id not in self.recv_store:
                if not self.recv_store_cv.wait(timeout=self._recv_timeout_s):
                    raise TimeoutError(
                        f"timed out waiting for CoreX P2P tensor {tensor_id}"
                    )
            return self.recv_store[tensor_id].to(self.device, non_blocking=True)

    def wait_for_sent(self):
        return None

    def get_finished(self, finished_req_ids, no_compile_layers):
        for request_id in finished_req_ids:
            with self.recv_store_cv:
                for layer_name in no_compile_layers:
                    self.recv_store.pop(f"{request_id}#{layer_name}", None)
                self.recv_request_id_to_tensor_ids.pop(request_id, None)
                self.send_request_id_to_tensor_ids.pop(request_id, None)
        return None, None

    def shutdown(self):
        """Release sockets so a replacement EngineCore can bind its port."""
        if self._closed.is_set():
            return
        self._closed.set()
        self.poller.unregister(self.router_socket)
        for socket in self._sockets.values():
            socket.close(linger=0)
        self._sockets.clear()
        self.router_socket.close(linger=0)
        self.context.term()
        self._listener.join(timeout=1)


class CoreXP2pNcclConnector(P2pNcclConnector):
    """P2pNcclConnector with optional CoreX NCCL symbols filtered."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        config = vllm_config.kv_transfer_config
        transport = config.get_from_extra_config("corex_transport", "zmq_cpu")
        if transport not in {"zmq_cpu", "nccl"}:
            raise ValueError("corex_transport must be 'zmq_cpu' or 'nccl'")
        original = upstream_p2p_connector.P2pNcclEngine
        if transport == "zmq_cpu":
            upstream_p2p_connector.P2pNcclEngine = CoreXZmqP2pEngine
        try:
            super().__init__(vllm_config, role, kv_cache_config)
        finally:
            upstream_p2p_connector.P2pNcclEngine = original

    def shutdown(self):
        engine = getattr(self, "p2p_nccl_engine", None)
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
        """Parse one endpoint from a shared P/D request ID.

        Upstream vLLM 0.11 uses a greedy ``.*`` expression. Llumnix's P/D
        orchestration carries both endpoint markers in one ID, so greediness
        would include the second marker in the hostname. Stop at the next
        marker/suffix explicitly.
        """
        marker = "___decode_addr_" if is_prefill else "___prefill_addr_"
        start = request_id.find(marker)
        if start < 0:
            raise ValueError(
                f"Request id {request_id} does not contain hostname and port"
            )
        value = request_id[start + len(marker):]
        value = value.split("___", 1)[0]
        host, separator, port = value.rpartition(":")
        if not host or not separator or not port.isdigit():
            raise ValueError(
                f"Request id {request_id} does not contain hostname and port"
            )
        return host, int(port)

    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        logger.info(
            "CoreX P2P metadata role=%s requests=%d ids=%s",
            "producer" if self.is_producer else "consumer",
            len(getattr(meta, "requests", ())),
            [request.request_id for request in getattr(meta, "requests", ())],
        )
        return meta

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        logger.info("CoreX P2P save layer=%s shape=%s", layer_name, tuple(kv_layer.shape))
        return super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        logger.info(
            "CoreX P2P load role=%s metadata_requests=%d",
            "producer" if self.is_producer else "consumer",
            len(getattr(metadata, "requests", ())),
        )
        return super().start_load_kv(forward_context, **kwargs)
