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

from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
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
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p import (
            p2p_nccl_engine,
        )

        original = p2p_nccl_engine.set_p2p_nccl_context

        @contextmanager
        def corex_context(num_channels):
            with original(num_channels):
                os.environ["NCCL_CUMEM_ENABLE"] = "0"
                yield

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


class CoreXP2pNcclConnector(P2pNcclConnector):
    """P2pNcclConnector with optional CoreX NCCL symbols filtered."""

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
