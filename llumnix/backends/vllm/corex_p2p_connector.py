"""CoreX-compatible vLLM V1 P2P connector.

CoreX 4.4 ships an NCCL-compatible library whose public API predates the
optional symmetric-memory window functions that vLLM's generic ctypes wrapper
tries to bind.  The V1 P2P connector does not use those optional functions, so
filter them before constructing the upstream connector.  No driver or shared
library is modified.
"""

from __future__ import annotations

import re

from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


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
