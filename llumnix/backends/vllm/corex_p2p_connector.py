"""CoreX-compatible vLLM V1 P2P connector.

CoreX 4.4 ships an NCCL-compatible library whose public API predates the
optional symmetric-memory window functions that vLLM's generic ctypes wrapper
tries to bind.  The V1 P2P connector does not use those optional functions, so
filter them before constructing the upstream connector.  No driver or shared
library is modified.
"""

from __future__ import annotations

from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
)


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
