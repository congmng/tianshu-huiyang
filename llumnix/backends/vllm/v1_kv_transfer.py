"""Build vLLM V1 KV-transfer configuration from Llumnix settings.

The legacy Llumnix migration backends mutate vLLM 0.6 block-manager state and
cannot be used by V1.  vLLM 0.11 exposes connector configuration through
``AsyncEngineArgs``; this module keeps that translation in one place and does
not start a connector unless the user explicitly selects ``kvtransfer``.
"""

from __future__ import annotations

import os
import hashlib
import ctypes
import copy
import socket
from typing import Any


P2P_REQUEST_ID_PREFIX = "___decode_addr_"
P2P_REQUEST_ID_SUFFIX = "___"
P2P_PREFILL_ID_PREFIX = "___prefill_addr_"
P2P_CONNECTORS = {"P2pNcclConnector", "CoreXP2pNcclConnector"}


def default_kv_ip() -> str:
    """Return this process' routable address for a cross-host connector.

    ``127.0.0.1`` is correct only for a single host.  Llumlets are created
    inside Ray actors, so deriving the address at configuration time gives
    each actor an endpoint reachable by its peer without requiring a per-node
    environment override.  The explicit variable remains authoritative for
    unusual multi-NIC deployments.
    """
    override = os.getenv("LLUMNIX_KV_IP")
    if override:
        return override
    try:
        address = socket.gethostbyname(socket.gethostname())
        if address and address != "127.0.0.1":
            return address
    except OSError:
        pass
    return "127.0.0.1"


def valid_p2p_endpoint(address: str | None) -> bool:
    """Return whether an endpoint is a concrete routable ``host:port``."""
    if not isinstance(address, str) or not address:
        return False
    host, separator, port = address.rpartition(":")
    return bool(host and separator and port.isdigit() and 0 < int(port) < 65536)


def corex_nccl_needs_compat() -> bool:
    """Return whether the loaded NCCL lacks vLLM's optional window API."""
    try:
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            find_nccl_library,
        )

        library = ctypes.CDLL(find_nccl_library())
        return not hasattr(library, "ncclCommWindowRegister")
    except (ImportError, OSError):
        return False


def strip_p2p_request_id(request_id: str) -> str:
    """Remove connector routing metadata from a vLLM output request id."""
    markers = [
        request_id.find(P2P_REQUEST_ID_PREFIX),
        request_id.find(P2P_PREFILL_ID_PREFIX),
    ]
    marker = min(marker for marker in markers if marker >= 0) if any(
        marker >= 0 for marker in markers
    ) else -1
    if marker < 0:
        return request_id
    return request_id[:marker]


def decorate_p2p_request_id(request_id: str, decode_address: str | None) -> str:
    """Attach P2pNcclConnector's required decode return address once."""
    if not decode_address or P2P_REQUEST_ID_PREFIX in request_id:
        return request_id
    if not valid_p2p_endpoint(decode_address):
        raise ValueError(
            "LLUMNIX_KV_DECODE_ADDRESS must be a concrete host:port for "
            "P2pNcclConnector"
        )
    return f"{request_id}{P2P_REQUEST_ID_PREFIX}{decode_address}{P2P_REQUEST_ID_SUFFIX}"


def decorate_p2p_consumer_request_id(
    request_id: str, prefill_address: str | None
) -> str:
    """Attach P2pNcclConnector's required prefill address for a consumer."""
    if not prefill_address or P2P_PREFILL_ID_PREFIX in request_id:
        return request_id
    if not valid_p2p_endpoint(prefill_address):
        raise ValueError(
            "prefill P2P endpoint must be a concrete host:port"
        )
    return f"{request_id}{P2P_PREFILL_ID_PREFIX}{prefill_address}{P2P_REQUEST_ID_SUFFIX}"


def decorate_p2p_pd_request_id(
    request_id: str, decode_address: str, prefill_address: str
) -> str:
    """Build one shared internal ID carrying both P/D routing endpoints."""
    return decorate_p2p_consumer_request_id(
        decorate_p2p_request_id(request_id, decode_address), prefill_address
    )


def p2p_connector_enabled(engine_args: Any) -> bool:
    config = getattr(engine_args, "kv_transfer_config", None)
    return config is not None and getattr(config, "kv_connector", None) in P2P_CONNECTORS


def producer_sampling_params(sampling_params: Any) -> Any:
    """Return P/D producer parameters that stop after the prefill handoff.

    ``P2pNcclConnector`` exports the completed prompt KV during the producer's
    first forward pass.  Letting that request use the public ``max_tokens``
    would continue decoding tokens that are never returned and needlessly
    occupy the prefill worker.  One token retains vLLM's normal prefill-to-
    decode transition while the consumer owns the full public generation.
    """
    if sampling_params is None or not hasattr(sampling_params, "max_tokens"):
        return sampling_params
    producer_params = copy.copy(sampling_params)
    producer_params.max_tokens = 1
    return producer_params


def validate_p2p_environment(engine_args: Any) -> None:
    """Fail early with actionable guidance for a P2P deployment.

    P2pNcclConnector requires exactly two transfer peers and a concrete
    endpoint for producer request IDs. It is unsafe to silently start a
    single-instance service with a half-configured connector.
    """
    config = getattr(engine_args, "kv_transfer_config", None)
    if config is None or getattr(config, "kv_connector", None) not in P2P_CONNECTORS:
        return
    if int(getattr(config, "kv_parallel_size", 0)) != 2:
        raise ValueError("P2pNcclConnector requires kv_parallel_size=2")
    if getattr(config, "kv_role", None) in ("kv_producer", "kv_both"):
        # In Llumnix P/D mode the peer endpoint is learned after both Llumlets
        # have initialized (via ``Manager`` instance metadata) and is then
        # embedded in the shared request ID.  Requiring the environment
        # variable during engine construction races that discovery and makes
        # every producer fail before its model is ready.  Standalone callers
        # still get a clear error from ``decorate_p2p_request_id`` when they
        # submit a request without an endpoint; an explicit value is checked
        # here for early validation.
        address = os.getenv("LLUMNIX_KV_DECODE_ADDRESS")
        if address is not None and not valid_p2p_endpoint(address):
            raise ValueError(
                "LLUMNIX_KV_DECODE_ADDRESS must be a concrete host:port"
            )


def configure_v1_kv_transfer(
    engine_args: Any,
    migration_config: Any,
    instance_id: str | None = None,
    instance_type: str | None = None,
) -> bool:
    """Apply Llumnix ``kvtransfer`` settings to vLLM V1 engine arguments.

    Returns ``True`` when a connector was configured. Existing explicit vLLM
    config objects are preserved, so callers can pass the full vLLM JSON
    configuration unchanged. ``LLUMNIX_KV_*`` environment variables provide
    per-instance rank/address values for multi-process deployments.
    """
    migration_backend = getattr(migration_config, "migration_backend", None)
    current = getattr(engine_args, "kv_transfer_config", None)
    # An explicit native vLLM P2P configuration is also in scope.  Llumnix
    # must still apply the CoreX ABI shim even when the legacy migration
    # backend flag is left at its default value.
    explicit_p2p = getattr(current, "kv_connector", None) in P2P_CONNECTORS
    if migration_backend != "kvtransfer" and not explicit_p2p:
        return False

    from vllm.config import KVEventsConfig, KVTransferConfig

    # Prefix hashes must be reproducible when a request is routed by a
    # different process/host. vLLM's V1 ``NONE_HASH`` is seeded from
    # PYTHONHASHSEED; use its cross-language CBOR algorithm and a fixed seed
    # unless the deployment explicitly chose otherwise.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if getattr(engine_args, "prefix_caching_hash_algo", None) in (None, "sha256"):
        engine_args.prefix_caching_hash_algo = "sha256_cbor"
    if getattr(engine_args, "enable_prefix_caching", None) is not True:
        engine_args.enable_prefix_caching = True

    if current is None:
        connector = getattr(migration_config, "migration_backend_transfer_type", "")
        if not connector or connector == "rdma":
            connector = "SharedStorageConnector"
        connector_module_path = None
        if connector == "P2pNcclConnector" and corex_nccl_needs_compat():
            # CoreX 4.4's NCCL 2.24 does not export the optional symmetric
            # memory window symbols that vLLM's generic wrapper probes. The
            # local connector shim removes only those descriptors; all normal
            # send/recv APIs continue to use the vendor library unchanged.
            connector = "CoreXP2pNcclConnector"
            connector_module_path = "llumnix.backends.vllm.corex_p2p_connector"
        default_role = {
            "prefill": "kv_producer",
            "decode": "kv_consumer",
        }.get(instance_type or "", "kv_both")
        role = os.getenv("LLUMNIX_KV_ROLE", default_role)
        rank = os.getenv(
            "LLUMNIX_KV_RANK",
            "0" if role == "kv_producer" else "1" if role == "kv_consumer" else "0",
        )
        default_parallel_size = "2" if connector in P2P_CONNECTORS else "1"
        parallel_size = int(os.getenv("LLUMNIX_KV_PARALLEL_SIZE", default_parallel_size))
        ip = default_kv_ip()
        port = int(os.getenv("LLUMNIX_KV_PORT", "14579"))
        extra: dict[str, Any] = {}
        naming = getattr(migration_config, "kvtransfer_migration_backend_naming_url", "")
        if naming:
            # SharedStorageConnector uses a filesystem path; retain the
            # historical naming URL as an explicit connector option.
            extra["shared_storage_path"] = naming.removeprefix("file:")
        current = KVTransferConfig(
            kv_connector=connector,
            kv_role=role,
            kv_rank=int(rank) if rank is not None else None,
            kv_parallel_size=parallel_size,
            kv_ip=ip,
            kv_port=port,
            kv_connector_extra_config=extra,
            kv_connector_module_path=connector_module_path,
        )
        engine_args.kv_transfer_config = current
    elif (
        getattr(current, "kv_connector", None) == "P2pNcclConnector"
        and corex_nccl_needs_compat()
    ):
        # Respect every explicit vLLM transfer setting while redirecting the
        # connector class through the CoreX ABI shim. This covers users that
        # pass --kv-transfer-config directly rather than Llumnix's legacy
        # migration options.
        current.kv_connector = "CoreXP2pNcclConnector"
        current.kv_connector_module_path = (
            "llumnix.backends.vllm.corex_p2p_connector"
        )

    # Events are useful for the affinity index and harmless for connectors
    # that do not consume them. Preserve an explicitly supplied configuration.
    if migration_backend == "kvtransfer" and getattr(engine_args, "kv_events_config", None) is None:
        endpoint = os.getenv("LLUMNIX_KV_EVENTS_ENDPOINT")
        if endpoint is None:
            # The V1 engine core binds the publisher. Give each colocated
            # Llumnix instance a deterministic port to avoid collisions while
            # retaining a single endpoint in the engine arguments.
            suffix = int.from_bytes(
                hashlib.blake2b((instance_id or "local").encode(), digest_size=2).digest(),
                "big",
            ) % 1000
            endpoint = f"tcp://*:{15557 + suffix}"
        replay_endpoint = os.getenv("LLUMNIX_KV_EVENTS_REPLAY_ENDPOINT")
        if replay_endpoint is None:
            # Keep the replay listener beside its per-instance PUB endpoint.
            # vLLM's publisher retains event batches, so a restarted adapter
            # can rebuild ownership before accepting cache-aware dispatches.
            if endpoint.startswith("tcp://"):
                replay_endpoint = endpoint.rsplit(":", 1)[0] + ":" + str(
                    int(endpoint.rsplit(":", 1)[1]) + 1000
                )
        engine_args.kv_events_config = KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=endpoint,
            replay_endpoint=replay_endpoint,
        )
    return True
