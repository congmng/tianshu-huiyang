"""Build vLLM V1 KV-transfer configuration from Llumnix settings.

The legacy Llumnix migration backends mutate vLLM 0.6 block-manager state and
cannot be used by V1.  vLLM 0.11 exposes connector configuration through
``AsyncEngineArgs``; this module keeps that translation in one place and does
not start a connector unless the user explicitly selects ``kvtransfer``.
"""

from __future__ import annotations

import os
import hashlib
from typing import Any


P2P_REQUEST_ID_PREFIX = "___decode_addr_"
P2P_REQUEST_ID_SUFFIX = "___"


def strip_p2p_request_id(request_id: str) -> str:
    """Remove connector routing metadata from a vLLM output request id."""
    marker = request_id.find(P2P_REQUEST_ID_PREFIX)
    if marker < 0:
        return request_id
    return request_id[:marker]


def decorate_p2p_request_id(request_id: str, decode_address: str | None) -> str:
    """Attach P2pNcclConnector's required decode return address once."""
    if not decode_address or P2P_REQUEST_ID_PREFIX in request_id:
        return request_id
    host, separator, port = decode_address.rpartition(":")
    if not host or not separator or not port.isdigit():
        raise ValueError(
            "LLUMNIX_KV_DECODE_ADDRESS must be a concrete host:port for "
            "P2pNcclConnector"
        )
    return f"{request_id}{P2P_REQUEST_ID_PREFIX}{decode_address}{P2P_REQUEST_ID_SUFFIX}"


def p2p_connector_enabled(engine_args: Any) -> bool:
    config = getattr(engine_args, "kv_transfer_config", None)
    return config is not None and getattr(config, "kv_connector", None) == "P2pNcclConnector"


def validate_p2p_environment(engine_args: Any) -> None:
    """Fail early with actionable guidance for a P2P deployment.

    P2pNcclConnector requires exactly two transfer peers and a concrete
    endpoint for producer request IDs. It is unsafe to silently start a
    single-instance service with a half-configured connector.
    """
    config = getattr(engine_args, "kv_transfer_config", None)
    if config is None or getattr(config, "kv_connector", None) != "P2pNcclConnector":
        return
    if int(getattr(config, "kv_parallel_size", 0)) != 2:
        raise ValueError("P2pNcclConnector requires kv_parallel_size=2")
    if getattr(config, "kv_role", None) in ("kv_producer", "kv_both"):
        address = os.getenv("LLUMNIX_KV_DECODE_ADDRESS")
        if not address:
            raise ValueError(
                "P2pNcclConnector producer requires LLUMNIX_KV_DECODE_ADDRESS=host:port"
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
    if getattr(migration_config, "migration_backend", None) != "kvtransfer":
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

    current = getattr(engine_args, "kv_transfer_config", None)
    if current is None:
        connector = getattr(migration_config, "migration_backend_transfer_type", "")
        if not connector or connector == "rdma":
            connector = "SharedStorageConnector"
        default_role = {
            "prefill": "kv_producer",
            "decode": "kv_consumer",
        }.get(instance_type or "", "kv_both")
        role = os.getenv("LLUMNIX_KV_ROLE", default_role)
        rank = os.getenv(
            "LLUMNIX_KV_RANK",
            "0" if role == "kv_producer" else "1" if role == "kv_consumer" else "0",
        )
        default_parallel_size = "2" if connector == "P2pNcclConnector" else "1"
        parallel_size = int(os.getenv("LLUMNIX_KV_PARALLEL_SIZE", default_parallel_size))
        ip = os.getenv("LLUMNIX_KV_IP", "127.0.0.1")
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
        )
        engine_args.kv_transfer_config = current

    # Events are useful for the affinity index and harmless for connectors
    # that do not consume them. Preserve an explicitly supplied configuration.
    if getattr(engine_args, "kv_events_config", None) is None:
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
        engine_args.kv_events_config = KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=endpoint,
        )
    return True
