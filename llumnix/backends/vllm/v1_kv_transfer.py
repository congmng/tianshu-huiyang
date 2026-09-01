"""Build vLLM V1 KV-transfer configuration from Llumnix settings.

The legacy Llumnix migration backends mutate vLLM 0.6 block-manager state and
cannot be used by V1.  vLLM 0.11 exposes connector configuration through
``AsyncEngineArgs``; this module keeps that translation in one place and does
not start a connector unless the user explicitly selects ``kvtransfer``.
"""

from __future__ import annotations

import os
from typing import Any


def configure_v1_kv_transfer(engine_args: Any, migration_config: Any) -> bool:
    """Apply Llumnix ``kvtransfer`` settings to vLLM V1 engine arguments.

    Returns ``True`` when a connector was configured. Existing explicit vLLM
    config objects are preserved, so callers can pass the full vLLM JSON
    configuration unchanged. ``LLUMNIX_KV_*`` environment variables provide
    per-instance rank/address values for multi-process deployments.
    """
    if getattr(migration_config, "migration_backend", None) != "kvtransfer":
        return False

    from vllm.config import KVEventsConfig, KVTransferConfig

    current = getattr(engine_args, "kv_transfer_config", None)
    if current is None:
        connector = getattr(migration_config, "migration_backend_transfer_type", "")
        if not connector or connector == "rdma":
            connector = "SharedStorageConnector"
        role = os.getenv("LLUMNIX_KV_ROLE", "kv_both")
        rank = os.getenv("LLUMNIX_KV_RANK")
        parallel_size = int(os.getenv("LLUMNIX_KV_PARALLEL_SIZE", "1"))
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
        endpoint = os.getenv("LLUMNIX_KV_EVENTS_ENDPOINT", "tcp://*:5557")
        engine_args.kv_events_config = KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=endpoint,
        )
    return True
