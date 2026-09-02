from types import SimpleNamespace


def test_kvtransfer_config_is_opt_in(monkeypatch):
    from llumnix.backends.vllm.v1_kv_transfer import configure_v1_kv_transfer

    args = SimpleNamespace(kv_transfer_config=None, kv_events_config=None)
    cfg = SimpleNamespace(
        migration_backend="gloo",
        migration_backend_transfer_type="rdma",
        kvtransfer_migration_backend_naming_url="",
    )
    assert configure_v1_kv_transfer(args, cfg) is False
    assert args.kv_transfer_config is None


def test_kvtransfer_config_maps_llumnix_options(monkeypatch):
    from llumnix.backends.vllm.v1_kv_transfer import configure_v1_kv_transfer

    monkeypatch.setenv("LLUMNIX_KV_ROLE", "kv_producer")
    monkeypatch.setenv("LLUMNIX_KV_RANK", "0")
    monkeypatch.setenv("LLUMNIX_KV_PARALLEL_SIZE", "2")
    monkeypatch.setenv("LLUMNIX_KV_IP", "10.0.0.4")
    monkeypatch.setenv("LLUMNIX_KV_PORT", "16000")
    args = SimpleNamespace(kv_transfer_config=None, kv_events_config=None)
    cfg = SimpleNamespace(
        migration_backend="kvtransfer",
        migration_backend_transfer_type="P2pNcclConnector",
        kvtransfer_migration_backend_naming_url="file:/var/lib/llumnix/kv",
    )
    assert configure_v1_kv_transfer(args, cfg) is True
    transfer = args.kv_transfer_config
    assert transfer.kv_connector in {"P2pNcclConnector", "CoreXP2pNcclConnector"}
    assert transfer.kv_role == "kv_producer"
    assert transfer.kv_rank == 0
    assert transfer.kv_parallel_size == 2
    assert transfer.kv_ip == "10.0.0.4"
    assert transfer.kv_port == 16000
    assert transfer.kv_connector_extra_config["shared_storage_path"] == "/var/lib/llumnix/kv"
    assert args.kv_events_config.enable_kv_cache_events is True
    assert args.kv_events_config.publisher == "zmq"


def test_kvtransfer_makes_cross_host_prefix_hashes_reproducible(monkeypatch):
    from llumnix.backends.vllm.v1_kv_transfer import configure_v1_kv_transfer

    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    args = SimpleNamespace(
        kv_transfer_config=None,
        kv_events_config=None,
        prefix_caching_hash_algo="sha256",
        enable_prefix_caching=False,
    )
    cfg = SimpleNamespace(
        migration_backend="kvtransfer",
        migration_backend_transfer_type="SharedStorageConnector",
        kvtransfer_migration_backend_naming_url="",
    )
    configure_v1_kv_transfer(args, cfg)
    assert args.prefix_caching_hash_algo == "sha256_cbor"
    assert args.enable_prefix_caching is True
    assert __import__("os").environ["PYTHONHASHSEED"] == "0"


def test_kvtransfer_uses_instance_scoped_event_and_replay_endpoints(monkeypatch):
    from llumnix.backends.vllm.v1_kv_transfer import configure_v1_kv_transfer

    monkeypatch.delenv("LLUMNIX_KV_EVENTS_ENDPOINT", raising=False)
    monkeypatch.delenv("LLUMNIX_KV_EVENTS_REPLAY_ENDPOINT", raising=False)
    cfg = SimpleNamespace(
        migration_backend="kvtransfer",
        migration_backend_transfer_type="SharedStorageConnector",
        kvtransfer_migration_backend_naming_url="",
    )
    first = SimpleNamespace(kv_transfer_config=None, kv_events_config=None)
    second = SimpleNamespace(kv_transfer_config=None, kv_events_config=None)
    configure_v1_kv_transfer(first, cfg, instance_id="instance-a")
    configure_v1_kv_transfer(second, cfg, instance_id="instance-b")
    assert first.kv_events_config.endpoint != second.kv_events_config.endpoint
    first_port = int(first.kv_events_config.endpoint.rsplit(":", 1)[1])
    replay_port = int(first.kv_events_config.replay_endpoint.rsplit(":", 1)[1])
    assert replay_port == first_port + 1000


def test_existing_vllm_configs_are_preserved():
    from llumnix.backends.vllm.v1_kv_transfer import configure_v1_kv_transfer

    transfer = object()
    events = object()
    args = SimpleNamespace(kv_transfer_config=transfer, kv_events_config=events)
    cfg = SimpleNamespace(
        migration_backend="kvtransfer",
        migration_backend_transfer_type="P2pNcclConnector",
        kvtransfer_migration_backend_naming_url="",
    )
    assert configure_v1_kv_transfer(args, cfg) is True
    assert args.kv_transfer_config is transfer
    assert args.kv_events_config is events


def test_p2p_request_id_has_required_decode_address():
    from llumnix.backends.vllm.v1_kv_transfer import decorate_p2p_request_id

    request_id = decorate_p2p_request_id("request-1", "10.0.0.8:17000")
    assert request_id == "request-1___decode_addr_10.0.0.8:17000___"
    assert decorate_p2p_request_id(request_id, "10.0.0.8:17000") == request_id


def test_p2p_request_id_rejects_non_address():
    from llumnix.backends.vllm.v1_kv_transfer import decorate_p2p_request_id
    import pytest

    with pytest.raises(ValueError, match="host:port"):
        decorate_p2p_request_id("request-1", "invalid")


def test_p2p_environment_validation(monkeypatch):
    from llumnix.backends.vllm.v1_kv_transfer import validate_p2p_environment
    from vllm.config import KVTransferConfig
    import pytest

    args = SimpleNamespace(kv_transfer_config=KVTransferConfig(
        kv_connector="P2pNcclConnector", kv_role="kv_producer", kv_parallel_size=1
    ))
    with pytest.raises(ValueError, match="kv_parallel_size=2"):
        validate_p2p_environment(args)
    args.kv_transfer_config.kv_parallel_size = 2
    with pytest.raises(ValueError, match="DECODE_ADDRESS"):
        validate_p2p_environment(args)
    monkeypatch.setenv("LLUMNIX_KV_DECODE_ADDRESS", "10.31.10.62:17000")
    validate_p2p_environment(args)


def test_corex_p2p_compat_keeps_p2p_defaults(monkeypatch):
    from llumnix.backends.vllm import v1_kv_transfer

    monkeypatch.setattr(v1_kv_transfer, "corex_nccl_needs_compat", lambda: True)
    args = SimpleNamespace(
        kv_transfer_config=None,
        kv_events_config=None,
        prefix_caching_hash_algo="sha256",
        enable_prefix_caching=False,
    )
    cfg = SimpleNamespace(
        migration_backend="kvtransfer",
        migration_backend_transfer_type="P2pNcclConnector",
        kvtransfer_migration_backend_naming_url="",
    )
    v1_kv_transfer.configure_v1_kv_transfer(args, cfg, instance_id="instance-a")
    assert args.kv_transfer_config.kv_connector == "CoreXP2pNcclConnector"
    assert args.kv_transfer_config.kv_connector_module_path == (
        "llumnix.backends.vllm.corex_p2p_connector"
    )
    assert args.kv_transfer_config.kv_parallel_size == 2


def test_corex_nccl_compat_detection_is_boolean():
    from llumnix.backends.vllm.v1_kv_transfer import corex_nccl_needs_compat

    assert isinstance(corex_nccl_needs_compat(), bool)


def test_strip_p2p_request_id():
    from llumnix.backends.vllm.v1_kv_transfer import strip_p2p_request_id

    assert strip_p2p_request_id("request-1___decode_addr_10.0.0.8:17000___") == "request-1"
    assert strip_p2p_request_id("request-1") == "request-1"
