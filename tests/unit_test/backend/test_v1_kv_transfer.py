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
    assert transfer.kv_connector == "P2pNcclConnector"
    assert transfer.kv_role == "kv_producer"
    assert transfer.kv_rank == 0
    assert transfer.kv_parallel_size == 2
    assert transfer.kv_ip == "10.0.0.4"
    assert transfer.kv_port == 16000
    assert transfer.kv_connector_extra_config["shared_storage_path"] == "/var/lib/llumnix/kv"
    assert args.kv_events_config.enable_kv_cache_events is True
    assert args.kv_events_config.publisher == "zmq"


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
