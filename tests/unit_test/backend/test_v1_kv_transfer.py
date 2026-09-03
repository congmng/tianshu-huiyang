from types import SimpleNamespace
import socket


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


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


def test_v1_pd_rejects_legacy_backend_before_engine_startup():
    import pytest
    from llumnix.entrypoints.vllm.arg_utils import validate_v1_pd_connector

    manager = SimpleNamespace(enable_pd_disagg=True)
    instance = SimpleNamespace(migration_backend="gloo")
    with pytest.raises(ValueError, match="requires --migration-backend kvtransfer"):
        validate_v1_pd_connector(manager, instance, "0.11.2")
    instance.migration_backend = "kvtransfer"
    validate_v1_pd_connector(manager, instance, "0.11.2")




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


def test_kvtransfer_derives_routable_ip_when_not_overridden(monkeypatch):
    from llumnix.backends.vllm.v1_kv_transfer import configure_v1_kv_transfer

    monkeypatch.delenv("LLUMNIX_KV_IP", raising=False)
    monkeypatch.setattr("socket.gethostname", lambda: "corex-u62")
    monkeypatch.setattr("socket.gethostbyname", lambda _: "10.31.10.62")
    args = SimpleNamespace(kv_transfer_config=None, kv_events_config=None)
    cfg = SimpleNamespace(
        migration_backend="kvtransfer",
        migration_backend_transfer_type="P2pNcclConnector",
        kvtransfer_migration_backend_naming_url="",
    )
    configure_v1_kv_transfer(args, cfg)
    assert args.kv_transfer_config.kv_ip == "10.31.10.62"


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


def test_p2p_endpoint_validation_rejects_invalid_ports():
    from llumnix.backends.vllm.v1_kv_transfer import valid_p2p_endpoint

    assert valid_p2p_endpoint("10.0.0.1:17000")
    assert not valid_p2p_endpoint("10.0.0.1:0")
    assert not valid_p2p_endpoint("10.0.0.1:70000")
    assert not valid_p2p_endpoint("not-an-endpoint")


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
    # Llumnix discovers the decode endpoint only after both P/D actors have
    # started, then carries it in the shared request ID.
    validate_p2p_environment(args)
    monkeypatch.setenv("LLUMNIX_KV_DECODE_ADDRESS", "invalid")
    with pytest.raises(ValueError, match="host:port"):
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


def test_corex_p2p_defaults_to_safe_cpu_staging(monkeypatch):
    from llumnix.backends.vllm.corex_p2p_connector import CoreXP2pNcclConnector
    from vllm.config import KVTransferConfig

    config = KVTransferConfig(
        kv_connector="CoreXP2pNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=18999,
    )
    assert config.get_from_extra_config("corex_transport", "zmq_cpu") == "zmq_cpu"
    assert CoreXP2pNcclConnector.__name__ == "CoreXP2pNcclConnector"


def test_corex_zmq_staging_shutdown_releases_listener():
    from llumnix.backends.vllm.corex_p2p_connector import CoreXZmqP2pEngine

    engine = CoreXZmqP2pEngine(
        local_rank=0,
        config=SimpleNamespace(kv_ip="127.0.0.1", kv_port=0),
    )
    engine.shutdown()
    engine.shutdown()


def test_corex_zmq_staging_keeps_worker_local_rank():
    from llumnix.backends.vllm.corex_p2p_connector import CoreXZmqP2pEngine

    engine = CoreXZmqP2pEngine(
        local_rank=3,
        config=SimpleNamespace(kv_ip="127.0.0.1", kv_port=0),
    )
    try:
        assert str(engine.device) == "cuda:3"
    finally:
        engine.shutdown()


def test_corex_zmq_staging_round_trip_cpu_tensor():
    import time
    import torch
    from llumnix.backends.vllm.corex_p2p_connector import CoreXZmqP2pEngine

    producer = CoreXZmqP2pEngine(
        local_rank=0,
        config=SimpleNamespace(kv_ip="127.0.0.1", kv_port=39101),
    )
    consumer = CoreXZmqP2pEngine(
        local_rank=0,
        config=SimpleNamespace(kv_ip="127.0.0.1", kv_port=39102),
    )
    try:
        expected = torch.arange(12, dtype=torch.float16).reshape(3, 4)
        assert producer.send_tensor(
            "round-trip#layer", expected, "127.0.0.1:39102"
        )
        deadline = time.monotonic() + 2
        while "round-trip#layer" not in consumer.recv_store:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        torch.testing.assert_close(consumer.recv_store["round-trip#layer"], expected)
    finally:
        producer.shutdown()
        consumer.shutdown()


def test_corex_zmq_staging_send_timeout_without_peer():
    import pytest
    import torch
    from llumnix.backends.vllm.corex_p2p_connector import CoreXZmqP2pEngine

    port = _free_port()

    class Config:
        kv_ip = "127.0.0.1"
        kv_port = port

        @staticmethod
        def get_from_extra_config(name, default):
            return 0.05 if name == "zmq_recv_timeout_s" else default

    engine = CoreXZmqP2pEngine(0, Config())
    try:
        with pytest.raises(TimeoutError, match="timed out waiting"):
            engine.send_tensor("missing#layer", torch.ones(1), f"127.0.0.1:{port + 1}")
    finally:
        engine.shutdown()


def test_corex_zmq_staging_receive_timeout_without_tensor():
    import pytest
    from llumnix.backends.vllm.corex_p2p_connector import CoreXZmqP2pEngine

    class Config:
        kv_ip = "127.0.0.1"
        kv_port = _free_port()

        @staticmethod
        def get_from_extra_config(name, default):
            return 0.05 if name == "zmq_recv_timeout_s" else default

    engine = CoreXZmqP2pEngine(0, Config())
    try:
        with pytest.raises(TimeoutError, match="timed out waiting"):
            engine.recv_tensor("missing#layer")
    finally:
        engine.shutdown()


def test_corex_zmq_staging_recovers_after_malformed_payload():
    import msgpack
    import torch
    import zmq
    from llumnix.backends.vllm.corex_p2p_connector import CoreXZmqP2pEngine

    consumer = CoreXZmqP2pEngine(
        local_rank=0,
        config=SimpleNamespace(kv_ip="127.0.0.1", kv_port=39106),
    )
    producer = CoreXZmqP2pEngine(
        local_rank=0,
        config=SimpleNamespace(kv_ip="127.0.0.1", kv_port=39107),
    )
    socket = consumer.context.socket(zmq.DEALER)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.connect("tcp://127.0.0.1:39106")
    try:
        socket.send_multipart([
            msgpack.dumps({
                "cmd": "PUT_CPU",
                "tensor_id": "bad#layer",
                "shape": (4,),
                "dtype": "float16",
            }),
            b"too-short",
        ])
        assert socket.recv() == b"1"
        expected = torch.arange(4, dtype=torch.float16)
        assert producer.send_tensor("good#layer", expected, "127.0.0.1:39106")
        deadline = __import__("time").monotonic() + 2
        while "good#layer" not in consumer.recv_store:
            assert __import__("time").monotonic() < deadline
            __import__("time").sleep(0.01)
        torch.testing.assert_close(consumer.recv_store["good#layer"], expected)
    finally:
        socket.close(linger=0)
        producer.shutdown()
        consumer.shutdown()


def test_corex_compat_rewrites_explicit_vllm_p2p_config(monkeypatch):
    from llumnix.backends.vllm import v1_kv_transfer
    from vllm.config import KVTransferConfig

    monkeypatch.setattr(v1_kv_transfer, "corex_nccl_needs_compat", lambda: True)
    transfer = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_parallel_size=2,
        kv_ip="10.31.10.210",
        kv_port=19052,
    )
    args = SimpleNamespace(
        kv_transfer_config=transfer,
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
    assert transfer.kv_connector == "CoreXP2pNcclConnector"
    assert transfer.kv_connector_module_path == (
        "llumnix.backends.vllm.corex_p2p_connector"
    )
    assert transfer.kv_ip == "10.31.10.210"
    assert transfer.kv_port == 19052


def test_explicit_vllm_p2p_config_is_compatible_without_legacy_backend(monkeypatch):
    from llumnix.backends.vllm import v1_kv_transfer
    from vllm.config import KVTransferConfig

    monkeypatch.setattr(v1_kv_transfer, "corex_nccl_needs_compat", lambda: True)
    transfer = KVTransferConfig(
        kv_connector="P2pNcclConnector", kv_role="kv_consumer", kv_parallel_size=2
    )
    args = SimpleNamespace(
        kv_transfer_config=transfer,
        kv_events_config=None,
        prefix_caching_hash_algo="sha256",
        enable_prefix_caching=False,
    )
    cfg = SimpleNamespace(migration_backend="gloo")
    assert v1_kv_transfer.configure_v1_kv_transfer(args, cfg) is True
    assert transfer.kv_connector == "CoreXP2pNcclConnector"
    assert args.kv_events_config is None


def test_corex_nccl_compat_detection_is_boolean():
    from llumnix.backends.vllm.v1_kv_transfer import corex_nccl_needs_compat

    assert isinstance(corex_nccl_needs_compat(), bool)


def test_strip_p2p_request_id():
    from llumnix.backends.vllm.v1_kv_transfer import strip_p2p_request_id

    assert strip_p2p_request_id("request-1___decode_addr_10.0.0.8:17000___") == "request-1"
    assert strip_p2p_request_id("request-1") == "request-1"


def test_consumer_request_id_has_required_prefill_address():
    from llumnix.backends.vllm.v1_kv_transfer import decorate_p2p_consumer_request_id

    request_id = decorate_p2p_consumer_request_id("request-1", "10.0.0.9:17001")
    assert request_id == "request-1___prefill_addr_10.0.0.9:17001___"
    assert decorate_p2p_consumer_request_id(request_id, "10.0.0.9:17001") == request_id


def test_pd_request_id_is_shared_by_producer_and_consumer():
    from llumnix.backends.vllm.v1_kv_transfer import decorate_p2p_pd_request_id

    value = decorate_p2p_pd_request_id(
        "request-1", "10.0.0.8:17000", "10.0.0.9:17001"
    )
    assert value == (
        "request-1___decode_addr_10.0.0.8:17000___"
        "___prefill_addr_10.0.0.9:17001___"
    )


def test_corex_connector_parses_each_endpoint_from_shared_request_id():
    from llumnix.backends.vllm.corex_p2p_connector import CoreXP2pNcclConnector
    from llumnix.backends.vllm.v1_kv_transfer import decorate_p2p_pd_request_id

    request_id = decorate_p2p_pd_request_id(
        "request-1", "10.0.0.8:17000", "10.0.0.9:17001"
    )
    assert CoreXP2pNcclConnector.parse_request_id(request_id, True) == (
        "10.0.0.8", 17000
    )
    assert CoreXP2pNcclConnector.parse_request_id(request_id, False) == (
        "10.0.0.9", 17001
    )


def test_v1_adapter_advertises_p2p_base_port(monkeypatch):
    from llumnix.backends.vllm.v1_engine import V1EngineAdapter

    adapter = object.__new__(V1EngineAdapter)
    adapter.engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="P2pNcclConnector", kv_port=19052, kv_rank=1
        )
    )
    monkeypatch.setenv("LLUMNIX_KV_IP", "10.31.10.210")
    assert adapter.get_kv_endpoint() == "10.31.10.210:19052"


def test_v1_adapter_uses_explicit_vllm_kv_ip(monkeypatch):
    from llumnix.backends.vllm.v1_engine import V1EngineAdapter

    adapter = object.__new__(V1EngineAdapter)
    adapter.engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="P2pNcclConnector", kv_ip="10.31.10.62", kv_port=19052
        )
    )
    monkeypatch.delenv("LLUMNIX_KV_IP", raising=False)
    assert adapter.get_kv_endpoint() == "10.31.10.62:19052"


def test_pd_producer_sampling_stops_after_handoff():
    from llumnix.backends.vllm.v1_kv_transfer import producer_sampling_params

    params = SimpleNamespace(max_tokens=32, temperature=0.7)
    producer = producer_sampling_params(params)
    assert producer is not params
    assert producer.max_tokens == 1
    assert params.max_tokens == 32
    assert producer.temperature == 0.7
