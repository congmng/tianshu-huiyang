from llumnix.backends.vllm.v1_kv import KVCacheAffinityIndex
from llumnix.instance_info import DispatchLoadComputation, InstanceInfo
from llumnix.internal_config import GlobalSchedulerConfig
from llumnix.global_scheduler.global_scheduler import GlobalScheduler
from llumnix.arg_utils import InstanceArgs
from llumnix.arg_utils import LlumnixArgumentParser, ManagerArgs


class BlockStored:
    def __init__(self, block_hashes, token_ids, block_size, medium="GPU"):
        self.block_hashes = block_hashes
        self.token_ids = token_ids
        self.block_size = block_size
        self.medium = medium


class BlockRemoved:
    def __init__(self, block_hashes):
        self.block_hashes = block_hashes


class AllBlocksCleared:
    pass


def test_v1_kv_affinity_tracks_store_remove_and_clear():
    index = KVCacheAffinityIndex()
    index.apply("a", [BlockStored([1, 2], [10, 11], 2)])
    index.apply("b", [BlockStored([2], [10, 11], 2)])
    assert index.affinity("a", [1, 2, 3]) == 2 / 3
    assert index.rank([2], ["a", "b"]) == ["a", "b"]
    index.apply("a", [BlockRemoved([1])])
    assert index.block_hashes("a") == frozenset({2})
    index.apply("a", [AllBlocksCleared()])
    assert index.block_hashes("a") == frozenset()


def test_v1_kv_affinity_preserves_default_vllm_byte_hashes():
    index = KVCacheAffinityIndex()
    block_hash = b"vllm-sha256-block-hash-0123456789"
    index.apply("a", [BlockStored([block_hash], [10, 11], 2)])
    assert index.block_hashes("a") == frozenset({block_hash})
    assert index.affinity("a", [block_hash]) == 1.0
    index.apply("a", [BlockRemoved([block_hash])])
    assert index.block_hashes("a") == frozenset()


def test_vllm_zmq_events_reach_affinity_index():
    import socket
    import threading
    import time
    from vllm.config import KVEventsConfig
    from vllm.distributed.kv_events import (
        BlockStored as VllmBlockStored,
        EventPublisherFactory,
        KVEventBatch,
    )
    from llumnix.backends.vllm.v1_kv import KVEventSubscriber

    index = KVCacheAffinityIndex()
    received = threading.Event()

    def apply(events):
        index.apply("instance-a", events)
        received.set()

    publisher = None
    endpoint = None
    for _ in range(10):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        endpoint = f"tcp://*:{port}"
        try:
            publisher = EventPublisherFactory.create(
                KVEventsConfig(enable_kv_cache_events=True, publisher="zmq", endpoint=endpoint)
            )
            break
        except Exception:
            if publisher is not None:
                publisher.shutdown()
            publisher = None
    assert publisher is not None, "unable to bind temporary ZMQ publisher port"
    subscriber = KVEventSubscriber(endpoint, apply)
    try:
        # PUB/SUB subscriptions are asynchronous; allow the subscription to
        # arrive before emitting the only test batch.
        time.sleep(0.1)
        block_hash = b"vllm-zmq-block-hash"
        publisher.publish(KVEventBatch(
            ts=time.time(),
            events=[VllmBlockStored([block_hash], None, [1, 2], 2, None, "GPU")],
        ))
        assert received.wait(3.0)
        assert index.affinity("instance-a", [block_hash]) == 1.0
    finally:
        subscriber.close()
        publisher.shutdown()


def test_vllm_zmq_replay_rebuilds_affinity_index():
    import socket
    import threading
    import time
    from vllm.config import KVEventsConfig
    from vllm.distributed.kv_events import (
        BlockStored as VllmBlockStored,
        EventPublisherFactory,
        KVEventBatch,
    )
    from llumnix.backends.vllm.v1_kv import KVEventSubscriber

    # Port probing has a small TOCTOU window (another Ray/ZMQ test may claim
    # the port after the probe socket closes).  Retry the complete publisher
    # bind instead of making the suite fail on transient EADDRINUSE.
    publisher = None
    endpoint = replay_endpoint = None
    for _ in range(10):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            pub_port = probe.getsockname()[1]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            replay_port = probe.getsockname()[1]
        endpoint = f"tcp://*:{pub_port}"
        replay_endpoint = f"tcp://*:{replay_port}"
        try:
            publisher = EventPublisherFactory.create(KVEventsConfig(
                enable_kv_cache_events=True,
                publisher="zmq",
                endpoint=endpoint,
                replay_endpoint=replay_endpoint,
            ))
            break
        except Exception:
            if publisher is not None:
                publisher.shutdown()
            publisher = None
    assert publisher is not None, "unable to bind temporary ZMQ publisher ports"
    block_hash = b"vllm-replay-block-hash"
    publisher.publish(KVEventBatch(
        ts=time.time(),
        events=[VllmBlockStored([block_hash], None, [1, 2], 2, None, "GPU")],
    ))
    # The publisher thread owns the replay buffer; wait until it processes
    # this pre-subscriber event before constructing the subscriber.
    time.sleep(0.2)
    index = KVCacheAffinityIndex()
    replayed = threading.Event()

    def apply(events):
        index.apply("instance-a", events)
        replayed.set()

    subscriber = KVEventSubscriber(endpoint, apply, replay_endpoint=replay_endpoint)
    try:
        assert replayed.wait(3.0)
        assert index.affinity("instance-a", [block_hash]) == 1.0
    finally:
        subscriber.close()
        publisher.shutdown()


def test_virtual_usage_uses_reported_heterogeneous_memory_and_capacity():
    calculator = DispatchLoadComputation("virtual_usage")
    fast = InstanceInfo(
        instance_id="fast",
        num_running_requests=8,
        gpu_memory_total_bytes=32 * 1024**3,
        gpu_memory_free_bytes=24 * 1024**3,
        compute_capacity=2.0,
    )
    constrained = InstanceInfo(
        instance_id="constrained",
        num_running_requests=8,
        gpu_memory_total_bytes=16 * 1024**3,
        gpu_memory_free_bytes=2 * 1024**3,
        compute_capacity=0.5,
    )
    assert calculator.compute_instance_load(fast) < calculator.compute_instance_load(constrained)


def test_virtual_usage_keeps_legacy_block_counter_fallback():
    calculator = DispatchLoadComputation("virtual_usage")
    info = InstanceInfo(
        instance_id="legacy",
        num_running_requests=1,
        num_total_gpu_blocks=100,
        num_used_gpu_blocks=50,
    )
    assert calculator.compute_instance_load(info) > 0


def test_global_scheduler_uses_v1_heterogeneous_load_and_kv_affinity_together():
    scheduler = GlobalScheduler(GlobalSchedulerConfig(
        0, "load", 1, "balanced", 1.0, "avg_load", "remaining_steps",
        1.0, 0.0, False, False,
    ))
    cached = InstanceInfo(
        instance_id="cached", num_running_requests=8,
        gpu_memory_total_bytes=32 * 1024**3,
        gpu_memory_free_bytes=24 * 1024**3,
        compute_capacity=2.0,
        kv_cache_block_hashes=frozenset({b"prefix"}),
    )
    uncached = InstanceInfo(
        instance_id="uncached", num_running_requests=1,
        gpu_memory_total_bytes=32 * 1024**3,
        gpu_memory_free_bytes=31 * 1024**3,
        compute_capacity=1.0,
    )
    calculator = DispatchLoadComputation("virtual_usage")
    cached.dispatch_load_metric = calculator.compute_instance_load(cached)
    uncached.dispatch_load_metric = calculator.compute_instance_load(uncached)
    assert uncached.dispatch_load_metric < cached.dispatch_load_metric
    scheduler.scale_up(["cached", "uncached"], [
        InstanceArgs(instance_type="no_constraints"),
        InstanceArgs(instance_type="no_constraints"),
    ])
    scheduler.update_instance_infos([cached, uncached])
    # Cache affinity is used only among sufficiently healthy nodes. The two
    # computed heterogeneous loads are within the 0.10 safety window, so the
    # stored prefix selects ``cached`` rather than blindly using load alone.
    instance_id, _ = scheduler.dispatch([b"prefix"])
    assert instance_id == "cached"


def test_global_scheduler_v1_load_drives_scale_up_signal():
    scheduler = GlobalScheduler(GlobalSchedulerConfig(
        0, "load", 1, "balanced", 1.0, "avg_load", "virtual_usage",
        0.20, -1.0, False, False,
    ))
    info = InstanceInfo(
        instance_id="busy-v1", num_running_requests=96,
        gpu_memory_total_bytes=32 * 1024**3,
        gpu_memory_free_bytes=4 * 1024**3,
        compute_capacity=1.0,
    )
    scheduler.scale_up("busy-v1", [InstanceArgs(instance_type="no_constraints")])
    scheduler.update_instance_infos([info])
    scale_up, scale_down = scheduler.check_scale()
    assert scale_up == 1
    assert scale_down == 0


def test_v1_virtual_usage_is_exposed_as_scaling_cli_metric():
    parser = LlumnixArgumentParser()
    ManagerArgs.add_cli_args(parser)
    action = next(a for a in parser._actions if a.dest == "scaling_load_metric")
    assert "virtual_usage" in action.choices


def test_v1_scaling_flag_is_not_rejected_by_legacy_validation():
    """V1 reactive scaling is implemented and must remain CLI-addressable."""
    from llumnix.arg_utils import ManagerArgs, LlumnixArgumentParser

    parser = LlumnixArgumentParser()
    ManagerArgs.add_cli_args(parser)
    args = ManagerArgs(enable_scaling=True, min_instances=1, max_instances=2)
    ManagerArgs.check_args(args, parser)


def test_scaling_scheduler_ignores_empty_or_stale_instance_snapshots():
    from llumnix.global_scheduler.scaling_scheduler import ScalingScheduler

    scheduler = ScalingScheduler(10, 60, "avg_load", "virtual_usage", False)
    scheduler.num_instances = 1
    scheduler.instance_id_set = {"gone"}
    scheduler.update_instance_infos({})
    assert scheduler.check_scale() == (0, 0)
    scheduler.update_instance_infos({"other": InstanceInfo(instance_id="other")})
    assert scheduler.check_scale() == (0, 0)
