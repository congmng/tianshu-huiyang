from llumnix.backends.vllm.v1_kv import KVCacheAffinityIndex


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

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"tcp://*:{port}"
    index = KVCacheAffinityIndex()
    received = threading.Event()

    def apply(events):
        index.apply("instance-a", events)
        received.set()

    publisher = EventPublisherFactory.create(
        KVEventsConfig(enable_kv_cache_events=True, publisher="zmq", endpoint=endpoint)
    )
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

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        pub_port = probe.getsockname()[1]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        replay_port = probe.getsockname()[1]
    endpoint = f"tcp://*:{pub_port}"
    replay_endpoint = f"tcp://*:{replay_port}"
    publisher = EventPublisherFactory.create(KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=endpoint,
        replay_endpoint=replay_endpoint,
    ))
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
