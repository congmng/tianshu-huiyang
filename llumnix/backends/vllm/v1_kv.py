"""vLLM V1 KV-cache event and affinity primitives.

vLLM 0.11 no longer exposes the 0.6 block manager used by the original
Llumnix migration coordinator.  V1 does expose stable cache events, however;
this module turns those events into a small, engine-independent index that can
be used by dispatch and migration planners.  It intentionally does not copy
device memory: transfer execution is kept behind a separate connector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence
import hashlib
import threading
import zmq
import msgspec


class KVEventSubscriber:
    """Subscribe to vLLM V1's ZMQ KV-event publisher.

    vLLM publishes ``KVEventBatch`` frames as ``topic, sequence, payload``.
    The subscriber is deliberately independent of Ray and invokes a callback
    on a daemon thread, keeping event handling off the engine event loop.
    """

    def __init__(self, endpoint: str, callback: Callable[[object], None], topic: str = ""):
        if not endpoint:
            raise ValueError("KV event subscriber endpoint must not be empty")
        self._endpoint = self._connect_endpoint(endpoint)
        self._callback = callback
        self._topic = topic.encode("utf-8")
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, self._topic)
        self._socket.connect(self._endpoint)
        # vLLM's tagged msgspec structs preserve event types when decoded with
        # the concrete KVEventBatch type; plain dict decoding would lose the
        # BlockStored/BlockRemoved tag needed by KVCacheAffinityIndex.apply.
        try:
            from vllm.distributed.kv_events import KVEventBatch
            self._decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        except ImportError:
            self._decoder = msgspec.msgpack.Decoder()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="llumnix-kv-events", daemon=True)
        self._thread.start()

    @staticmethod
    def _connect_endpoint(endpoint: str) -> str:
        # A publisher commonly binds tcp://*:PORT. A subscriber must connect
        # to a concrete address; localhost is correct for the colocated V1
        # scheduler/adapter process.
        if endpoint.startswith("tcp://*:"):
            return "tcp://127.0.0.1:" + endpoint.rsplit(":", 1)[1]
        if endpoint.startswith("tcp://0.0.0.0:"):
            return "tcp://127.0.0.1:" + endpoint.rsplit(":", 1)[1]
        return endpoint

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if not self._socket.poll(100):
                    continue
                frames = self._socket.recv_multipart()
                if len(frames) != 3:
                    continue
                batch = self._decoder.decode(frames[2])
                self._callback(batch.events if hasattr(batch, "events") else batch)
            except (zmq.ZMQError, msgspec.DecodeError, ValueError):
                if not self._stop.is_set():
                    continue
            except Exception:
                # Event loss must never take down inference. The publisher's
                # replay endpoint can be used by a future durable consumer.
                continue

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._socket.close(linger=0)


@dataclass(frozen=True)
class KVBlockLocation:
    """A hashable cache block known to an instance."""

    instance_id: str
    block_hash: object
    token_ids: tuple[int, ...] = ()
    block_size: int = 0
    medium: str = "GPU"
    last_seen_step: int = 0


@dataclass
class KVCacheAffinityIndex:
    """Track prefix blocks and estimate reuse on candidate instances.

    ``BlockStored`` events are authoritative for ownership.  A block removed
    from an instance is deleted only for that instance, allowing the same
    prefix to be cached independently on multiple workers.
    """

    # vLLM V1 emits either 32-byte ``sha256`` hashes or compact integer hashes
    # (controlled by VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES). Keep both forms
    # losslessly; coercing bytes to int would make the default event stream
    # unusable.
    _blocks: dict[str, dict[object, KVBlockLocation]] = field(default_factory=dict)
    _step: int = 0

    def apply(self, instance_id: str, events: Iterable[object]) -> None:
        blocks = self._blocks.setdefault(instance_id, {})
        for event in events:
            name = type(event).__name__
            if name == "BlockStored":
                hashes = tuple(getattr(event, "block_hashes", ()))
                tokens = tuple(getattr(event, "token_ids", ()))
                block_size = int(getattr(event, "block_size", 0) or 0)
                medium = getattr(event, "medium", None) or "GPU"
                for block_hash in hashes:
                    normalized_hash = bytes(block_hash) if isinstance(block_hash, (bytes, bytearray)) else int(block_hash)
                    blocks[normalized_hash] = KVBlockLocation(
                        instance_id, normalized_hash, tokens, block_size,
                        medium, self._step)
            elif name == "BlockRemoved":
                for block_hash in getattr(event, "block_hashes", ()):
                    normalized_hash = bytes(block_hash) if isinstance(block_hash, (bytes, bytearray)) else int(block_hash)
                    blocks.pop(normalized_hash, None)
            elif name == "AllBlocksCleared":
                blocks.clear()
        self._step += 1

    def block_hashes(self, instance_id: str) -> frozenset[object]:
        return frozenset(self._blocks.get(instance_id, {}))

    def prefix_hashes(
        self,
        token_ids: Sequence[int],
        block_size: int,
        hash_algo: str = "sha256_cbor",
    ) -> tuple[object, ...]:
        """Return hashes for complete blocks in a token prefix.

        This helper is deliberately caller-supplied-hash agnostic.  vLLM's
        event already contains hashes; callers can use this method to compare
        a request's known block-hash sequence with the index.
        """
        if block_size <= 0:
            return ()
        full_tokens = len(token_ids) - (len(token_ids) % block_size)
        if full_tokens <= 0:
            return ()
        try:
            from vllm.utils.hashing import get_hash_fn_by_name
            from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash

            hash_fn = get_hash_fn_by_name(hash_algo)
            # vLLM derives the initial hash from PYTHONHASHSEED. The V1
            # connector configuration sets this to a fixed value so that the
            # API process and EngineCore agree across hosts.
            # EngineCore initializes NONE_HASH once per process. Avoid
            # re-seeding it here when it has already been initialized so the
            # generated values match vLLM's scheduler exactly.
            import vllm.v1.core.kv_cache_utils as kv_utils
            if not hasattr(kv_utils, "NONE_HASH"):
                init_none_hash(hash_fn)
            parent = None
            hashes = []
            for offset in range(0, full_tokens, block_size):
                parent = hash_block_tokens(
                    hash_fn, parent, token_ids[offset : offset + block_size], None
                )
                hashes.append(parent)
            return tuple(hashes)
        except ImportError:
            # Keep the index usable by the simulator/legacy Python installs.
            parent = b""
            hashes = []
            for offset in range(0, full_tokens, block_size):
                parent = hashlib.blake2b(
                    parent + repr(tuple(token_ids[offset : offset + block_size])).encode(),
                    digest_size=32,
                ).digest()
                hashes.append(parent)
            return tuple(hashes)

    def affinity(self, instance_id: str, block_hashes: Iterable[int]) -> float:
        requested = {
            bytes(value) if isinstance(value, (bytes, bytearray)) else int(value)
            for value in block_hashes
        }
        if not requested:
            return 0.0
        return len(requested & self.block_hashes(instance_id)) / len(requested)

    def rank(self, block_hashes: Iterable[int], candidates: Iterable[str]) -> list[str]:
        """Rank candidates by prefix reuse, then deterministic instance id."""
        return sorted(candidates,
                      key=lambda instance_id: (-self.affinity(instance_id, block_hashes),
                                               instance_id))

    def snapshot(self) -> Mapping[str, frozenset[object]]:
        return {instance_id: self.block_hashes(instance_id)
                for instance_id in self._blocks}
