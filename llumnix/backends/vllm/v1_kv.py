"""vLLM V1 KV-cache event and affinity primitives.

vLLM 0.11 no longer exposes the 0.6 block manager used by the original
Llumnix migration coordinator.  V1 does expose stable cache events, however;
this module turns those events into a small, engine-independent index that can
be used by dispatch and migration planners.  It intentionally does not copy
device memory: transfer execution is kept behind a separate connector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class KVBlockLocation:
    """A hashable cache block known to an instance."""

    instance_id: str
    block_hash: int
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

    _blocks: dict[str, dict[int, KVBlockLocation]] = field(default_factory=dict)
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
                    blocks[int(block_hash)] = KVBlockLocation(
                        instance_id, int(block_hash), tokens, block_size,
                        medium, self._step)
            elif name == "BlockRemoved":
                for block_hash in getattr(event, "block_hashes", ()):
                    blocks.pop(int(block_hash), None)
            elif name == "AllBlocksCleared":
                blocks.clear()
        self._step += 1

    def block_hashes(self, instance_id: str) -> frozenset[int]:
        return frozenset(self._blocks.get(instance_id, {}))

    def prefix_hashes(self, token_ids: Sequence[int], block_size: int) -> tuple[int, ...]:
        """Return hashes for complete blocks in a token prefix.

        This helper is deliberately caller-supplied-hash agnostic.  vLLM's
        event already contains hashes; callers can use this method to compare
        a request's known block-hash sequence with the index.
        """
        if block_size <= 0:
            return ()
        return tuple(hash(tuple(token_ids[i:i + block_size]))
                     for i in range(0, len(token_ids) - block_size + 1, block_size))

    def affinity(self, instance_id: str, block_hashes: Iterable[int]) -> float:
        requested = {int(value) for value in block_hashes}
        if not requested:
            return 0.0
        return len(requested & self.block_hashes(instance_id)) / len(requested)

    def rank(self, block_hashes: Iterable[int], candidates: Iterable[str]) -> list[str]:
        """Rank candidates by prefix reuse, then deterministic instance id."""
        return sorted(candidates,
                      key=lambda instance_id: (-self.affinity(instance_id, block_hashes),
                                               instance_id))

    def snapshot(self) -> Mapping[str, frozenset[int]]:
        return {instance_id: self.block_hashes(instance_id)
                for instance_id in self._blocks}

