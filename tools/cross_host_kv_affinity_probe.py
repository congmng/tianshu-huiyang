#!/usr/bin/env python3
"""Deterministic cross-host KV-affinity probe.

This probe exercises the part of the V1 scheduler that is independent of a
GPU: both hosts derive the same prefix hashes, ingest the same stored-block
events, and rank candidate instances identically.  It is intentionally
offline and never downloads a model or starts Ray.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from llumnix.backends.vllm.v1_kv import KVCacheAffinityIndex


@dataclass
class BlockStored:
    block_hashes: tuple[object, ...]
    token_ids: tuple[int, ...] = ()
    block_size: int = 0
    medium: str = "GPU"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--block-size", type=int, default=4)
    args = parser.parse_args()
    tokens = tuple(int(value) for value in args.tokens.split(",") if value)

    index = KVCacheAffinityIndex()
    hashes = index.prefix_hashes(tokens, args.block_size, "sha256_cbor")
    # Simulate a prefix cached on candidate-b and only its first block on a.
    index.apply("candidate-a", [BlockStored(hashes[:1])])
    index.apply("candidate-b", [BlockStored(hashes)])
    print(json.dumps({
        "tokens": tokens,
        "block_size": args.block_size,
        "hashes": [value.hex() if isinstance(value, bytes) else value for value in hashes],
        "affinity": {name: index.affinity(name, hashes) for name in ("candidate-a", "candidate-b")},
        "rank": index.rank(hashes, ("candidate-a", "candidate-b")),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
