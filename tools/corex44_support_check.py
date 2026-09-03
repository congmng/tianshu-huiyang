#!/usr/bin/env python3
"""Run the non-destructive Python 3.12/CoreX V1 support gate.

This check deliberately does not start Ray, allocate a GPU, or download a
model.  It is suitable for running on both nodes before a real serving test.
"""

from __future__ import annotations

import json
import platform
import sys


def main() -> None:
    import torch
    import vllm
    import ray

    from llumnix.backends.vllm.corex_p2p_connector import CoreXP2pNcclConnector
    from llumnix.backends.vllm.v1_engine import V1EngineAdapter
    from llumnix.backends.vllm.v1_kv import KVCacheAffinityIndex

    tokens = (1, 2, 3, 4, 5, 6, 7, 8)
    hashes = KVCacheAffinityIndex().prefix_hashes(tokens, 4, "sha256_cbor")
    index = KVCacheAffinityIndex()

    class Event:
        block_hashes = hashes

    index.apply("candidate-a", [Event()])
    result = {
        "python": platform.python_version(),
        "vllm": vllm.__version__,
        "ray": ray.__version__,
        "torch": torch.__version__,
        "corex_v1_imports": True,
        "affinity_hashes": [value.hex() for value in hashes],
        "affinity_rank": index.rank(hashes, ("candidate-a", "candidate-b")),
        "connector": CoreXP2pNcclConnector.__name__,
        "adapter": V1EngineAdapter.__name__,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
