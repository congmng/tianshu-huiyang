#!/usr/bin/env python3
"""Run the non-destructive Python 3.12/CoreX V1 support gate.

This check deliberately does not start Ray, allocate a GPU, or download a
model.  It is suitable for running on both nodes before a real serving test.
"""

from __future__ import annotations

import json
import platform
import sys
from typing import Mapping


def validate_versions(versions: Mapping[str, str]) -> list[str]:
    """Return actionable errors for versions outside the supported V1 stack."""
    errors = []
    if not versions["python"].startswith("3.12."):
        errors.append(f"Python 3.12 is required, found {versions['python']}")
    if not versions["vllm"].startswith("0.11."):
        errors.append(f"vLLM 0.11.x is required, found {versions['vllm']}")
    if not versions["torch"].startswith("2.7."):
        errors.append(f"CoreX PyTorch 2.7.x is required, found {versions['torch']}")
    if not versions["ray"].startswith("2.52."):
        errors.append(f"CoreX Ray 2.52.x is required, found {versions['ray']}")
    return errors


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
    versions = {
        "python": platform.python_version(),
        "vllm": vllm.__version__,
        "ray": ray.__version__,
        "torch": torch.__version__,
    }
    errors = validate_versions(versions)
    result = {
        **versions,
        "corex_v1_imports": True,
        "affinity_hashes": [value.hex() for value in hashes],
        "affinity_rank": index.rank(hashes, ("candidate-a", "candidate-b")),
        "connector": CoreXP2pNcclConnector.__name__,
        "adapter": V1EngineAdapter.__name__,
        "supported": not errors,
        "errors": errors,
    }
    print(json.dumps(result, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
