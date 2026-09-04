#!/usr/bin/env python3
"""Run the non-destructive Python 3.12/CoreX V1 support gate.

This check deliberately does not start Ray, allocate a GPU, or download a
model.  It is suitable for running on both nodes before a real serving test.
"""

from __future__ import annotations

import json
import platform
import sys
import argparse
import subprocess
import hashlib
from pathlib import Path
from typing import Mapping


# The gate is intentionally usable as ``python /path/to/script`` after the
# CoreX environment is sourced.  Activating that environment need not set
# PYTHONPATH, so make the checked-out project importable by construction.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_FINGERPRINT_FILES = (
    "llumnix/backends/vllm/v1_engine.py",
    "llumnix/backends/vllm/v1_kv.py",
    "llumnix/backends/vllm/v1_kv_transfer.py",
    "llumnix/backends/vllm/corex_p2p_connector.py",
    "llumnix/global_scheduler/dispatch_scheduler.py",
    "llumnix/global_scheduler/global_scheduler.py",
    "llumnix/instance_info.py",
)


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FINGERPRINT_FILES:
        digest.update(relative.encode())
        digest.update((PROJECT_ROOT / relative).read_bytes())
    return digest.hexdigest()


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


def compare_hosts(local: Mapping[str, object], remote: Mapping[str, object]) -> list[str]:
    """Return mismatches that invalidate a deterministic two-host gate."""
    mismatches = []
    for key in ("python", "vllm", "ray", "torch", "affinity_hashes", "source_fingerprint"):
        if local.get(key) != remote.get(key):
            mismatches.append(f"{key} differs: local={local.get(key)!r} remote={remote.get(key)!r}")
    if not remote.get("supported", False):
        mismatches.extend(str(error) for error in remote.get("errors", []))
    return mismatches


def collect_result() -> dict[str, object]:
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
        "source_fingerprint": source_fingerprint(),
        "supported": not errors,
        "errors": errors,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", help="SSH host to check as a second node")
    parser.add_argument("--remote-project", default="/data1/congmng/llumnix")
    args = parser.parse_args()
    result = collect_result()
    errors = list(result["errors"])
    if args.remote_host:
        command = (
            f"cd {args.remote_project} && source tools/corex44_env.sh && "
            "PYTHONPATH=. python tools/corex44_support_check.py"
        )
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.remote_host, command],
            check=False, capture_output=True, text=True,
        )
        if completed.returncode:
            errors.append(f"remote gate failed with exit code {completed.returncode}")
        else:
            remote = json.loads(completed.stdout.strip().splitlines()[-1])
            errors.extend(compare_hosts(result, remote))
            result["remote"] = remote
    result["supported"] = not errors
    result["errors"] = errors
    print(json.dumps(result, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
