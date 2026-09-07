#!/usr/bin/env python3
"""Check that two hosts use the same V1 migration protocol and fork commit.

This is intentionally read-only: it never changes the remote Python
environment or starts a worker.
"""
from __future__ import annotations

import argparse
import subprocess


def probe(host: str, fork: str) -> tuple[str, str]:
    command = (
        "source /data1/congmng/llumnix/tools/corex44_env.sh 2>/dev/null || true; "
        f"PYTHONPATH={fork} python -c '"
        "import hashlib, pathlib, vllm; "
        "from vllm.v1.migration import MIGRATION_PROTOCOL_VERSION; "
        f"root=pathlib.Path(\"{fork}\"); "
        "files=(root / \"vllm/v1/migration.py\", root / \"vllm/v1/request.py\", "
        "root / \"vllm/v1/engine/core.py\"); "
        "digest=hashlib.sha256(b\"\".join(p.read_bytes() for p in files)).hexdigest(); "
        "print(digest); print(vllm.__version__, MIGRATION_PROTOCOL_VERSION)'"
    )
    if host in {"localhost", "127.0.0.1"}:
        out = subprocess.check_output(["bash", "-lc", command], text=True)
    else:
        out = subprocess.check_output(["ssh", host, command], text=True)
    lines = out.strip().splitlines()
    if len(lines) != 2:
        raise RuntimeError(f"unexpected support-gate response from {host}: {out!r}")
    return tuple(lines)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-host", default="localhost")
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--fork", default="/data1/congmng/vllm-corex44-v1-migration")
    args = parser.parse_args()
    source = probe(args.source_host, args.fork)
    target = probe(args.target_host, args.fork)
    if source != target:
        raise SystemExit(f"SUPPORT_GATE_FAIL source={source} target={target}")
    print(f"SUPPORT_GATE_PASS migration_digest={source[0]} version_protocol={source[1]}")


if __name__ == "__main__":
    main()
