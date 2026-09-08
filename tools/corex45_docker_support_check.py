#!/usr/bin/env python3
"""Non-destructive CoreX 4.5 / vLLM 0.23 Docker runtime support gate.

The 4.5 TG-V300 software stack is published as a Docker image, not as a
project-local conda environment with vLLM 0.11.  This gate is intended to be
run inside that image.  It validates the 4.5 device/SDK/vLLM/torch contract
and the checked-out Llumnix source fingerprint, but deliberately does not
claim the 0.11-fork KV migration protocol is present.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_44_gate_module():
    path = PROJECT_ROOT / "tools" / "corex44_support_check.py"
    spec = importlib.util.spec_from_file_location("corex44_support_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect() -> dict[str, object]:
    import torch
    import vllm

    gate44 = _load_44_gate_module()
    source_fingerprint = gate44.source_fingerprint()
    versions = {
        "python": platform.python_version(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "ray": "not-required-in-docker-gate",
    }
    corex_sdk = ""
    release_candidates = (
        Path("/data/tianshu/20260720/corex/release-corex.txt"),
        Path("/usr/local/corex-4.5.0/release-corex.txt"),
    )
    for candidate in release_candidates:
        try:
            corex_sdk = candidate.read_text(encoding="utf-8").strip()
            if corex_sdk:
                break
        except OSError:
            continue
    cuda_available = bool(torch.cuda.is_available())
    runtime = {
        "corex_sdk": corex_sdk,
        "cuda_available": cuda_available,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "",
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    errors: list[str] = []
    if not versions["python"].startswith("3.12."):
        errors.append(f"Python 3.12 is required, found {versions['python']}")
    if not versions["vllm"].startswith("0.23."):
        errors.append(f"CoreX 4.5 vLLM 0.23.x is required, found {versions['vllm']}")
    if not versions["torch"].startswith("2.10."):
        errors.append(f"CoreX 4.5 PyTorch 2.10.x is required, found {versions['torch']}")
    if "4.5.0" not in corex_sdk:
        errors.append(f"CoreX SDK 4.5.0 is required, found {corex_sdk!r}")
    if not cuda_available:
        errors.append("CoreX accelerator is unavailable to PyTorch")
    elif not str(runtime["device_name"]).startswith(
        ("Iluvatar TG-V300", "Iluvatar BI-V300")
    ):
        errors.append(f"CoreX 4.5 V300 device is required, found {runtime['device_name']!r}")
    return {
        **versions,
        **runtime,
        "corex_stack": "45",
        "vllm_engine": "v1",
        "source_fingerprint": source_fingerprint,
        "migration_protocol": "unported-vllm-0.23",
        "supported": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()
    result = collect()
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
