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
import re
import subprocess
import hashlib
import os
import importlib.util
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
    "llumnix/backends/vllm/v1_migration.py",
    "llumnix/backends/utils.py",
    "llumnix/global_scheduler/dispatch_scheduler.py",
    "llumnix/global_scheduler/global_scheduler.py",
    "llumnix/global_scheduler/scaling_scheduler.py",
    "llumnix/global_scheduler/scaling_policy.py",
    "llumnix/manager.py",
    "llumnix/launcher.py",
    "llumnix/llumlet/llumlet.py",
    "llumnix/entrypoints/vllm/arg_utils.py",
    "llumnix/entrypoints/vllm/client.py",
    "llumnix/entrypoints/vllm/v1_api_server.py",
    "llumnix/instance_info.py",
    # Keep both hosts on the same executable validation contract as well as
    # the same serving implementation. These runners exercise the public V1
    # boundary and must not silently drift between deployment nodes.
    "tools/run_corex44_validation.py",
    "tools/run_llumnix_v1_http_e2e.py",
    "tools/run_v1_true_kv_migration_service.py",
    "tools/v1_p2p_model_probe.py",
    "tools/corex44_native_nccl_probe.py",
    "tools/corex_env.sh",
    "tools/corex44_env.sh",
    "tools/corex45_env.sh",
    "configs/corex44_v1_pd.yml",
    "docs/vLLM_V1_True_KV_Migration_Plan.md",
)



CORE_X_STACKS = {
    "44": {
        "sdk_marker": "4.4.0",
        "device_names": ("Iluvatar BI-V150",),
        "torch_prefixes": ("2.7.",),
        "vllm_prefixes": ("0.11.",),
        "ray_prefixes": ("2.52.",),
    },
    "45": {
        "sdk_marker": "4.5.0",
        "device_names": ("Iluvatar TG-V300", "Iluvatar BI-V300"),
        "torch_prefixes": ("2.10.",),
        # CoreX 4.5 is published in both the 0.23 Docker image and the
        # project-local 0.25 conda environment.  The serving boundary is V1
        # in both releases; migration support remains explicitly unported.
        "vllm_prefixes": ("0.23.", "0.25."),
        "ray_prefixes": ("2.56.",),
    },
}


def normalize_stack(value: str | None) -> str:
    """Return the canonical ``44`` or ``45`` stack key from a CLI value."""
    if value is None:
        value = os.getenv("LLUMNIX_COREX_STACK", "44")
    value = str(value)
    if value.startswith("4.4"):
        return "44"
    if value.startswith("4.5"):
        return "45"
    if value in CORE_X_STACKS:
        return value
    raise ValueError(f"unsupported CoreX stack: {value!r}")


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FINGERPRINT_FILES:
        digest.update(relative.encode())
        digest.update((PROJECT_ROOT / relative).read_bytes())
    return digest.hexdigest()


def validate_versions(versions: Mapping[str, str], stack: str = "44") -> list[str]:
    """Return actionable errors for versions outside a supported V1 stack."""
    stack = normalize_stack(stack)
    stack_spec = CORE_X_STACKS[stack]
    errors = []
    if not versions["python"].startswith("3.12."):
        errors.append(f"Python 3.12 is required, found {versions['python']}")
    if not any(versions["vllm"].startswith(prefix)
               for prefix in stack_spec["vllm_prefixes"]):
        errors.append(
            f"CoreX {stack} vLLM {stack_spec['vllm_prefixes']} is required, "
            f"found {versions['vllm']}"
        )
    if not any(versions["torch"].startswith(prefix)
               for prefix in stack_spec["torch_prefixes"]):
        errors.append(
            f"CoreX {stack} PyTorch {stack_spec['torch_prefixes']} is required, "
            f"found {versions['torch']}"
        )
    if not any(versions["ray"].startswith(prefix)
               for prefix in stack_spec["ray_prefixes"]):
        errors.append(
            f"CoreX {stack} Ray {stack_spec['ray_prefixes']} is required, "
            f"found {versions['ray']}"
        )
    return errors


def validate_corex_runtime(runtime: Mapping[str, object], stack: str = "44") -> list[str]:
    """Validate that the supported Python stack is actually CoreX-backed."""
    stack = normalize_stack(stack)
    stack_spec = CORE_X_STACKS[stack]
    errors = []
    sdk = str(runtime.get("corex_sdk", ""))
    if stack_spec["sdk_marker"] not in sdk:
        errors.append(
            f"CoreX SDK {stack_spec['sdk_marker']} is required, found {sdk!r}"
        )
    if not bool(runtime.get("cuda_available", False)):
        errors.append("CoreX accelerator is unavailable to PyTorch")
    device_name = str(runtime.get("device_name", ""))
    if not device_name.startswith("Iluvatar"):
        errors.append(f"Iluvatar CoreX device is required, found {device_name!r}")
    elif not device_name.startswith(stack_spec["device_names"]):
        errors.append(
            f"CoreX {stack} device {stack_spec['device_names']} is required, "
            f"found {device_name!r}"
        )
    return errors


def compare_hosts(local: Mapping[str, object], remote: Mapping[str, object],
                 mixed_stack: bool = False) -> list[str]:
    """Return mismatches that invalidate a multi-host scheduling gate.

    By default a mixed V150/4.4 and V300/4.5 deployment is allowed; the
    per-node SDK/device/torch/vLLM/Ray fields are intentionally allowed to
    differ because the stacks serve the same API through different vendor
    runtimes.  Source code and KV hash affinity must still match so the two
    node classes can coexist in one dispatch pool.
    """
    mismatches = []
    if mixed_stack:
        common = ("python", "affinity_hashes", "source_fingerprint")
    else:
        common = ("python", "vllm", "ray", "torch", "corex_sdk",
                  "cuda_available", "device_name", "affinity_hashes",
                  "source_fingerprint", "migration_protocol_version")
    for key in common:
        if local.get(key) != remote.get(key):
            mismatches.append(f"{key} differs: local={local.get(key)!r} remote={remote.get(key)!r}")
    if not remote.get("supported", False):
        mismatches.extend(str(error) for error in remote.get("errors", []))
    return mismatches


def collect_result(stack: str = "44") -> dict[str, object]:
    import torch
    import vllm
    import ray

    from llumnix.backends.vllm.v1_engine import V1EngineAdapter
    from llumnix.backends.vllm.v1_kv import KVCacheAffinityIndex

    stack = normalize_stack(stack)
    migration_available = (
        importlib.util.find_spec("vllm.v1.migration") is not None
    )
    if migration_available:
        from vllm.v1.migration import MIGRATION_PROTOCOL_VERSION
        migration_protocol_version = MIGRATION_PROTOCOL_VERSION
        migration_protocol = f"vllm-{vllm.__version__}-{MIGRATION_PROTOCOL_VERSION}"
    else:
        # CoreX 4.5 ships a later vLLM V1 without the 0.11 migration fork.
        # It is servable, but must never be advertised as migration-ready.
        migration_protocol_version = 0
        migration_protocol = "unported-vllm-0.23"
    try:
        from llumnix.backends.vllm.corex_p2p_connector import CoreXP2pNcclConnector
        connector_name = CoreXP2pNcclConnector.__name__
    except Exception:
        connector_name = ""

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
    stack_root = os.getenv(
        "LLUMNIX_COREX_ROOT", f"/usr/local/corex-4.{'5' if stack == '45' else '4'}.0"
    )
    release_candidates = []
    release_env = os.getenv("LLUMNIX_COREX_RELEASE_FILE")
    if release_env:
        release_candidates.append(release_env)
    if stack == "45":
        # CoreX 4.5 V300 nodes publish the release marker under the shared
        # vendor tree rather than the SDK prefix used by the 4.4 image.
        release_candidates.append(
            "/data/tianshu/20260720/corex/release-corex.txt"
        )
    release_candidates.append(f"{stack_root}/release-corex.txt")
    corex_sdk = ""
    for candidate in release_candidates:
        try:
            corex_sdk = Path(candidate).read_text(encoding="utf-8").strip()
            if corex_sdk:
                break
        except OSError:
            continue
    runtime = {
        "corex_sdk": corex_sdk,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "cuda_device_count": torch.cuda.device_count(),
    }
    errors = validate_versions(versions, stack) + validate_corex_runtime(runtime, stack)
    result = {
        **versions,
        **runtime,
        "corex_stack": stack,
        "corex_v1_imports": migration_available,
        "migration_protocol_version": migration_protocol_version,
        "migration_protocol": migration_protocol,
        "affinity_hashes": [value.hex() for value in hashes],
        "affinity_rank": index.rank(hashes, ("candidate-a", "candidate-b")),
        "connector": connector_name,
        "adapter": V1EngineAdapter.__name__,
        "source_fingerprint": source_fingerprint(),
        "supported": not errors,
        "errors": errors,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corex-stack", default=os.getenv("LLUMNIX_COREX_STACK", "44"),
                        help="local CoreX stack: 44 or 45")
    parser.add_argument("--remote-host", help="SSH host to check as a second node")
    parser.add_argument("--remote-project", default="/data1/congmng/llumnix")
    parser.add_argument("--remote-stack", default=None,
                        help="remote CoreX stack; defaults to --corex-stack")
    parser.add_argument("--mixed-stack", action="store_true",
                        help="allow V150/4.4 and V300/4.5 nodes in one gate")
    parser.add_argument("--ssh-password", default=None,
                        help="optional password for sshpass-based remote login")
    args = parser.parse_args()
    stack = normalize_stack(args.corex_stack)
    remote_stack = normalize_stack(args.remote_stack or stack)
    result = collect_result(stack)
    errors = list(result["errors"])
    if args.remote_host:
        remote_cli = (
            f"cd {args.remote_project} && LLUMNIX_COREX_STACK={remote_stack} "
            "source tools/corex_env.sh && PYTHONPATH=. "
            "python tools/corex44_support_check.py "
            f"--corex-stack {remote_stack}"
        )
        ssh_command = ["ssh", "-o", "StrictHostKeyChecking=no",
                       "-o", "UserKnownHostsFile=/dev/null"]
        if args.ssh_password:
            ssh_command = ["sshpass", "-p", args.ssh_password, *ssh_command]
        else:
            ssh_command.extend(["-o", "BatchMode=yes"])
        ssh_command.extend([args.remote_host, remote_cli])
        completed = subprocess.run(
            ssh_command, check=False, capture_output=True, text=True,
        )
        if completed.returncode:
            errors.append(f"remote gate failed with exit code {completed.returncode}")
            if completed.stderr:
                errors.append(completed.stderr.strip().splitlines()[-1])
        else:
            remote = json.loads(completed.stdout.strip().splitlines()[-1])
            errors.extend(compare_hosts(
                result, remote,
                mixed_stack=args.mixed_stack or stack != remote_stack,
            ))
            result["remote"] = remote
    result["supported"] = not errors
    result["errors"] = errors
    print(json.dumps(result, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
