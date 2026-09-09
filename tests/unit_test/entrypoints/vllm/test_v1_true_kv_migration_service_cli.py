import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
SERVICE_TOOL = ROOT / "tools" / "run_v1_true_kv_migration_service.py"


def _load_service_tool():
    spec = importlib.util.spec_from_file_location(
        "run_v1_true_kv_migration_service", SERVICE_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_service_gpu_ids_derive_two_tp_groups():
    module = _load_service_tool()
    assert module.migration_gpu_ids(1, "") == "0,1"
    assert module.migration_gpu_ids(2, "") == "0,1,2,3"
    assert module.migration_gpu_ids(3, "") == "0,1,2,3,4,5"


def test_migration_service_gpu_ids_preserve_explicit_selection():
    module = _load_service_tool()
    assert module.migration_gpu_ids(2, "4,5,6,7") == "4,5,6,7"


def test_migration_service_gpu_ids_reject_invalid_tp():
    module = _load_service_tool()
    with pytest.raises(ValueError, match=">= 1"):
        module.migration_gpu_ids(0, "")
