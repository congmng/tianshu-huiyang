import importlib

import pytest


def test_profiling_module_imports_without_analysis_extras():
    profiling = importlib.import_module("llumnix.backends.profiling")
    assert profiling.LatencyMemData({}, {}, {}) is not None


def test_profiling_reports_missing_curve_fit_dependency():
    from llumnix.backends.profiling import ProfilingResult, SimParallelConfig

    result = ProfilingResult("model", {})
    result.add_latency_result(SimParallelConfig(1, 1), "prefill", 1, 8, [1.0])
    with pytest.raises(RuntimeError, match="optional dependency 'scipy'"):
        result.fit_from_database(SimParallelConfig(1, 1))
