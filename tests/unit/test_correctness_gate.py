"""Correctness gate tests (spec 0002)."""

from autokernel_pbt.harness.correctness import DEFAULT_STAGES, should_run_benchmark
from autokernel_pbt.harness.result import HarnessResultBuilder


def test_default_stage_names_match_spec():
    assert DEFAULT_STAGES == [
        "smoke",
        "shape_sweep",
        "numerical_stress",
        "determinism",
        "edge_cases",
    ]


def test_benchmark_skipped_on_failure():
    builder = HarnessResultBuilder(kernel_path="k", reference_path="r")
    builder.add_stage("smoke", passed=False, message="fail")
    assert should_run_benchmark(builder) is False


def test_benchmark_allowed_when_all_pass():
    builder = HarnessResultBuilder(kernel_path="k", reference_path="r")
    builder.add_stage("smoke", passed=True)
    assert should_run_benchmark(builder) is True
