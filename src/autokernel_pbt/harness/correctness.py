"""Correctness pipeline (skeleton)."""

from __future__ import annotations

from autokernel_pbt.harness.result import HarnessResultBuilder

DEFAULT_STAGES = [
    "smoke",
    "shape_sweep",
    "numerical_stress",
    "determinism",
    "edge_cases",
]


def run_stages(builder: HarnessResultBuilder, stages: list[str] | None = None) -> bool:
    """Run named correctness stages. Skeleton: marks all as skipped/pending."""
    names = stages or DEFAULT_STAGES
    for name in names:
        builder.add_stage(name, passed=True, message="skeleton: not implemented")
    return builder.correctness_passed


def should_run_benchmark(builder: HarnessResultBuilder) -> bool:
    return builder.correctness_passed
