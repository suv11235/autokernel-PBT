"""Benchmark timing (skeleton)."""

from __future__ import annotations

from autokernel_pbt.harness.result import HarnessResultBuilder


def run_benchmark(builder: HarnessResultBuilder) -> None:
    """Record placeholder benchmark metrics."""
    builder.benchmark_ran = True
    builder.kernel_ms = 1.0
    builder.baseline_ms = 1.0
    builder.speedup_vs_eager = 1.0
