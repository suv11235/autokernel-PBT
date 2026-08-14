"""Harness orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from autokernel_pbt.harness.benchmark import run_benchmark
from autokernel_pbt.harness.correctness import run_stages, should_run_benchmark
from autokernel_pbt.harness.result import HarnessResultBuilder
from autokernel_pbt.schema import validate


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def run_harness(
    kernel_path: str,
    reference_path: str,
    config: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    cfg = config or {}
    builder = HarnessResultBuilder(kernel_path=kernel_path, reference_path=reference_path)
    stages = (cfg.get("correctness") or {}).get("stages")
    run_stages(builder, stages=stages)
    if should_run_benchmark(builder):
        if not dry_run:
            run_benchmark(builder)
        else:
            builder.benchmark_ran = True
            builder.speedup_vs_eager = 1.0
            builder.kernel_ms = 0.0
            builder.baseline_ms = 0.0
    result = builder.to_dict()
    validate(result, "harness_result.schema.json")
    return result
