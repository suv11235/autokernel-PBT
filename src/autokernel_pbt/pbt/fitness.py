"""Fitness from harness results."""

from __future__ import annotations

from typing import Any

FAILED_FITNESS = float("-inf")


def fitness_from_harness_result(result: dict[str, Any]) -> float:
    if not result.get("passed"):
        return FAILED_FITNESS
    bench = result.get("benchmark") or {}
    if not bench.get("ran"):
        return FAILED_FITNESS
    speedup = bench.get("speedup_vs_eager")
    if speedup is None:
        return FAILED_FITNESS
    return float(speedup)
