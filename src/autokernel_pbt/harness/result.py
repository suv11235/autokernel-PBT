"""Harness result builder (skeleton)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    name: str
    passed: bool
    message: str = ""


@dataclass
class HarnessResultBuilder:
    kernel_path: str
    reference_path: str
    run_id: str = ""
    candidate_id: str = ""
    stages: list[StageResult] = field(default_factory=list)
    benchmark_ran: bool = False
    speedup_vs_eager: float | None = None
    kernel_ms: float | None = None
    baseline_ms: float | None = None

    def add_stage(self, name: str, passed: bool, message: str = "") -> None:
        self.stages.append(StageResult(name=name, passed=passed, message=message))

    @property
    def correctness_passed(self) -> bool:
        return bool(self.stages) and all(s.passed for s in self.stages)

    def to_dict(self) -> dict[str, Any]:
        passed = self.correctness_passed
        benchmark: dict[str, Any] = {"ran": False}
        if passed and self.benchmark_ran:
            benchmark = {
                "ran": True,
                "speedup_vs_eager": self.speedup_vs_eager,
                "kernel_ms": self.kernel_ms,
                "baseline_ms": self.baseline_ms,
            }
        return {
            "version": "1",
            "passed": passed and (not self.benchmark_ran or (self.speedup_vs_eager or 0) > 0),
            "correctness": {
                "passed": self.correctness_passed,
                "stages": [
                    {"name": s.name, "passed": s.passed, "message": s.message} for s in self.stages
                ],
            },
            "benchmark": benchmark,
            "meta": {
                "kernel_path": self.kernel_path,
                "reference_path": self.reference_path,
                "run_id": self.run_id,
                "candidate_id": self.candidate_id,
            },
        }
