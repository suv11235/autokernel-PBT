"""CPU backend. First-class, not a stub: CI runs entirely on it."""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

import numpy as np

from autokernel_pbt.props.backends.base import (
    OUTPUT_NAME,
    STATUS_LAUNCH_ERROR,
    STATUS_OK,
    STATUS_OUTPUT_ERROR,
    ExecutionResult,
    OutputContractError,
    kernel_inputs,
    single_output,
)
from autokernel_pbt.props.case import Case


class NumpyBackend:
    name = "numpy"

    def run(self, kernel: Callable[..., np.ndarray], case: Case) -> ExecutionResult:
        inputs = kernel_inputs(case)
        start = time.perf_counter()
        try:
            output = kernel(**inputs)
        except Exception as exc:  # noqa: BLE001 - a failing kernel is data, not an error
            # Time spent before the failure is still real signal (a kernel that
            # fails after 30s is a different problem from one that fails
            # immediately), so measure it rather than reporting a flat 0.0.
            return self._failed(case, start, STATUS_LAUNCH_ERROR, exc)

        try:
            array = single_output(output)
        except OutputContractError as exc:
            return self._failed(case, start, STATUS_OUTPUT_ERROR, exc)

        return ExecutionResult(
            case=case,
            outputs={OUTPUT_NAME: array},
            telemetry=self._telemetry(start),
            status=STATUS_OK,
        )

    def _telemetry(self, start: float) -> dict[str, Any]:
        return {"backend": self.name, "wall_ms": (time.perf_counter() - start) * 1000.0}

    def _failed(
        self, case: Case, start: float, status: str, exc: BaseException
    ) -> ExecutionResult:
        return ExecutionResult(
            case=case,
            telemetry=self._telemetry(start),
            status=status,
            error=f"{exc}\n{traceback.format_exc()}",
        )
