"""CPU backend. First-class, not a stub: CI runs entirely on it."""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

import numpy as np

from autokernel_pbt.props.backends.base import (
    OUTPUT_NAME,
    TELEMETRY_BACKEND,
    TELEMETRY_WALL_MS,
    ExecutionResult,
    OutputContractError,
    Status,
    kernel_inputs,
    readonly_inputs,
    single_output,
)
from autokernel_pbt.props.case import Case


class NumpyBackend:
    name = "numpy"

    def run(self, kernel: Callable[..., np.ndarray], case: Case) -> ExecutionResult:
        inputs = kernel_inputs(case)
        start = time.perf_counter()
        try:
            # Inputs are aliased, not copied; read-only makes an in-place write
            # a loud failure instead of silent corruption of the case.
            with readonly_inputs(inputs):
                output = kernel(**inputs)
        except Exception as exc:  # noqa: BLE001 - a failing kernel is data, not an error
            # Time spent before the failure is still real signal (a kernel that
            # fails after 30s is a different problem from one that fails
            # immediately), so measure it rather than reporting a flat 0.0.
            return self._failed(case, start, Status.LAUNCH_ERROR, exc)

        try:
            array = single_output(output)
        except OutputContractError as exc:
            return self._failed(case, start, Status.OUTPUT_ERROR, exc)

        return ExecutionResult(
            case=case,
            outputs={OUTPUT_NAME: array},
            telemetry=self._telemetry(start),
            status=Status.OK,
        )

    def _telemetry(self, start: float) -> dict[str, Any]:
        return {
            TELEMETRY_BACKEND: self.name,
            TELEMETRY_WALL_MS: (time.perf_counter() - start) * 1000.0,
        }

    def _failed(
        self, case: Case, start: float, status: Status, exc: BaseException
    ) -> ExecutionResult:
        return ExecutionResult(
            case=case,
            telemetry=self._telemetry(start),
            status=status,
            error="".join(traceback.format_exception(exc)),
        )
