"""Device backend for Triton kernels.

Mirrors `NumpyBackend`'s shape deliberately: same protocol, same failure discipline, a
richer telemetry payload. A kernel that fails is *data*, never an exception that
escapes -- an escaping exception aborts a scoring pass over executions that have
already cost rented hardware time.

Three things differ from the CPU backend, and each is a device reality rather than an
implementation choice. See the design doc section 6.

*Compilation is lazy.* Triton compiles on first call, so a compile error arrives
during execution. `Status.COMPILE_ERROR` has existed unused since phase 1 for exactly
this. The distinction matters because a kernel that never compiled says nothing about
numerics, while one that launched and produced garbage says a great deal -- and both
must still be INCONCLUSIVE in every arm.

*The read-only-inputs guarantee is rebuilt elsewhere.* `readonly_inputs` protects the
host array, which a device kernel never touches. This backend cannot check it either,
for the same reason -- it holds only host arrays. The check lives in the launcher,
which owns the device buffers; see `InputMutatedError`. What this backend owes is
classifying that error rather than letting it escape.

*Execution is not bitwise reproducible.* Atomics and reduction order mean re-running
need not reproduce the recorded output. Nothing here depends on it -- the recorded
execution is the one the arms score -- but it does mean "re-run and compare" is not
available as a check, on device or in a test.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from autokernel_pbt.props.backends.base import (
    OUTPUT_NAME,
    TELEMETRY_BACKEND,
    TELEMETRY_WALL_MS,
    ExecutionResult,
    OutputContractError,
    Status,
    kernel_inputs,
)
from autokernel_pbt.props.backends.telemetry import extract
from autokernel_pbt.props.backends.triton_kernel import InputMutatedError, TritonKernel
from autokernel_pbt.props.case import Case


class TritonCompilationError(Exception):
    """Raised by a launcher when Triton fails to compile the kernel.

    Its own type rather than a string match on the message: Triton's compile errors
    are not a stable, documented surface, and classifying by substring would silently
    reclassify on a version bump -- turning compile failures into launch failures in
    the artifacts, where nothing downstream could tell.
    """


class TritonBackend:
    """Executes `TritonKernel` adapters and records device telemetry."""

    name = "triton"

    def __init__(self, device_probe: Any = None) -> None:
        # Injected so the backend is exercisable with no CUDA present. The default
        # imports torch lazily inside itself, not at module scope, so this module
        # imports cleanly on a machine that has none.
        self.device_probe = device_probe or _default_device_probe

    def run(self, kernel: Any, case: Case) -> ExecutionResult:
        if not isinstance(kernel, TritonKernel):
            # A bad *call*, not bad data: it can only come from a coding error, costs
            # nothing to re-run, and a TypeError from deep inside a launch would name
            # neither the backend nor the kernel.
            msg = (
                f"{type(self).__name__} requires a TritonKernel adapter, got "
                f"{type(kernel).__name__}; wrap the jit function first"
            )
            raise TypeError(msg)

        inputs = kernel_inputs(case)
        start = time.perf_counter()
        try:
            output = kernel(**inputs)
        except TritonCompilationError as exc:
            return self._failed(kernel, case, start, Status.COMPILE_ERROR, exc)
        except InputMutatedError as exc:
            # Raised by the launcher, the only layer holding the device buffers.
            #
            # This handler is NOT redundant with the generic one below, even though
            # both yield LAUNCH_ERROR: it sets `input_mutated` in the telemetry. A
            # kernel that wrote to its own input and one that merely crashed are
            # different fault classes, and without the flag the distinction is
            # unrecoverable from the artifacts -- InputMutatedError subclasses
            # RuntimeError, so the generic path would swallow it indistinguishably.
            return self._failed(
                kernel, case, start, Status.LAUNCH_ERROR, exc, flags={"input_mutated": True}
            )
        except OutputContractError as exc:
            return self._failed(kernel, case, start, Status.OUTPUT_ERROR, exc)
        except Exception as exc:  # noqa: BLE001 - a failing kernel is data, not an error
            return self._failed(kernel, case, start, Status.LAUNCH_ERROR, exc)

        return ExecutionResult(
            case=case,
            outputs={OUTPUT_NAME: output},
            telemetry=self._telemetry(kernel, start),
            status=Status.OK,
        )

    def _telemetry(
        self, kernel: TritonKernel, start: float, flags: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = extract(
            kernel.compiled,
            device=self.device_probe(),
            launch=kernel.launch_telemetry(),
            flags=flags,
        )
        payload[TELEMETRY_BACKEND] = self.name
        payload[TELEMETRY_WALL_MS] = (time.perf_counter() - start) * 1000.0
        payload["kernel_source_hash"] = kernel.source_hash
        return payload

    def _failed(
        self,
        kernel: TritonKernel,
        case: Case,
        start: float,
        status: Status,
        exc: BaseException,
        flags: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        # Time before the failure is still signal: a kernel that dies after 30s is a
        # different problem from one that dies immediately.
        return ExecutionResult(
            case=case,
            telemetry=self._telemetry(kernel, start, flags),
            status=status,
            error="".join(traceback.format_exception(exc)),
        )


def _default_device_probe() -> dict[str, Any]:
    """Device and toolchain facts, read once per execution.

    Imports inside the function: this module must import cleanly on a machine with no
    torch, so that every structural test runs on CPU.
    """
    import torch

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:  # pragma: no cover - triton present wherever this runs
        triton_version = None
    return {
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multi_processor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "driver_version": torch.version.cuda,
        "runtime_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_version": triton_version,
    }
