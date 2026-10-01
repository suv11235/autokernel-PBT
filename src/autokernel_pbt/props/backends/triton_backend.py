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

import importlib
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
from autokernel_pbt.props.backends.telemetry import MISSING, extract
from autokernel_pbt.props.backends.triton_kernel import InputMutatedError, TritonKernel
from autokernel_pbt.props.case import Case


class TritonCompilationError(Exception):
    """Raised by a launcher that detects a compile failure itself.

    Its own type rather than a string match on the message: Triton's compile errors
    are not a stable, documented surface, and classifying by substring would silently
    reclassify on a version bump -- turning compile failures into launch failures in
    the artifacts, where nothing downstream could tell.

    No ladder or mutant launcher raises this. What `jit_fn[grid](...)` raises is
    Triton's own `CompilationError`, which the backend recognises by TYPE as well; see
    `_triton_compile_error_types`.
    """


#: Where Triton keeps its compile-error type, as (module, attribute): defined in
#: `triton.compiler.errors`, re-exported from `triton.compiler`. Probed, not imported
#: from one fixed path: its home is not a stable surface, and a release that moved it
#: must not silently turn every compile failure into a launch failure. Written without
#: a Triton to check against, like the telemetry probes; the device test
#: `test_a_kernel_that_fails_to_compile_is_a_compile_error_on_device` is what confirms
#: it on a given release.
_TRITON_COMPILE_ERROR_LOCATIONS = (
    ("triton.compiler.errors", "CompilationError"),
    ("triton.compiler", "CompilationError"),
)


def _triton_compile_error_types() -> tuple[type[BaseException], ...]:
    """Triton's compile-error types that this installation actually has.

    Looked up only when a kernel has already failed, so a machine without Triton pays
    nothing on the success path, and any import problem means "not found" rather than
    a second exception from inside an error handler.
    """
    found: list[type[BaseException]] = []
    for module_name, attribute in _TRITON_COMPILE_ERROR_LOCATIONS:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001, S112 - absent or broken: this location does not count
            continue
        candidate = getattr(module, attribute, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            found.append(candidate)
    return tuple(found)


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
            # Matched by Triton's own type, never by message. Checked here rather
            # than as its own `except` clause so the lookup -- an import -- runs only
            # on the failure path.
            compile_failed = isinstance(exc, _triton_compile_error_types())
            status = Status.COMPILE_ERROR if compile_failed else Status.LAUNCH_ERROR
            return self._failed(kernel, case, start, status, exc)

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
            device=self._probe_device(),
            launch=kernel.launch_telemetry(),
            flags=flags,
        )
        payload[TELEMETRY_BACKEND] = self.name
        payload[TELEMETRY_WALL_MS] = (time.perf_counter() - start) * 1000.0
        payload["kernel_source_hash"] = kernel.source_hash
        return payload

    def _probe_device(self) -> dict[str, Any]:
        """The device group, or an empty one -- every key MISSING -- if the probe fails.

        The probe runs on the failure path too. After a sticky CUDA error (an illegal
        address from a broken kernel) every `torch.cuda.*` call raises, the probe's
        included, so an unguarded probe re-raised from inside the error handler and
        escaped `run` on exactly the row that most needed recording.
        """
        try:
            return dict(self.device_probe())
        except Exception:  # noqa: BLE001 - probed, not asserted
            return {}

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
    except ImportError:
        triton_version = MISSING
    return {
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multi_processor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "driver_version": _driver_version(torch),
        # The CUDA toolkit torch was BUILT against, which is not the driver.
        "runtime_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_version": triton_version,
    }


def _driver_version(torch: Any) -> Any:
    """The CUDA version the installed DRIVER supports, as "major.minor", or MISSING.

    `torch.version.cuda` was recorded here before, which is the toolkit torch was built
    with -- a property of the wheel, not of the machine -- so driver and runtime always
    read identically and a driver mismatch was invisible in every artifact.

    `cuDriverGetVersion`, which `_cuda_getDriverVersion` wraps, encodes 1000*major +
    10*minor: 12090 is a 12.9 driver. It is formatted like `runtime_version` ("12.8")
    so the two compare directly. A private torch API, so probed: absent, raising, zero
    (no driver) or not an int all record MISSING rather than a guess.
    """
    try:
        encoded = torch._C._cuda_getDriverVersion()
    except Exception:  # noqa: BLE001 - probed, not asserted
        return MISSING
    if isinstance(encoded, bool) or not isinstance(encoded, int) or encoded <= 0:
        return MISSING
    return f"{encoded // 1000}.{encoded % 1000 // 10}"
