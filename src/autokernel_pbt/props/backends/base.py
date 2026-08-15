"""Phase B: the execution boundary.

This is the only component that ever touches real hardware. Phase 3 puts
CUDA / Triton / NKI behind this same `Backend` protocol, so everything
upstream of it stays unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from autokernel_pbt.props.case import Case

# Tensors whose names start with this prefix are generator bookkeeping (e.g. a
# recorded permutation) and are never passed to the kernel.
HELPER_PREFIX = "__"

# The name the single kernel output is persisted under. Properties read
# `row.outputs[OUTPUT_NAME]`.
OUTPUT_NAME = "y"

# Status vocabulary. Named constants rather than bare literals so a typo is a
# NameError at import time instead of a comparison that silently never matches.
# The wire values are what lands in the execution table; do not change them.
STATUS_OK = "ok"
STATUS_LAUNCH_ERROR = "launch_error"
# Reserved for later phases; defined here so there is one vocabulary.
STATUS_COMPILE_ERROR = "compile_error"
STATUS_TIMEOUT = "timeout"
# The kernel ran but returned something this phase cannot represent (see
# `single_output` below).
STATUS_OUTPUT_ERROR = "output_error"

STATUSES = frozenset(
    {
        STATUS_OK,
        STATUS_LAUNCH_ERROR,
        STATUS_COMPILE_ERROR,
        STATUS_TIMEOUT,
        STATUS_OUTPUT_ERROR,
    }
)


@dataclass
class ExecutionResult:
    """One kernel execution. Persisted verbatim as an execution-table row."""

    case: Case
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    status: str = STATUS_OK
    error: str = ""


class Backend(Protocol):
    name: str

    def run(self, kernel: Callable[..., np.ndarray], case: Case) -> ExecutionResult: ...


def kernel_inputs(case: Case) -> dict[str, np.ndarray]:
    """Tensors the kernel actually receives, excluding generator bookkeeping.

    The returned arrays are the case's own objects, not copies: a defensive
    copy of every input would double peak memory on the hardware backends this
    boundary exists for. A kernel that writes to its input in place therefore
    corrupts the case. That is accepted for Phase 1 — the mitigation is to
    persist inputs before execution, not to copy here.
    """
    return {k: v for k, v in case.tensors.items() if not k.startswith(HELPER_PREFIX)}


class OutputContractError(TypeError):
    """A kernel returned something Phase 1 cannot represent."""


def single_output(value: Any) -> np.ndarray:
    """Coerce a kernel return value to the one output tensor Phase 1 supports.

    Phase 1 is deliberately single-output: the corpus tasks are elementwise and
    reduction kernels, and every property reads `outputs["y"]`. Multi-output
    kernels are *rejected*, not accommodated, because the alternative is worse
    than an error — `np.asarray((a, b))` silently stacks two same-shaped arrays
    into one array of higher rank, and `np.asarray({...})` produces a 0-d object
    array. Both would flow into the oracle as plausible-looking data. Widening
    to named multi-output belongs in a later phase, where the persistence layer
    can carry the names.
    """
    if value is None:
        msg = "kernel returned None; a single array-like output is required"
        raise OutputContractError(msg)
    if isinstance(value, (tuple, list, dict, set)):
        msg = (
            f"kernel returned {type(value).__name__}; Phase 1 supports a single "
            f"array-like output only (multi-output kernels are not supported yet)"
        )
        raise OutputContractError(msg)
    array = np.asarray(value)
    if array.dtype == object:
        msg = (
            f"kernel returned {type(value).__name__}, which converts to an object "
            f"array; a single numeric array-like output is required"
        )
        raise OutputContractError(msg)
    return array
