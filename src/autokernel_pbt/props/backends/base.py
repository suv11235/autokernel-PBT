"""Phase B: the execution boundary.

This is the only component that ever touches real hardware. Phase 3 puts
CUDA / Triton / NKI behind this same `Backend` protocol, so everything
upstream of it stays unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np

from autokernel_pbt.props.case import Case

# Tensors whose names start with this prefix are generator bookkeeping (e.g. a
# recorded permutation) and are never passed to the kernel.
HELPER_PREFIX = "__"

# The name the single kernel output is persisted under. Properties read
# `row.outputs[OUTPUT_NAME]`.
OUTPUT_NAME = "y"

# Telemetry keys every backend must populate. Telemetry cannot be backfilled —
# re-running a Trainium job to recover a missing counter is the exact cost this
# architecture exists to avoid — so a Phase 3 backend writing "wall_time_ms"
# would yield properties that read None forever with no error. The open-ended
# tier-2 fields stay free-form; the universal core does not.
TELEMETRY_BACKEND = "backend"
TELEMETRY_WALL_MS = "wall_ms"

# NumPy dtype kinds that survive the round trip: bool, signed/unsigned int,
# float. This is the intersection of what safetensors can persist and what the
# oracles can numerically compare. Anything else (str, bytes, complex, void,
# object) must be rejected here, as data, rather than reaching the persistence
# layer — Task 7 writes the whole batch in one loop, so a single kernel
# returning a string would abort the entire run's write.
PERSISTABLE_KINDS = "biuf"


class Status(str, Enum):
    """Execution outcome. A `str` subclass, so it lands in Parquet and JSON as
    its wire value and compares equal to it, while a typo is a `AttributeError`
    at import rather than a comparison that silently never matches.
    """

    OK = "ok"
    LAUNCH_ERROR = "launch_error"
    OUTPUT_ERROR = "output_error"
    # Reserved for later phases; defined here so there is one vocabulary.
    COMPILE_ERROR = "compile_error"
    TIMEOUT = "timeout"


@dataclass
class ExecutionResult:
    """One kernel execution. Persisted verbatim as an execution-table row."""

    case: Case
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    status: str = Status.OK
    error: str = ""


class Backend(Protocol):
    name: str

    def run(self, kernel: Callable[..., np.ndarray], case: Case) -> ExecutionResult: ...


def kernel_inputs(case: Case) -> dict[str, np.ndarray]:
    """Tensors the kernel actually receives, excluding generator bookkeeping.

    The returned arrays are the case's own objects, not copies: a defensive
    copy of every input would double peak memory on the hardware backends this
    boundary exists for. Backends must run the kernel inside `readonly_inputs`
    so that aliasing cannot silently corrupt the case.
    """
    return {k: v for k, v in case.tensors.items() if not k.startswith(HELPER_PREFIX)}


@contextmanager
def readonly_inputs(inputs: dict[str, np.ndarray]) -> Iterator[dict[str, np.ndarray]]:
    """Make the kernel's inputs temporarily read-only.

    Inputs are aliased, not copied, so a kernel writing in place would corrupt
    the case — and that corruption is invisible downstream: an oracle
    recomputing the reference from a corrupted `x` and comparing it against an
    output that *is* that corrupted `x` can silently agree. Flipping the
    writeable flag turns silent corruption into a loud, correctly classified
    `launch_error`, at no memory cost.

    Two known, accepted caveats:
      * An identity-like kernel returns an array that inherits the read-only
        flag. safetensors persists such an array fine.
      * Phase 3's `torch.from_numpy` warns on a read-only array. Restoring in
        `finally` keeps that scoped to the call itself.
    """
    saved = {k: v.flags.writeable for k, v in inputs.items()}
    for value in inputs.values():
        value.flags.writeable = False
    try:
        yield inputs
    finally:
        for key, value in inputs.items():
            value.flags.writeable = saved[key]


class OutputContractError(TypeError):
    """A kernel returned something Phase 1 cannot represent or persist."""


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

    The dtype gate is equally load-bearing: everything that reaches here gets
    handed to `safetensors.save_file` by Task 7, in one loop over the whole
    batch. A kernel returning a string or a complex array would raise there and
    take the entire run's persistence with it, so it is caught here as data.
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
    if array.dtype.kind not in PERSISTABLE_KINDS:
        msg = (
            f"kernel returned dtype {array.dtype!r} (kind {array.dtype.kind!r}), "
            f"which cannot be persisted or compared; a bool/int/float array-like "
            f"output is required"
        )
        raise OutputContractError(msg)
    # 0-d output is deliberately NOT normalized to (1,). safetensors 0.8.0
    # round-trips shape [] unchanged through every read path (load_file,
    # safe_open.get_tensor, load), so there is nothing to repair — and
    # normalizing would actively break reductions: `residual_ratio` returns inf
    # on any shape mismatch, so a (1,) candidate against a 0-d reference
    # (`np.sum(x)`) would fail every reduction case. Pinned by
    # `test_zero_dim_output_round_trips_at_a_stable_shape`.
    return array
