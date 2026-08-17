"""Phase B: the execution boundary.

This is the only component that ever touches real hardware. Phase 3 puts
CUDA / Triton / NKI behind this same `Backend` protocol, so everything
upstream of it stays unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np

from autokernel_pbt.props.case import Case
from autokernel_pbt.props.spec import CaseSpec

# Tensors whose names start with this prefix are generator bookkeeping (e.g. a
# recorded permutation) and are never passed to the kernel.
HELPER_PREFIX = "__"

# The name the single kernel output is persisted under. Properties read
# `row.outputs[OUTPUT_NAME]`.
OUTPUT_NAME = "y"

# The name of the primary input tensor. `relations.py` derives its partners from it
# and `Case.dtype`/`Case.shape` describe it, so a property that needs to look at what
# the kernel was *given* — rather than only at what it returned — reads this key.
PRIMARY_INPUT = "x"

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
    its wire value and compares equal to it, while a typo is an `AttributeError`
    at import rather than a comparison that silently never matches.
    """

    OK = "ok"
    LAUNCH_ERROR = "launch_error"
    OUTPUT_ERROR = "output_error"
    # Reserved for later phases; defined here so there is one vocabulary.
    COMPILE_ERROR = "compile_error"
    TIMEOUT = "timeout"

    # Without these, `str()` and `format()` of a str-mixin Enum return the wire
    # value on 3.10/3.11 but the repr-style name ("Status.OK") on 3.12+ — so
    # `f"status={row.status}"` would render differently across the range this
    # project declares support for (requires-python = ">=3.10"). Task 9 puts
    # exactly that interpolation in a human-readable detail field.
    # `enum.StrEnum` does this natively but is 3.11+, so it is unavailable here.
    __str__ = str.__str__
    __format__ = str.__format__


@dataclass
class ExecutionResult:
    """One kernel execution. Persisted verbatim as an execution-table row."""

    case: Case
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    status: Status = Status.OK
    error: str = ""
    # Ground truth about the kernel that produced this row. kernel_is_broken is
    # tri-state on purpose: None means "not stated", False means "stated correct".
    # Collapsing them would silently enlarge the correct-kernel denominator of the
    # false-positive rate.
    kernel_id: str = ""
    kernel_is_broken: bool | None = None
    # The recipe that regenerated this row's case group, denormalized onto every row
    # of the group exactly as `corpus_fingerprint` is, and for the same reason:
    # Parquet dictionary-encodes a repeated string column, so a side table would cost
    # a join to save nothing. Without it a recorded run carries no way to rebuild a
    # group offline, and a later shrinker would have to guess (seed, index, shape,
    # transforms) back out of the domain -- the retrofit `spec.py` exists to prevent.
    case_spec: CaseSpec | None = None


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
    # A 0-d output (any reduction: `np.sum(x)`) comes back from persistence as
    # shape (1,), so normalize up front rather than letting the shape change
    # under us. Replay fairness is the guarantee at stake: an oracle must score
    # the same bytes in-process as it does from the table.
    #
    # The culprit is *not* safetensors, which round-trips shape [] unchanged;
    # it is `np.ascontiguousarray`, which Task 7's writer applies to every
    # tensor and which is documented as `ndmin=1` — it promotes 0-d to (1,).
    # Measured on numpy 2.5.2 / safetensors 0.8.0:
    #   save_file({"y": a})                       -> header shape [],  loads ()
    #   save_file({"y": np.ascontiguousarray(a)}) -> header shape [1], loads (1,)
    #
    # REQUIRED COUNTERPART (Task 9): `residual_ratio` must apply the same
    # `np.atleast_1d` to the *reference* side. It returns inf on any shape
    # mismatch, so normalizing only the candidate would make every reduction
    # case fail against its own 0-d reference.
    return np.atleast_1d(array)
