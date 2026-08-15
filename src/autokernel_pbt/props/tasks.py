"""The development ladder: the kernels Phase 1 records, and their input domains.

A *task* is a name, an input domain, and (here, for the harness's own use) a
trusted reference implementation. It is deliberately not a kernel: the kernel is
whatever a code-generating model produced and is supplied at execution time,
which is the whole point of recording executions rather than assertions.

The reference implementations live here rather than beside the kernels because
they belong to the *oracle*, not to the corpus. ``ReferenceOracle`` calls them as
``reference_fn(**kernel_inputs(case))``, so their parameter names are part of the
contract: they must match the kernel's, and they never see generator bookkeeping
tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autokernel_pbt.props.domain import InputDomain, TensorSpec
from autokernel_pbt.props.relations import ShiftRows

#: The ladder shapes, in the order ``Generator`` cycles through them.
#:
#: Every entry is 2-D, without exception: ``ShiftRows`` rejects anything else,
#: because a 1-D ``x`` is ambiguous (one row of n, or n rows of one?) and an
#: ``(n, 1)`` shift silently outer-broadcasts it to ``(n, n)``. Keeping the relu
#: task on the same 2-D ladder even though it declares no relations costs nothing
#: and means adding a relation to it later is a one-line change rather than a
#: reshape of the whole corpus.
#:
#: The three flavours are chosen for what they can break, not for coverage of a
#: size range:
#:   * powers of two — the sizes a vectorized or tiled kernel is written for, and
#:     the only ones where every tile is full;
#:   * odd remainders — primes and near-primes, so the last tile is partial and
#:     any masking or tail-handling bug is reachable. This is where hand-tuned
#:     kernels actually fail;
#:   * single rows and single columns — the degenerate ends. A one-column input
#:     makes softmax's output identically 1.0, which is the case where a kernel
#:     that ignores its input entirely still looks correct, and a one-row input
#:     removes any chance a row-index bug cancels out.
_LADDER_SHAPES: tuple[tuple[int, ...], ...] = (
    (8, 8),
    (4, 16),
    (16, 32),
    (3, 7),
    (5, 33),
    (7, 129),
    (1, 1),
    (1, 64),
    (17, 1),
)


def relu_reference(x: np.ndarray) -> np.ndarray:
    """max(x, 0), elementwise, in the input's own dtype.

    The zero is built as a 0-d array of ``x.dtype`` rather than a Python ``0``.
    NEP 50 promotes a Python scalar to the array's dtype, so ``np.maximum(x, 0)``
    happens to be right today for float32 — but the same expression against a
    float16 input under the older promotion rules returned float32, and a
    reference whose *dtype* depends on the numpy version is a reference that
    silently changes the shape-and-dtype comparison ``ReferenceOracle`` performs.
    """
    return np.maximum(x, np.zeros((), dtype=x.dtype))


def softmax_reference(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis, with max-subtraction.

    Max-subtraction is not a stylistic choice here, it is a correctness
    requirement of the harness. ``ShiftRows`` draws its shifts at half the
    exponent where ``exp`` overflows in the tensor's dtype, precisely so that a
    kernel *without* max-subtraction is caught; a reference without it would
    overflow on the very same rows. ``ReferenceOracle`` wraps the call in
    ``np.errstate(all="ignore")`` and would then book those rows INCONCLUSIVE
    with "the reference, not the kernel" — silently deleting from the reference
    arm exactly the rows where the metamorphic arm scores, and biasing the
    comparison the project reports.

    Accumulation is in float64 and the result is cast back to the input dtype.
    The reference is allowed a wider intermediate — it is the trusted side — and
    ``ReferenceOracle`` measures the residual against the *kernel's* dtype, so the
    rounding budget stays the one the kernel actually had.
    """
    wide = np.asarray(x, dtype=np.float64)
    shifted = wide - np.max(wide, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(x.dtype)


@dataclass(frozen=True)
class Task:
    """One rung of the development ladder.

    Only ``task_id`` and ``domain`` are structural. ``domain.task_id`` must agree
    with ``task_id``, because the generator stamps case ids from the domain while
    every other lookup is keyed by the task — a disagreement would produce a
    recorded table whose rows cannot be joined back to the task that made them.
    """

    task_id: str
    domain: InputDomain

    def __post_init__(self) -> None:
        if self.domain.task_id != self.task_id:
            msg = (
                f"task {self.task_id!r} carries a domain for {self.domain.task_id!r}; "
                f"recorded case ids would name the domain and never join back"
            )
            raise ValueError(msg)


#: Elementwise, unbounded above, and with no metamorphic partner. Its value in the
#: ladder is that it is the simplest thing a backend can get wrong: a kernel that
#: mishandles the tail of a partial tile fails here with nothing else to blame.
#: ``relations = ()`` is a claim, not an omission — no relation in ``RELATIONS``
#: constrains relu in a way softmax's do not already cover, and declaring one that
#: no property consumes would produce partner cases nobody scores.
RELU = Task(
    task_id="relu",
    domain=InputDomain(
        task_id="relu",
        tensors=(TensorSpec(name="x", dtype="float32", distribution="normal"),),
        shapes=_LADDER_SHAPES,
        relations=(),
    ),
)

#: Row-wise softmax. Carries ``shift_rows`` because shift-invariance is the one
#: law that separates a numerically stable implementation from a naive one, and
#: it is checkable with no reference implementation at all — which is the
#: tolerance-free detection the project's headline claim is about.
SOFTMAX = Task(
    task_id="softmax",
    domain=InputDomain(
        task_id="softmax",
        tensors=(TensorSpec(name="x", dtype="float32", distribution="normal"),),
        shapes=_LADDER_SHAPES,
        relations=(ShiftRows.name,),
    ),
)

#: Name -> task. Keyed off each task's own ``task_id`` so a key and the ids
#: stamped into its recorded rows cannot disagree.
TASKS: dict[str, Task] = {task.task_id: task for task in (RELU, SOFTMAX)}

#: Reference implementation per task, for the reference arm. Kept separate from
#: ``Task`` because a reference is harness-side and not serializable: ``Task`` is
#: a description of what to run, and Task 13 will build one from a contract file
#: where a Python callable cannot appear.
REFERENCES = {
    RELU.task_id: relu_reference,
    SOFTMAX.task_id: softmax_reference,
}
