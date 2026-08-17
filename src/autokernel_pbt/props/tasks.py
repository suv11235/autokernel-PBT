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
#:
#: MEASURED COST, which the paper must state rather than discover. Two of the nine
#: rungs are single-column — ``(1, 1)`` and ``(17, 1)`` — and softmax on a
#: one-column input is exactly 1.0 for *any* implementation. A whole class of
#: normalization bug is therefore not merely undetected on those rungs, it is
#: absent: the broken kernel is genuinely correct there, and both arms correctly
#: PASS. So roughly 22% of groups (2/9) score such a kernel as clean, and the
#: absolute detection rate this corpus reports is understated by that constant.
#:
#: They are kept anyway, and the choice is deliberate. They still catch row-index
#: and tail-handling bugs, which is what they were added for, and an asserted
#: blind spot is worth more than a hidden one — see
#: ``tests/integration/test_record_replay.py::test_unnormalized_softmax_is_caught_by_both_arms``,
#: which pins the blindness so it cannot be misread as a gap in an oracle. The
#: deflation applies to every arm equally, so the arm-vs-arm comparison the
#: project actually reports stays unbiased; only the absolute number moves.
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


#: Added to the row variance before the square root. 1e-5 is PyTorch's ``LayerNorm``
#: default, chosen so the reference matches the implementation a kernel author is most
#: likely to have been targeting.
LAYERNORM_EPS = 1e-5


def layernorm_reference(x: np.ndarray) -> np.ndarray:
    """Row-wise layer normalization over the last axis, without affine parameters.

    No learnable scale or shift. Those are a separate op, fused in practice, and
    including them would mean handing the declarative arm weights it plays no part in
    choosing; the normalization itself is what has interesting properties.

    Variance is the *population* variance (divide by n), matching PyTorch and every
    kernel implementation of it. The sample variance would put the reference a factor
    of sqrt(n/(n-1)) away from every correct kernel -- 8% at the ladder's (3, 7) rung,
    far above any tolerance -- so every kernel would be booked as a caught bug.

    Accumulation is in float64 and the result cast back, exactly as
    ``softmax_reference`` does: the reference is the trusted side and is allowed a
    wider intermediate, while the arms measure the residual against the dtype the
    kernel actually produced.

    ``eps`` goes *inside* the square root, which is where PyTorch puts it. Outside, a
    constant row gives 0/eps rather than 0/sqrt(eps) -- a different value, for a case
    the ladder reaches at two of its nine rungs.
    """
    wide = np.asarray(x, dtype=np.float64)
    mean = np.mean(wide, axis=-1, keepdims=True)
    centered = wide - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return (centered / np.sqrt(variance + LAYERNORM_EPS)).astype(x.dtype)


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

#: Row-wise layer normalization. The normalization rung: it introduces mean-zero and
#: unit-variance, which are structural facts about the output that need no reference
#: implementation to check, and a division whose denominator can be driven to zero --
#: a second numerical-stability story independent of softmax's overflow one.
#:
#: ``shift_rows`` is carried because layernorm is *exactly* shift invariant in real
#: arithmetic: subtracting the row mean removes any per-row constant. Unlike softmax,
#: whose shift invariance breaks only when ``exp`` overflows and which therefore needs
#: overflow-scale shifts to be non-vacuous, this is a genuine algebraic law at any
#: scale. The relation's dtype-derived default scale is kept anyway, because a kernel
#: that computes its mean in reduced precision degrades exactly there.
#: MEASURED, and the reason this task does not share softmax's N(0, 1) inputs.
#: Under N(0, 1) a layernorm output and a *merely centered* output both have row
#: variance near 1 -- the sample variance of 8 standard normal draws ran 0.31 to 1.04
#: on the first ladder rung -- so ``rows_have_unit_variance`` barely discriminated the
#: exact defect it exists to catch. Widening the input makes an undivided row's
#: variance ~33, which the property separates from 1 decisively.
#:
#: It also fixes a false alarm. The deviation of a *correct* output's variance from 1
#: is not rounding, it is the ``eps`` inside the square root: ``var/(var + eps) - 1``
#: reproduced every observed value exactly. That term scales as ``eps/var``, so it
#: grows as the row variance shrinks -- 3.2e-5 at var=0.31, against a rounding budget
#: of 2.9e-5, which is a FAIL on a correct reference. At var~33 the same term is
#: ~3e-7, two orders below the budget. See
#: ``test_layernorm_properties_pass_the_real_reference``, which pins that margin so
#: narrowing this distribution fails loudly rather than silently reintroducing it.
LAYERNORM = Task(
    task_id="layernorm",
    domain=InputDomain(
        task_id="layernorm",
        tensors=(
            TensorSpec(
                name="x", dtype="float32", distribution="uniform", low=-10.0, high=10.0
            ),
        ),
        shapes=_LADDER_SHAPES,
        relations=(ShiftRows.name,),
    ),
)


#: Powers of two spanning log2(n) = 3..14, plus two non-power-of-two lengths so the
#: sweep is not blind to tail handling. Few rows per shape: the quantity under study
#: is the reduction length, and extra rows buy variance reduction at a linear cost in
#: hardware time.
_SWEEP_SHAPES: tuple[tuple[int, ...], ...] = (
    (4, 8),
    (4, 64),
    (4, 512),
    (4, 4096),
    (4, 16384),
    (4, 129),
    (4, 4095),
)

#: Not a new op -- softmax again, over a much wider range of reduction lengths.
#:
#: The ladder spans log2(n) from 0 to 7; the CPU measurements that chose log2(n) as
#: the reference arm's normalization swept to 16384, log2 = 14. Measuring the
#: tolerance on the ladder alone would cover half the dynamic range, concentrated at
#: the low end where the ratio is noisiest, so this task exists to make the
#: normalization question answerable on device at all.
#:
#: It reuses ``softmax_reference`` deliberately. A separate reference would make these
#: numbers incomparable with the softmax numbers already recorded on CPU, which is the
#: comparison the whole exercise rests on.
TOLERANCE_SWEEP = Task(
    task_id="tolerance_sweep",
    domain=InputDomain(
        task_id="tolerance_sweep",
        tensors=(TensorSpec(name="x", dtype="float32", distribution="normal"),),
        shapes=_SWEEP_SHAPES,
        relations=(),
    ),
)


#: Name -> task. Keyed off each task's own ``task_id`` so a key and the ids
#: stamped into its recorded rows cannot disagree.
TASKS: dict[str, Task] = {
    task.task_id: task for task in (RELU, SOFTMAX, LAYERNORM, TOLERANCE_SWEEP)
}

#: Reference implementation per task, for the reference arm. Kept separate from
#: ``Task`` because a reference is harness-side and not serializable: ``Task`` is
#: a description of what to run, and Task 13 will build one from a contract file
#: where a Python callable cannot appear.
REFERENCES = {
    RELU.task_id: relu_reference,
    SOFTMAX.task_id: softmax_reference,
    LAYERNORM.task_id: layernorm_reference,
    TOLERANCE_SWEEP.task_id: softmax_reference,
}
