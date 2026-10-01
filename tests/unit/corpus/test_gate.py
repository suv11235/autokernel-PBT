"""Admission-gate tests.

An agent-authored mutant cannot be taken at its word, and each way it can be wrong
corrupts a different number invisibly. These pin the two admission criteria and the
recording of rejections.
"""

from __future__ import annotations

import numpy as np

from autokernel_pbt.corpus.gate import Rejection, admit
from autokernel_pbt.props.backends.base import (
    OUTPUT_NAME,
    ExecutionResult,
    Status,
    kernel_inputs,
)
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.oracle import ReferenceOracle
from autokernel_pbt.props.tasks import RELU, SOFTMAX, relu_reference, softmax_reference
from autokernel_pbt.props.verdict import Verdict

#: The published rejection reasons, written out rather than imported. They are tallied
#: as a finding, so a reworded reason is a change to that finding and must fail here.
NOT_BROKEN = "not broken on any group: it matches the reference within tolerance"
EXACT_DTYPE = "not verifiable on any group: the output dtype is exact, so no test ratio is defined for it"
NONFINITE_REFERENCE = (
    "not verifiable on any group: the reference output is non-finite, a defect of the "
    "reference and not of the candidate"
)


def _case(cid: str, gid: str) -> Case:
    return Case(
        case_id=cid, group_id=gid, relation="base", task_id="t",
        dtype="float32", shape=(2, 3),
        tensors={"x": np.ones((2, 3), dtype=np.float32)},
    )


def _row(cid, gid, y, status=Status.OK) -> ExecutionResult:
    outputs = {} if y is None else {OUTPUT_NAME: np.asarray(y, dtype=np.float32)}
    return ExecutionResult(case=_case(cid, gid), outputs=outputs, status=status)


def _ref(**kw):
    return np.ones((2, 3), dtype=np.float32)


def test_a_candidate_that_is_not_broken_is_rejected():
    """The criterion A_CORRECT_MUTANT_IS_REJECTED.

    A mutant that is secretly correct enters the detection denominator as a bug
    nobody can catch. Every arm's rate drops for free and the corpus looks harder
    than it is -- the most dangerous of the three failure modes, because nothing
    downstream looks wrong.
    """
    rows = [_row("c0", "g0", np.ones((2, 3))), _row("c1", "g1", np.ones((2, 3)))]
    verdict = admit(rows, reference_fn=_ref)
    assert isinstance(verdict, Rejection)
    assert "not broken" in verdict.reason


def test_a_candidate_that_never_runs_is_rejected():
    """The criterion A_CATASTROPHIC_MUTANT_IS_REJECTED."""
    rows = [
        _row("c0", "g0", None, status=Status.LAUNCH_ERROR),
        _row("c1", "g1", None, status=Status.COMPILE_ERROR),
    ]
    verdict = admit(rows, reference_fn=_ref)
    assert isinstance(verdict, Rejection)
    assert "judgeable" in verdict.reason


def test_a_candidate_wrong_on_every_group_is_admitted():
    """The criterion A_KERNEL_WRONG_EVERYWHERE_IS_ADMITTED.

    An earlier draft also demanded agreement somewhere, on the theory that a kernel
    wrong everywhere is suspicious. A kernel wrong on every group is an ordinary bug
    that should score a detection rate of 1.0; that criterion would have rejected
    valid mutants for being too easy to catch.
    """
    rows = [_row("c0", "g0", np.full((2, 3), 9.0)), _row("c1", "g1", np.full((2, 3), 9.0))]
    assert admit(rows, reference_fn=_ref) is True


def test_a_partially_broken_candidate_is_admitted():
    rows = [_row("c0", "g0", np.ones((2, 3))), _row("c1", "g1", np.full((2, 3), 9.0))]
    assert admit(rows, reference_fn=_ref) is True


def test_a_rejection_records_its_reason():
    """The criterion REJECTIONS_CARRY_A_REASON.

    The rejection rate is a finding in its own right: it says what proportion of an
    agent's attempts at a named fault class are not that fault. Dropping refused
    candidates silently would discard it.
    """
    rows = [_row("c0", "g0", np.ones((2, 3)))]
    verdict = admit(rows, reference_fn=_ref)
    assert isinstance(verdict, Rejection)
    assert verdict.reason
    assert verdict.groups_broken == 0
    assert verdict.groups_judgeable == 1


def test_a_mixed_candidate_that_crashes_somewhere_is_still_admitted():
    # Crashing on some cases is normal for a real bug; only crashing EVERYWHERE is
    # disqualifying, because that is what makes every arm abstain.
    rows = [
        _row("c0", "g0", None, status=Status.LAUNCH_ERROR),
        _row("c1", "g1", np.full((2, 3), 9.0)),
    ]
    assert admit(rows, reference_fn=_ref) is True


def test_a_shape_change_counts_as_broken():
    # A wrong output shape is a real and common kernel bug; it must not slip through
    # as "not broken" merely because a test ratio is undefined for it.
    rows = [_row("c0", "g0", np.ones((2, 5)))]
    assert admit(rows, reference_fn=_ref) is True


def _run(task, kernel) -> list[ExecutionResult]:
    """Every case of a full ladder, through the real backend."""
    groups = Generator(task.domain, seed=42).generate(len(task.domain.shapes))
    return [NumpyBackend().run(kernel, case) for group in groups for case in group.cases]


def test_a_correct_kernel_that_differs_in_the_last_bits_is_rejected():
    """The gate has a tolerance, and a correct kernel sits inside it.

    Multiplying by the reciprocal of the row sum instead of dividing is algebraically
    identical and differs from the float64 reference in the last bits on most rows.
    That is the realistic correct-but-different kernel the false-positive population
    is built from, and the gate must call it NOT broken. The GPU suite checks this on
    the Triton variants; without a CPU test, a gate that called every bitwise
    difference "broken" passed the default suite.
    """
    def reciprocal_softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return (e * (1.0 / np.sum(e, axis=-1, keepdims=True))).astype(x.dtype)

    rows = _run(SOFTMAX, reciprocal_softmax)
    bit_different = [
        r for r in rows
        if not np.array_equal(r.outputs[OUTPUT_NAME], softmax_reference(**kernel_inputs(r.case)))
    ]
    assert bit_different, "kernel is bit-identical to the reference; the test proves nothing"
    assert admit(rows, reference_fn=softmax_reference) == Rejection(
        reason=NOT_BROKEN, groups_broken=0, groups_judgeable=9, groups_unverifiable=0
    )


def test_an_exact_dtype_output_is_unverifiable_rather_than_unbroken():
    """An int output has no test ratio, so the gate never compared it.

    A data-type mutant that casts to int32 truncates every positive value, and the
    backend persists it faithfully. The gate once counted such a group as judgeable,
    skipped the comparison, and then rejected the candidate as "it matches the
    reference within tolerance" -- a statement about a comparison it never made, in a
    reason the project publishes as a finding.
    """
    def relu_int(x):
        return np.maximum(x, 0).astype(np.int32)

    assert admit(_run(RELU, relu_int), reference_fn=relu_reference) == Rejection(
        reason=EXACT_DTYPE, groups_broken=0, groups_judgeable=0, groups_unverifiable=9
    )


def test_a_wrong_shape_is_broken_whatever_its_dtype():
    # Shape is checked before the dtype, as the reference arm does: a wrong shape is
    # evidence without any test ratio, so an int output of the wrong shape is broken.
    case = _case("c0", "g0")
    row = ExecutionResult(
        case=case, outputs={OUTPUT_NAME: np.ones((2, 5), dtype=np.int32)}, status=Status.OK
    )
    assert admit([row], reference_fn=_ref) is True


def _rowsum_case() -> Case:
    x = np.ones((2, 1024), dtype=np.float32)
    return Case(
        case_id="rowsum-g00000-base", group_id="rowsum-g00000", relation="base",
        task_id="rowsum", dtype="float32", shape=x.shape, tensors={"x": x},
    )


def _rowsum_reference(x):
    return np.sum(np.asarray(x, dtype=np.float64), axis=-1).astype(np.float32)


def test_the_tolerance_is_normalized_by_the_inputs_reduction_length():
    """A (2, 1024) -> (2,) row sum is 1024 accumulations long, not 2.

    The output is exactly 100 ulps above the true sum 1024.0. Normalized by
    log2(1024) = 10 that is a ratio of 10, inside the threshold of 30, and the
    reference arm passes it; normalized by the OUTPUT's last axis, log2(2) floors to
    1 and the ratio is 100. The gate used the output's axis, so it admitted as
    "broken" a kernel the reference arm it is meant to agree with calls correct.
    """
    case = _rowsum_case()
    off = np.full(2, 1024.0 + 100 * 2.0**-13, dtype=np.float32)
    row = ExecutionResult(case=case, outputs={OUTPUT_NAME: off}, status=Status.OK)
    assert ReferenceOracle(_rowsum_reference).evaluate([row])[0].verdict is Verdict.PASS
    assert admit([row], reference_fn=_rowsum_reference) == Rejection(
        reason=NOT_BROKEN, groups_broken=0, groups_judgeable=1, groups_unverifiable=0
    )


def _overflowing_reference(x):
    """Softmax without max-subtraction: overflows to inf/inf = NaN on large rows."""
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(x.dtype)


def _softmax_row(gid: str, x: np.ndarray, y: np.ndarray) -> ExecutionResult:
    case = Case(
        case_id=f"{gid}-base", group_id=gid, relation="base", task_id="softmax",
        dtype="float32", shape=x.shape, tensors={"x": x},
    )
    return ExecutionResult(case=case, outputs={OUTPUT_NAME: y}, status=Status.OK)


def test_a_non_finite_reference_is_unverifiable_rather_than_broken():
    """NaN from the trusted side is a harness defect, never evidence about the candidate.

    The reference arm books it INCONCLUSIVE with "the reference, not the kernel". The
    gate fed it to ``residual_ratio``, got inf, and admitted a CORRECT kernel as
    broken -- the most dangerous admission there is, a bug nobody can catch entering
    the detection denominator.
    """
    big = np.full((2, 4), 100.0, dtype=np.float32)
    row = _softmax_row("g0", big, softmax_reference(big))
    assert admit([row], reference_fn=_overflowing_reference) == Rejection(
        reason=NONFINITE_REFERENCE, groups_broken=0, groups_judgeable=0, groups_unverifiable=1
    )


def test_an_unverifiable_group_does_not_block_a_broken_one():
    # Unverifiable is abstention, not disqualification: a candidate broken where it
    # could be compared is admitted, exactly as one that crashes somewhere is.
    big = np.full((2, 4), 100.0, dtype=np.float32)
    small = np.zeros((2, 4), dtype=np.float32)
    rows = [
        _softmax_row("g0", big, softmax_reference(big)),
        _softmax_row("g1", small, np.full((2, 4), 9.0, dtype=np.float32)),
    ]
    assert admit(rows, reference_fn=_overflowing_reference) is True


def test_an_unverifiable_group_is_counted_beside_a_matching_one():
    # The reason stays the published "not broken" category; the record says how many
    # groups that statement could not cover.
    big = np.full((2, 4), 100.0, dtype=np.float32)
    small = np.zeros((2, 4), dtype=np.float32)
    rows = [
        _softmax_row("g0", big, softmax_reference(big)),
        _softmax_row("g1", small, softmax_reference(small)),
    ]
    assert admit(rows, reference_fn=_overflowing_reference) == Rejection(
        reason=NOT_BROKEN, groups_broken=0, groups_judgeable=1, groups_unverifiable=1
    )


def test_both_causes_are_named_when_no_group_could_be_compared():
    # One group has an int output, the other a NaN reference. Naming only one cause
    # would send whoever reads the tally to fix half the problem.
    small = np.zeros((2, 4), dtype=np.float32)
    big = np.full((2, 4), 100.0, dtype=np.float32)
    rows = [
        _softmax_row("g0", small, np.zeros((2, 4), dtype=np.int32)),
        _softmax_row("g1", big, softmax_reference(big)),
    ]
    assert admit(rows, reference_fn=_overflowing_reference) == Rejection(
        reason=(
            "not verifiable on any group: the output dtype is exact, so no test ratio is "
            "defined for it; and the reference output is non-finite, a defect of the "
            "reference and not of the candidate"
        ),
        groups_broken=0, groups_judgeable=0, groups_unverifiable=2,
    )
