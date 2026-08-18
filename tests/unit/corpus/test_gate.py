"""Admission-gate tests.

An agent-authored mutant cannot be taken at its word, and each way it can be wrong
corrupts a different number invisibly. These pin the two admission criteria and the
recording of rejections.
"""

from __future__ import annotations

import numpy as np

from autokernel_pbt.corpus.gate import Rejection, admit
from autokernel_pbt.props.backends.base import OUTPUT_NAME, ExecutionResult, Status
from autokernel_pbt.props.case import Case


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
