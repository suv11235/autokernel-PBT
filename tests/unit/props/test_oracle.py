"""The three oracle arms.

The research design rests on these arms *disagreeing* on some kernels: if the
reference arm and the declarative arm always reached the same verdict, there would
be no comparison to report. So this file does not only test each arm in isolation —
it constructs kernels the arms judge differently and pins the disagreement down,
in both directions (a bug only the reference arm sees, and one only the declarative
arm can see).

The other load-bearing assertions here:

* every result from every arm is attributed to exactly one of case_id/group_id, on
  the PASS, FAIL and INCONCLUSIVE paths alike — a flat concatenated list from
  ``HybridOracle`` is not resplittable otherwise;
* the hybrid short-circuit is *real*, proved with a reference function that raises
  if it is ever called;
* an integer output reaches the reference arm as INCONCLUSIVE, never FAIL and never
  a traceback.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, ExecutionResult, Status
from autokernel_pbt.props.case import BASE_RELATION, Case
from autokernel_pbt.props.oracle import (
    REFERENCE_PROPERTY,
    DeclarativeOracle,
    HybridOracle,
    Oracle,
    ReferenceOracle,
    summary,
    validate_property_set,
)
from autokernel_pbt.props.properties import (
    SOFTMAX_CASE_PROPERTIES,
    SOFTMAX_GROUP_PROPERTIES,
    OutputsAreFinite,
    RowsSumToOne,
    ShiftInvariance,
    ValuesInUnitInterval,
)
from autokernel_pbt.props.relations import ShiftRows
from autokernel_pbt.props.tolerance import DEFAULT_THRESH
from autokernel_pbt.props.verdict import TIER_PORTABLE, PropertyResult, Verdict

EPS32 = float(np.finfo(np.float32).eps)

X = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float32)
SHIFT = np.array([[10.0], [-5.0]], dtype=np.float32)


# --------------------------------------------------------------------------
# Fixtures: kernels and rows
# --------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return (e / e.sum(axis=-1, keepdims=True)).astype(x.dtype)


def _sharpened(x: np.ndarray) -> np.ndarray:
    """softmax(2x): a valid, shift-invariant distribution that is the wrong one.

    Every declarative law holds — the values are in [0, 1], the rows sum to one, and
    softmax(2(x + c)) == softmax(2x) — but it is not the kernel that was asked for.
    This is the shape of bug only the reference arm can see.
    """
    return _softmax(2.0 * x).astype(x.dtype)


def _onehot_int(x: np.ndarray) -> np.ndarray:
    """An argmax-style integer kernel whose values escape [0, 1]."""
    out = np.zeros(x.shape, dtype=np.int64)
    out[:, 0] = 7
    return out


def _row(
    x: np.ndarray,
    y: np.ndarray | None,
    relation: str = BASE_RELATION,
    status: Status = Status.OK,
) -> ExecutionResult:
    case = Case(
        case_id=f"c-{relation}",
        group_id="g0",
        relation=relation,
        task_id="softmax",
        dtype=str(x.dtype),
        shape=x.shape,
        tensors={"x": x},
    )
    outputs = {} if y is None else {OUTPUT_NAME: y}
    return ExecutionResult(case=case, outputs=outputs, telemetry={}, status=status)


def _group(fn) -> list[ExecutionResult]:
    """A base row plus its shift partner, both produced by ``fn``."""
    shifted = X + SHIFT
    return [_row(X, fn(X)), _row(shifted, fn(shifted), ShiftRows.name)]


def _declarative() -> DeclarativeOracle:
    return DeclarativeOracle(SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES)


def _reference() -> ReferenceOracle:
    return ReferenceOracle(_softmax)


def _hybrid() -> HybridOracle:
    return HybridOracle(_declarative(), _reference())


def _exploding_reference() -> ReferenceOracle:
    """A reference arm that makes any consultation of it observable as an error."""

    def boom(**_: np.ndarray) -> np.ndarray:
        raise AssertionError("reference arm was consulted")

    return ReferenceOracle(boom)


def _arms() -> list[Oracle]:
    return [_declarative(), _reference(), _hybrid()]


# --------------------------------------------------------------------------
# Protocol, names, shared summary
# --------------------------------------------------------------------------


def test_every_arm_satisfies_the_oracle_protocol():
    for arm in _arms():
        assert isinstance(arm, Oracle)


def test_arms_are_named_for_the_report_layer():
    assert [arm.name for arm in _arms()] == ["declarative", "reference", "hybrid"]


def test_summary_delegates_to_the_verdict_precedence():
    """One combination rule for all three arms; FAIL dominates, empty is inconclusive."""
    def r(verdict: Verdict) -> PropertyResult:
        return PropertyResult("p", TIER_PORTABLE, True, verdict, case_id="c")

    assert summary([]) is Verdict.INCONCLUSIVE
    assert summary([r(Verdict.PASS)]) is Verdict.PASS
    assert summary([r(Verdict.PASS), r(Verdict.INCONCLUSIVE)]) is Verdict.INCONCLUSIVE
    assert summary([r(Verdict.PASS), r(Verdict.FAIL)]) is Verdict.FAIL
    assert summary([r(Verdict.INCONCLUSIVE), r(Verdict.FAIL)]) is Verdict.FAIL


# --------------------------------------------------------------------------
# A. Property-set validation at construction
# --------------------------------------------------------------------------


def test_property_set_missing_a_deferral_target_is_rejected():
    """A set that defers non-finite output to a property it does not contain is blind.

    values_in_unit_interval and rows_sum_to_one both return INCONCLUSIVE on NaN so
    that outputs_are_finite owns the finding. Without outputs_are_finite in the set,
    a NaN-producing kernel produces nothing but INCONCLUSIVE and the declarative arm
    silently records a miss — an understatement of the very arm being measured.
    """
    with pytest.raises(ValueError, match="defers non-finite output"):
        validate_property_set([ValuesInUnitInterval(), RowsSumToOne()])


def test_declarative_oracle_rejects_a_blind_property_set_at_construction():
    with pytest.raises(ValueError, match="defers non-finite output"):
        DeclarativeOracle([ValuesInUnitInterval(), RowsSumToOne()])


def test_declarative_oracle_rejects_a_blind_group_property_set():
    """Group properties defer too, and are validated against the same name pool."""
    with pytest.raises(ValueError, match="defers non-finite output"):
        DeclarativeOracle([ValuesInUnitInterval()], [ShiftInvariance()])


def test_a_satisfied_property_set_is_accepted():
    validate_property_set([OutputsAreFinite(), ValuesInUnitInterval(), RowsSumToOne()])
    validate_property_set([*SOFTMAX_CASE_PROPERTIES, *SOFTMAX_GROUP_PROPERTIES])
    # The shipped bundle is exactly what DeclarativeOracle is built with elsewhere.
    assert _declarative().name == "declarative"


def test_an_empty_property_set_is_rejected():
    """An oracle with no properties judges nothing and reports INCONCLUSIVE.

    Same defect as an empty case group, through a different door: it adds a group
    that established nothing to the denominator of the detection rate, with no error
    anywhere. Task 13 builds sets from acceptance.yaml by name, so an omitted or
    misspelled `properties:` key reaches exactly here.
    """
    with pytest.raises(ValueError, match="empty property set"):
        validate_property_set([])
    with pytest.raises(ValueError, match="empty property set"):
        DeclarativeOracle([])
    with pytest.raises(ValueError, match="empty property set"):
        DeclarativeOracle([], [])


def test_an_empty_property_set_would_otherwise_report_inconclusive():
    """Guards the test above by showing what it prevents, via the raw summary path."""
    assert summary([]) is Verdict.INCONCLUSIVE


def test_a_property_set_with_no_deferrals_at_all_is_accepted():
    validate_property_set([OutputsAreFinite()])
    assert DeclarativeOracle([OutputsAreFinite()]).case_properties


def test_validation_is_not_vacuous_on_the_blind_set():
    """Guards the rejection test: the blind set really does catch nothing on NaN.

    Without this, a future refactor could make the rejection test pass for the wrong
    reason. Here the blind set is run directly (bypassing the constructor) and shown
    to produce only INCONCLUSIVE on an output that is unambiguously defective.
    """
    bad = _softmax(X).copy()
    bad[0, 0] = np.nan
    row = _row(X, bad)
    blind = [ValuesInUnitInterval(), RowsSumToOne()]
    assert all(p.check(row).verdict is Verdict.INCONCLUSIVE for p in blind)
    assert OutputsAreFinite().check(row).verdict is Verdict.FAIL


# --------------------------------------------------------------------------
# PROPERTY_ATTRIBUTION (acceptance criterion, spelled exactly)
# --------------------------------------------------------------------------


def test_declarative_oracle_records_tolerance_free_flag():
    """Each verdict carries the property name and its tolerance-free flag.

    This is the headline claim's bookkeeping: "bugs found without a tolerance
    argument" is only answerable if every result says which property produced it and
    whether that property consulted a tolerance at all.
    """
    results = _declarative().evaluate(_group(_softmax))

    by_name = {r.property_name: r for r in results}
    assert by_name["outputs_are_finite"].tolerance_free is True
    assert by_name["values_in_unit_interval"].tolerance_free is True
    assert by_name["rows_sum_to_one"].tolerance_free is False
    assert by_name["shift_invariance"].tolerance_free is False
    # Every result names a property and a tier; nothing is anonymous.
    assert all(r.property_name for r in results)
    assert all(r.tier == TIER_PORTABLE for r in results)


def test_reference_oracle_records_a_tolerance_dependent_property():
    """The reference arm is by construction not tolerance-free: it compares numbers."""
    results = _reference().evaluate(_group(_softmax))
    assert results
    for result in results:
        assert result.property_name == REFERENCE_PROPERTY == "matches_reference"
        assert result.tolerance_free is False
        assert result.tier == TIER_PORTABLE


# --------------------------------------------------------------------------
# C. Attribution on every path
# --------------------------------------------------------------------------


def _assert_attributed(results: list[PropertyResult]) -> None:
    assert results, "no results to check attribution on"
    for result in results:
        assert bool(result.case_id) != bool(result.group_id), (
            f"{result.property_name} produced an unattributed (or doubly attributed) "
            f"result: case_id={result.case_id!r} group_id={result.group_id!r}"
        )


@pytest.mark.parametrize(
    ("label", "kernel"),
    [
        ("pass", _softmax),
        ("fail", _sharpened),
        ("inconclusive", _onehot_int),
    ],
)
def test_every_arm_attributes_every_result(label, kernel):
    """PASS, FAIL and INCONCLUSIVE alike. A flat hybrid list is not resplittable
    otherwise, and an orphaned result joins to no recorded execution."""
    rows = _group(kernel)
    for arm in _arms():
        _assert_attributed(arm.evaluate(rows))


def test_reference_oracle_attributes_an_unusable_row():
    """The path with no output at all still has a case_id in hand."""
    results = _reference().evaluate([_row(X, None, status=Status.LAUNCH_ERROR)])
    _assert_attributed(results)
    assert results[0].verdict is Verdict.INCONCLUSIVE
    assert results[0].case_id == "c-base"


def test_reference_oracle_attributes_a_row_missing_its_output():
    results = _reference().evaluate([_row(X, None)])
    assert results[0].verdict is Verdict.INCONCLUSIVE
    assert results[0].case_id == "c-base"


def test_reference_oracle_attributes_an_empty_output():
    """Zero elements: no evidence, and `n=0` would abort the run inside residual_ratio."""
    empty = np.zeros((2, 0), dtype=np.float32)
    results = _reference().evaluate([_row(empty, empty)])
    assert results[0].verdict is Verdict.INCONCLUSIVE
    assert results[0].case_id == "c-base"


def test_reference_results_are_one_per_row():
    rows = _group(_softmax)
    results = _reference().evaluate(rows)
    assert [r.case_id for r in results] == [row.case.case_id for row in rows]


# --------------------------------------------------------------------------
# D. The integer-output path
# --------------------------------------------------------------------------


def test_reference_oracle_is_inconclusive_on_an_integer_output():
    """ExactDtypeError becomes INCONCLUSIVE — never FAIL, never a traceback.

    PERSISTABLE_KINDS is "biuf", so an int-returning kernel genuinely reaches this
    arm. FAIL would record a possibly-correct int kernel as a caught bug, straight
    into the headline detection metric; propagating would abort a run whose hardware
    time cannot be recovered.
    """
    y = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int64)
    results = _reference().evaluate([_row(X, y)])
    assert len(results) == 1
    assert results[0].verdict is Verdict.INCONCLUSIVE
    assert results[0].case_id == "c-base"
    assert "exact dtype" in results[0].detail


def test_reference_oracle_is_inconclusive_on_a_boolean_output():
    y = np.array([[False, True, False], [True, False, False]])
    assert _reference().evaluate([_row(X, y)])[0].verdict is Verdict.INCONCLUSIVE


# --------------------------------------------------------------------------
# ReferenceOracle: the ordinary pass/fail behaviour
# --------------------------------------------------------------------------


def test_reference_oracle_passes_on_the_reference_kernel_itself():
    assert summary(_reference().evaluate(_group(_softmax))) is Verdict.PASS


def test_reference_oracle_fails_on_a_wrong_kernel():
    assert summary(_reference().evaluate(_group(_sharpened))) is Verdict.FAIL


def test_reference_oracle_fails_on_a_wrong_shaped_kernel_output():
    """The detection the shape branch must not lose: the kernel returned the wrong shape."""
    y = _softmax(X)[:, :2]
    result = _reference().evaluate([_row(X, y)])[0]
    assert result.verdict is Verdict.FAIL
    assert "shape mismatch" in result.detail


def test_reference_oracle_threshold_is_configurable_and_used():
    """A thresh that admits nothing turns a correct kernel into a FAIL.

    Without this the `thresh` argument could be ignored entirely and every other
    test in this file would still pass.
    """
    strict = ReferenceOracle(_softmax, thresh=0.0)
    assert strict.evaluate([_row(X, _softmax(X))])[0].verdict is Verdict.FAIL
    assert _reference().evaluate([_row(X, _softmax(X))])[0].verdict is Verdict.PASS


def test_reference_oracle_reads_the_kernel_inputs_not_the_bookkeeping_tensors():
    """Helper tensors are generator bookkeeping and are not kernel arguments.

    reference_fn is called with **kernel_inputs(case); a helper leaking in would be
    an unexpected-keyword TypeError, which is why this is asserted rather than assumed.
    """
    row = _row(X, _softmax(X))
    row.case.tensors["__perm__"] = np.array([0, 1, 2])
    seen: dict[str, np.ndarray] = {}

    def spy(**kwargs: np.ndarray) -> np.ndarray:
        seen.update(kwargs)
        return _softmax(kwargs["x"])

    assert ReferenceOracle(spy).evaluate([row])[0].verdict is Verdict.PASS
    assert set(seen) == {"x"}


# --------------------------------------------------------------------------
# A broken reference is the harness's fault, not the kernel's
# --------------------------------------------------------------------------


def _reference_defect_result(bad_reference) -> PropertyResult:
    row = _row(X, _softmax(X))  # the kernel is CORRECT in every one of these
    return ReferenceOracle(bad_reference).evaluate([row])[0]


def test_a_nan_returning_reference_does_not_fail_a_correct_kernel():
    """residual_ratio answers inf for a non-finite reference, and inf is a FAIL.

    Left unchecked that books a correct kernel as a caught bug — a false positive in
    the headline metric, manufactured by the harness — with a detail string that
    names the kernel and sends triage to the wrong place entirely.
    """
    result = _reference_defect_result(lambda **kw: np.full_like(kw["x"], np.nan))
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "the reference, not the kernel" in result.detail
    assert result.case_id == "c-base"


def test_a_shape_disagreement_is_a_fail_that_does_not_blame_either_side():
    """Shape mismatch is symmetric, so it is deliberately NOT the reference's fault.

    Nothing here knows the correct output shape independently, so "the reference is
    wrong" and "the kernel returned the wrong shape" are the same observation.
    Downgrading it to INCONCLUSIVE would permanently blind the strong arm to
    wrong-shaped output — a common, serious kernel bug this arm exists to catch —
    to guard against a harness bug that disagrees on *every* row and so is loudly
    visible. The honest thing is to FAIL and say in the detail that the arm cannot
    tell which side is wrong.
    """
    result = _reference_defect_result(lambda **kw: _softmax(kw["x"])[:, :2])
    assert result.verdict is Verdict.FAIL
    assert "cannot tell which" in result.detail
    # Both shapes are named, so triage can see at a glance which side looks wrong.
    assert "(2, 3)" in result.detail and "(2, 2)" in result.detail


def test_an_overflowing_reference_neither_aborts_nor_fails_a_correct_kernel():
    """The config-dependent path, which is disqualifying for a replay-fairness project.

    An unstable reference (exp with no max-subtraction) overflows to inf. Under this
    project's `filterwarnings = ["error"]` that RuntimeWarning is raised, aborting the
    whole scoring pass partway and discarding the already-paid-for hardware time in
    the rest of the table; under production config the same defect silently FAILs a
    correct kernel. Two different behaviours from one reference bug. Suppressing numpy
    error state around the reference call and validating its output collapses both to
    one INCONCLUSIVE row.
    """
    def unstable(**kw: np.ndarray) -> np.ndarray:
        e = np.exp(kw["x"] * np.float32(1e3))  # overflows float32
        return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

    result = _reference_defect_result(unstable)  # must not raise
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "the reference, not the kernel" in result.detail


def test_a_reference_that_raises_still_propagates():
    """The defensible core of the original position, now actually pinned.

    An exception is unambiguous and cannot be mistaken for a finding about the
    kernel. Mapping it to INCONCLUSIVE would let a wholly broken reference reduce
    this arm to a silent no-op across an entire run, which is the one failure the
    comparison cannot detect from its own output.
    """
    def broken(**_: np.ndarray) -> np.ndarray:
        raise RuntimeError("reference implementation is broken")

    with pytest.raises(RuntimeError, match="reference implementation is broken"):
        ReferenceOracle(broken).evaluate([_row(X, _softmax(X))])


def test_reference_validation_does_not_swallow_a_real_kernel_defect():
    """Guards the three tests above: the check is on the reference side only.

    A NaN in the *kernel's* output must still FAIL — it is the kernel's defect, and
    the reference arm is the strong arm precisely because it sees it.
    """
    bad = _softmax(X).copy()
    bad[0, 0] = np.nan
    assert _reference().evaluate([_row(X, bad)])[0].verdict is Verdict.FAIL


def test_a_zero_dimensional_reference_matches_a_normalized_output():
    """single_output promotes 0-d to (1,); a plain reduction reference stays 0-d.

    The shape validation must not reject that legitimate pairing, or every reduction
    kernel in the corpus would come back INCONCLUSIVE.
    """
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    got = np.atleast_1d(np.sum(x))  # what the execution boundary records
    result = ReferenceOracle(lambda **kw: np.sum(kw["x"])).evaluate([_row(x, got)])[0]
    assert result.verdict is Verdict.PASS


# --------------------------------------------------------------------------
# The accumulation length comes from the input, not the output
# --------------------------------------------------------------------------


def test_reduction_ratio_normalizes_by_the_input_length_not_the_output_length():
    """A reduction's output last axis is what the error was reduced *into*.

    This is the RowsSumToOne trap one module over, on the other side of the
    comparison. The kernel here is correct to within 120 ulps of a unit-scale
    reference, over an accumulation of 262144 elements. Normalized by the input's
    last axis the ratio is 120/log2(262144) = 6.67 and PASSes; normalized by the
    output's last axis (1) the divisor floors to 1.0, the ratio is 120, and a correct
    kernel is booked as a caught bug. The two straddle DEFAULT_THRESH = 30, so this
    test flips outright if the length is taken from the wrong side.
    """
    cols = 262144
    x = np.zeros((2, cols), dtype=np.float32)
    got = np.array([np.float32(1.0) + 120 * EPS32], dtype=np.float32)

    result = ReferenceOracle(lambda **kw: np.array([1.0], dtype=np.float32)).evaluate(
        [_row(x, got)]
    )[0]

    assert result.verdict is Verdict.PASS
    ratio = float(result.detail.split("=")[-1])
    assert ratio == pytest.approx(120.0 / np.log2(cols), rel=1e-3)
    assert ratio < DEFAULT_THRESH < 120.0


def test_the_accumulation_length_can_be_overridden_for_a_contraction():
    """GEMM's K is not an output dimension and not the input's last axis either."""
    cols = 262144
    x = np.zeros((2, 4), dtype=np.float32)  # input says 4; the real contraction is huge
    got = np.array([np.float32(1.0) + 120 * EPS32], dtype=np.float32)
    ref = lambda **kw: np.array([1.0], dtype=np.float32)

    assert ReferenceOracle(ref).evaluate([_row(x, got)])[0].verdict is Verdict.FAIL
    assert ReferenceOracle(ref, n=cols).evaluate([_row(x, got)])[0].verdict is Verdict.PASS


def test_a_scalar_input_shape_falls_back_to_no_length_normalization():
    """`shape=()` has no last axis, and n=0 would abort the pass inside residual_ratio."""
    x = np.array(2.0, dtype=np.float32)
    got = np.atleast_1d(np.float32(4.0))
    result = ReferenceOracle(lambda **kw: np.atleast_1d(kw["x"] * 2)).evaluate([_row(x, got)])[0]
    assert result.verdict is Verdict.PASS


def test_a_zero_length_input_axis_does_not_abort_the_scoring_pass():
    """residual_ratio rejects n < 1 with a bare ValueError; that must not escape."""
    x = np.zeros((2, 0), dtype=np.float32)
    got = np.array([1.0], dtype=np.float32)
    result = ReferenceOracle(lambda **kw: np.array([1.0], dtype=np.float32)).evaluate(
        [_row(x, got)]
    )[0]
    assert result.verdict is Verdict.PASS


# --------------------------------------------------------------------------
# DeclarativeOracle
# --------------------------------------------------------------------------


def test_declarative_oracle_runs_case_properties_per_row_and_group_once():
    rows = _group(_softmax)
    results = _declarative().evaluate(rows)

    case_results = [r for r in results if r.case_id]
    group_results = [r for r in results if r.group_id]
    assert len(case_results) == len(SOFTMAX_CASE_PROPERTIES) * len(rows)
    assert len(group_results) == len(SOFTMAX_GROUP_PROPERTIES)
    assert {r.group_id for r in group_results} == {"g0"}


def test_declarative_oracle_with_no_group_properties_emits_only_case_results():
    results = DeclarativeOracle(SOFTMAX_CASE_PROPERTIES).evaluate(_group(_softmax))
    assert all(r.case_id and not r.group_id for r in results)


def test_declarative_oracle_passes_on_a_correct_kernel():
    assert summary(_declarative().evaluate(_group(_softmax))) is Verdict.PASS


def test_declarative_oracle_catches_a_missing_normalization():
    unnormalized = [
        _row(X, np.exp(X - X.max(-1, keepdims=True)).astype(np.float32)),
        _row(
            X + SHIFT,
            np.exp((X + SHIFT) - (X + SHIFT).max(-1, keepdims=True)).astype(np.float32),
            ShiftRows.name,
        ),
    ]
    assert summary(_declarative().evaluate(unnormalized)) is Verdict.FAIL


# --------------------------------------------------------------------------
# B. The hybrid short-circuit is real
# --------------------------------------------------------------------------


def test_hybrid_does_not_consult_the_reference_arm_when_the_laws_fail():
    """The cost saving the hybrid arm exists to demonstrate.

    Proved, not asserted by inspection: the reference function raises if it is ever
    called, so a clean return is evidence that it was not. A reference recompute is
    the expensive half of the comparison, so skipping it once the cheap laws have
    already reached FAIL is the arm's whole thesis.
    """
    hybrid = HybridOracle(_declarative(), _exploding_reference())
    rows = _group(_onehot_int)  # values_in_unit_interval FAILs: 7 is outside [0, 1]

    results = hybrid.evaluate(rows)  # must not raise

    assert summary(results) is Verdict.FAIL
    assert all(r.property_name != REFERENCE_PROPERTY for r in results)


def test_hybrid_consults_the_reference_arm_when_the_laws_pass():
    """The opposite direction, so the short-circuit test above is not vacuous."""
    with pytest.raises(AssertionError, match="reference arm was consulted"):
        HybridOracle(_declarative(), _exploding_reference()).evaluate(_group(_softmax))


def test_hybrid_consults_the_reference_arm_when_the_laws_are_inconclusive():
    """INCONCLUSIVE is not FAIL: the laws established nothing, so reference is the
    only thing left that could. Short-circuiting here would lose detections."""
    hybrid = HybridOracle(_declarative(), _reference())
    y = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int64)  # in [0,1], but exact dtype
    rows = [_row(X, y)]
    results = hybrid.evaluate(rows)
    assert any(r.property_name == REFERENCE_PROPERTY for r in results)


def test_hybrid_returns_the_declarative_results_verbatim_when_it_short_circuits():
    rows = _group(_onehot_int)
    declarative_only = _declarative().evaluate(rows)
    hybrid_results = HybridOracle(_declarative(), _exploding_reference()).evaluate(rows)
    assert [(r.property_name, r.verdict) for r in hybrid_results] == [
        (r.property_name, r.verdict) for r in declarative_only
    ]


def test_hybrid_concatenates_both_arms_when_it_does_not_short_circuit():
    rows = _group(_softmax)
    expected = len(_declarative().evaluate(rows)) + len(_reference().evaluate(rows))
    assert len(_hybrid().evaluate(rows)) == expected


# --------------------------------------------------------------------------
# F. The short-circuit's effect on later analysis is recoverable
# --------------------------------------------------------------------------


def test_hybrid_reference_coverage_is_readable_off_the_results():
    """Whether hybrid consulted the reference arm must be recoverable after the fact.

    The short-circuit makes hybrid's reference coverage *conditional*: on a
    declarative-FAIL group it never learns what reference would have said. For the
    detection-rate table that is harmless (both arms caught it), but a cost-per-bug
    or per-property attribution analysis must be able to exclude the groups where
    the reference arm never ran. No extra field is needed for that — the presence or
    absence of a `matches_reference` result is itself the record — but it is only a
    usable record if it is asserted to be reliable, which is what this pins down.
    """
    consulted = _hybrid().evaluate(_group(_softmax))
    skipped = _hybrid().evaluate(_group(_onehot_int))

    assert any(r.property_name == REFERENCE_PROPERTY for r in consulted)
    assert not any(r.property_name == REFERENCE_PROPERTY for r in skipped)
    # And the reference arm's own name is unique to it, so the marker cannot collide
    # with a declarative property.
    declarative_names = {r.property_name for r in _declarative().evaluate(_group(_softmax))}
    assert REFERENCE_PROPERTY not in declarative_names


# --------------------------------------------------------------------------
# E. The arms actually disagree
# --------------------------------------------------------------------------


def test_arms_disagree_on_a_law_abiding_but_wrong_kernel():
    """softmax(2x): every declarative law holds, and the answer is still wrong.

    This is the reference arm's reason to exist. It is a valid probability
    distribution (values in [0, 1], rows sum to one) and it is genuinely shift
    invariant, so the declarative arm has nothing to say against it — but it is not
    the requested kernel, and the reference arm sees that immediately.
    """
    rows = _group(_sharpened)
    assert summary(_declarative().evaluate(rows)) is Verdict.PASS
    assert summary(_reference().evaluate(rows)) is Verdict.FAIL
    assert summary(_hybrid().evaluate(rows)) is Verdict.FAIL


def test_arms_disagree_on_an_integer_kernel_the_reference_arm_cannot_judge():
    """The opposite direction: a defect only the declarative arm can reach.

    An integer output has no unit roundoff, so a normalized test ratio is undefined
    and the reference arm must answer INCONCLUSIVE. The structural law needs no
    tolerance at all and reports the violation — a bug caught with no tolerance
    argument, which is precisely the headline claim.
    """
    rows = _group(_onehot_int)
    assert summary(_reference().evaluate(rows)) is Verdict.INCONCLUSIVE
    assert summary(_declarative().evaluate(rows)) is Verdict.FAIL
    assert summary(_hybrid().evaluate(rows)) is Verdict.FAIL

    caught = [
        r for r in _declarative().evaluate(rows)
        if r.verdict is Verdict.FAIL
    ]
    assert caught and all(r.tolerance_free for r in caught)


def test_the_arms_agree_on_the_easy_cases_so_the_disagreements_are_meaningful():
    """A correct kernel passes everywhere and a NaN kernel fails everywhere.

    Without this, "the arms disagree" would be unremarkable — an arm that answered
    at random would also disagree.
    """
    good = _group(_softmax)
    assert all(summary(arm.evaluate(good)) is Verdict.PASS for arm in _arms())

    nan_y = _softmax(X).copy()
    nan_y[0, 0] = np.nan
    bad = [_row(X, nan_y), _row(X + SHIFT, nan_y, ShiftRows.name)]
    assert all(summary(arm.evaluate(bad)) is Verdict.FAIL for arm in _arms())


# --------------------------------------------------------------------------
# G. The empty group
# --------------------------------------------------------------------------


def test_every_arm_rejects_an_empty_group():
    """An empty group can only be an evaluation-layer bug, so it raises.

    read_groups() never yields an empty list and CaseGroup rejects a group with no
    base, so there is no legitimate path here. Returning [] instead would summarize
    to INCONCLUSIVE and quietly add a group that judged nothing to the denominator
    of the detection rate. Raising uniformly across the three arms also removes an
    inconsistency that would otherwise be latent: delegating to the properties makes
    the behaviour depend on whether the set happens to contain a group property —
    ShiftInvariance.check_group([]) raises, while a case-only set would silently
    return [].
    """
    for arm in _arms():
        with pytest.raises(ValueError, match="empty"):
            arm.evaluate([])
    # Case-only sets raise too, not just the ones carrying a group property.
    with pytest.raises(ValueError, match="empty"):
        DeclarativeOracle(SOFTMAX_CASE_PROPERTIES).evaluate([])


def test_the_empty_group_error_names_the_arm():
    with pytest.raises(ValueError, match="reference"):
        _reference().evaluate([])
