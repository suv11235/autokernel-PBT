"""Tier-1 property tests.

Every property here is asserted to be *able to fail*: a property that cannot fail
contributes nothing to the detection metric it feeds. Every property is also
asserted to attribute each result to exactly one of case_id/group_id, and to
return INCONCLUSIVE — never FAIL, never an exception — on each of the inputs it
cannot judge.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, ExecutionResult, Status
from autokernel_pbt.props.case import BASE_RELATION, Case
from autokernel_pbt.props.properties import (
    CASE_PROPERTY_REGISTRY,
    GROUP_PROPERTY_REGISTRY,
    LAYERNORM_CASE_PROPERTIES,
    LAYERNORM_GROUP_PROPERTIES,
    SOFTMAX_CASE_PROPERTIES,
    SOFTMAX_GROUP_PROPERTIES,
    CaseProperty,
    GroupProperty,
    OutputsAreFinite,
    RowsHaveUnitVariance,
    RowsHaveZeroMean,
    RowsSumToOne,
    ShiftInvariance,
    ValuesInUnitInterval,
    _result,
)
from autokernel_pbt.props.relations import ShiftRows
from autokernel_pbt.props.tolerance import DEFAULT_THRESH
from autokernel_pbt.props.verdict import TIER_PORTABLE, PropertyResult, Verdict

EPS32 = float(np.finfo(np.float32).eps)

X = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float32)
SHIFT = np.array([[10.0], [-5.0]], dtype=np.float32)


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


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return (e / e.sum(axis=-1, keepdims=True)).astype(x.dtype)


def _unnormalized(x: np.ndarray) -> np.ndarray:
    """exp(x) with no denominator — the classic missing-normalization bug."""
    return np.exp(x - x.max(axis=-1, keepdims=True)).astype(x.dtype)


def _naive_normalize(x: np.ndarray) -> np.ndarray:
    """x / sum(x): sums to one, but genuinely not shift invariant."""
    return (x / x.sum(axis=-1, keepdims=True)).astype(x.dtype)


def _shift_group(fn) -> list[ExecutionResult]:
    shifted = X + SHIFT
    return [_row(X, fn(X)), _row(shifted, fn(shifted), ShiftRows.name)]


CASE_PROPS = [OutputsAreFinite, ValuesInUnitInterval, RowsSumToOne]


# --------------------------------------------------------------------------
# Declared metadata and registries
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "name", "tolerance_free"),
    [
        (OutputsAreFinite, "outputs_are_finite", True),
        (ValuesInUnitInterval, "values_in_unit_interval", True),
        (RowsSumToOne, "rows_sum_to_one", False),
        (ShiftInvariance, "shift_invariance", False),
    ],
)
def test_declared_metadata(cls, name, tolerance_free):
    prop = cls()
    assert prop.name == name
    assert prop.tier == TIER_PORTABLE
    assert prop.tolerance_free is tolerance_free


def test_shift_invariance_declares_the_relation_it_consumes():
    # Bound to the relation class so the two names cannot drift apart.
    assert ShiftInvariance().requires_relation == ShiftRows.name == "shift_rows"


def test_nonfinite_deferral_is_declared_as_data_not_prose():
    """The properties that defer non-finite output must name their dependency.

    ValuesInUnitInterval and RowsSumToOne return INCONCLUSIVE on NaN/Inf so the
    defect is counted once, by OutputsAreFinite. That makes them *dependent*: a
    property set built from acceptance.yaml that lists them without
    outputs_are_finite is structurally incapable of catching a NaN-producing
    kernel, and would silently record a miss for the declarative arm. Task 13
    validates this at config-load time, which it can only do if the dependency is
    machine-readable here.
    """
    assert ValuesInUnitInterval.defers_nonfinite_to == "outputs_are_finite"
    assert RowsSumToOne.defers_nonfinite_to == "outputs_are_finite"
    # ShiftInvariance defers only the base side; the partner side is its own finding.
    assert ShiftInvariance.defers_nonfinite_to == "outputs_are_finite"
    # OutputsAreFinite is the terminus: it defers to nobody.
    assert OutputsAreFinite.defers_nonfinite_to == ""


def test_every_declared_deferral_names_a_registered_property():
    registries = {**CASE_PROPERTY_REGISTRY, **GROUP_PROPERTY_REGISTRY}
    for cls in registries.values():
        target = cls.defers_nonfinite_to
        assert target == "" or target in CASE_PROPERTY_REGISTRY, (
            f"{cls.__name__} defers to {target!r}, which no registry key provides"
        )


def test_registries_are_keyed_by_property_name():
    for name, cls in CASE_PROPERTY_REGISTRY.items():
        assert cls().name == name
    for name, cls in GROUP_PROPERTY_REGISTRY.items():
        assert cls().name == name
    assert set(CASE_PROPERTY_REGISTRY) == {
        "outputs_are_finite",
        "values_in_unit_interval",
        "rows_sum_to_one",
        "rows_have_zero_mean",
        "rows_have_unit_variance",
    }
    assert set(GROUP_PROPERTY_REGISTRY) == {"shift_invariance"}


def test_registered_classes_conform_to_their_scope_protocol():
    """Runtime-checkable, because ruff is the only static gate in this repo.

    An annotation of dict[str, type[CaseProperty]] buys nothing without a type
    checker; an isinstance assertion at import time is what actually stops a class
    missing `check` from being registered as a case property.
    """
    for cls in CASE_PROPERTY_REGISTRY.values():
        assert isinstance(cls(), CaseProperty)
    for cls in GROUP_PROPERTY_REGISTRY.values():
        assert isinstance(cls(), GroupProperty)
    # The two scopes are distinguishable, so the check above is not vacuous.
    assert not isinstance(ShiftInvariance(), CaseProperty)
    assert not isinstance(OutputsAreFinite(), GroupProperty)


def test_per_task_bundles_are_a_subset_of_the_registries():
    """Subset, not equality, and the difference is load-bearing.

    The bundles used to be built from every registered property, which was correct
    only while every property happened to hold for softmax. layernorm's
    rows_have_zero_mean holds for no softmax output at all -- a softmax row sums to
    one, so its mean is 1/n -- and a registry-wide bundle would have made the
    declarative arm fail every correct softmax kernel while looking like a detection.
    """
    registered_cases = set(CASE_PROPERTY_REGISTRY.values())
    registered_groups = set(GROUP_PROPERTY_REGISTRY.values())
    for bundle in (SOFTMAX_CASE_PROPERTIES, LAYERNORM_CASE_PROPERTIES):
        assert {type(p) for p in bundle} <= registered_cases
    for bundle in (SOFTMAX_GROUP_PROPERTIES, LAYERNORM_GROUP_PROPERTIES):
        assert {type(p) for p in bundle} <= registered_groups
    # Non-vacuous: the two task bundles genuinely differ.
    assert {type(p) for p in SOFTMAX_CASE_PROPERTIES} != {
        type(p) for p in LAYERNORM_CASE_PROPERTIES
    }


# --------------------------------------------------------------------------
# The _result helper always attributes
# --------------------------------------------------------------------------


def test_result_helper_rejects_an_orphaned_result():
    with pytest.raises(ValueError, match="exactly one"):
        _result(OutputsAreFinite(), Verdict.PASS)


def test_result_helper_rejects_a_doubly_attributed_result():
    with pytest.raises(ValueError, match="exactly one"):
        _result(OutputsAreFinite(), Verdict.PASS, case_id="c", group_id="g")


# --------------------------------------------------------------------------
# OutputsAreFinite
# --------------------------------------------------------------------------


def test_outputs_are_finite_passes_on_correct_output():
    assert OutputsAreFinite().check(_row(X, _softmax(X))).verdict is Verdict.PASS


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_outputs_are_finite_fails_on_non_finite(bad_value):
    bad = _softmax(X).copy()
    bad[0, 0] = bad_value
    assert OutputsAreFinite().check(_row(X, bad)).verdict is Verdict.FAIL


def test_outputs_are_finite_accepts_an_integer_output():
    # isfinite is defined on ints; no tolerance is consulted, so no ExactDtypeError.
    y = np.array([[0, 1], [1, 0]], dtype=np.int64)
    assert OutputsAreFinite().check(_row(X, y)).verdict is Verdict.PASS


# --------------------------------------------------------------------------
# ValuesInUnitInterval
# --------------------------------------------------------------------------


def test_values_in_unit_interval_passes_on_correct_output():
    assert ValuesInUnitInterval().check(_row(X, _softmax(X))).verdict is Verdict.PASS


def test_values_in_unit_interval_fails_on_negative():
    bad = _softmax(X).copy()
    bad[0, 0] = -0.1
    assert ValuesInUnitInterval().check(_row(X, bad)).verdict is Verdict.FAIL


def test_values_in_unit_interval_fails_above_one():
    assert ValuesInUnitInterval().check(_row(X, _unnormalized(X) * 3.0)).verdict is Verdict.FAIL


@pytest.mark.parametrize(
    "boundary",
    [np.float32(0.0), np.float32(1.0), np.float32(-0.0), np.float32(0.5)],
)
def test_values_in_unit_interval_admits_the_exact_boundaries(boundary):
    y = np.full((2, 3), boundary, dtype=np.float32)
    assert ValuesInUnitInterval().check(_row(X, y)).verdict is Verdict.PASS


@pytest.mark.parametrize(
    "outside",
    [
        np.nextafter(np.float32(1.0), np.float32(2.0)),   # 1 ulp above 1.0
        np.nextafter(np.float32(0.0), np.float32(-1.0)),  # 1 ulp below 0.0
    ],
)
def test_values_in_unit_interval_rejects_a_one_ulp_excursion(outside):
    """The bound is exact, with no slack — that is what tolerance_free = True claims.

    A slack of even 1e-3 would still pass every other test in this file, so without
    this the module's headline tag would be unverified. Exactness is defensible
    because softmax's rounding error is one-sided down: over all 2^23 float32
    mantissas in [1, 2), fl(x * fl(1/x)) never exceeds 1.0, so a correct kernel
    cannot land here by rounding alone.
    """
    y = _softmax(X).copy()
    y[0, 0] = outside
    assert float(y[0, 0]) not in (0.0, 1.0)
    assert ValuesInUnitInterval().check(_row(X, y)).verdict is Verdict.FAIL


@pytest.mark.parametrize(
    "outside",
    [
        np.nextafter(np.float32(1.0), np.float32(2.0)),
        np.nextafter(np.float32(0.0), np.float32(-1.0)),
    ],
)
def test_boundary_violation_detail_is_legible_at_one_ulp(outside):
    """The reported bounds must be distinguishable from the interval they violate.

    A rounded format renders a one-ulp overshoot as "range [1, 1] outside [0, 1]",
    which tells a human staring at the boundary case nothing at all. `detail` is
    read by people during triage and parsed by nothing, so full round-trip precision
    is the only thing it owes them. Asserted because a reformat would silently undo it.
    """
    y = _softmax(X).copy()
    y[0, 0] = outside
    detail = ValuesInUnitInterval().check(_row(X, y)).detail

    reported, _, interval = detail.partition(" outside ")
    assert interval == "[0, 1]"
    assert reported != f"range {interval}", "bounds rendered indistinguishably from the interval"
    assert repr(outside) in reported
    # The offending value survives the round trip through the message.
    assert np.float32(repr(outside).split("(")[-1].rstrip(")")) == outside


def test_values_in_unit_interval_is_inconclusive_on_non_finite_output():
    # Reporting NaN here as well as in OutputsAreFinite would count one defect twice.
    bad = _softmax(X).copy()
    bad[0, 0] = np.nan
    result = ValuesInUnitInterval().check(_row(X, bad))
    assert result.verdict is Verdict.INCONCLUSIVE


def test_values_in_unit_interval_judges_an_integer_output():
    y = np.array([[0, 2], [1, 0]], dtype=np.int64)
    assert ValuesInUnitInterval().check(_row(X, y)).verdict is Verdict.FAIL


# --------------------------------------------------------------------------
# RowsSumToOne
# --------------------------------------------------------------------------


def test_rows_sum_to_one_passes_on_correct_output():
    assert RowsSumToOne().check(_row(X, _softmax(X))).verdict is Verdict.PASS


def test_rows_sum_to_one_fails_on_unnormalized_output():
    assert RowsSumToOne().check(_row(X, _unnormalized(X))).verdict is Verdict.FAIL


def test_rows_sum_to_one_is_inconclusive_on_non_finite_output():
    bad = _softmax(X).copy()
    bad[0, 0] = np.inf
    assert RowsSumToOne().check(_row(X, bad)).verdict is Verdict.INCONCLUSIVE


def test_rows_sum_to_one_is_inconclusive_on_an_integer_output():
    """ExactDtypeError must become INCONCLUSIVE, not FAIL and not a traceback.

    A one-hot int output genuinely sums to one; calling it a caught bug would be a
    false positive injected straight into the headline detection metric, and letting
    the exception escape would abort the whole run.
    """
    y = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int64)
    result = RowsSumToOne().check(_row(X, y))
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.case_id == "c-base"


def test_rows_sum_to_one_normalizes_by_the_reduction_length_not_the_row_count():
    """The `n=` keyword must be the reduction length (cols), not `sums`' last axis.

    Rows and columns differ by three orders of magnitude here, so the two choices
    differ by log2(4096)/log2(2) = 12x. The row sum is exact by construction (all
    but one element is zero), so the measured ratio is entirely the injected error:
    120 eps of deviation is 10.0 under the correct normalization and 120.0 under the
    default, which straddles DEFAULT_THRESH = 30. Passing the default would flip
    this test from PASS to FAIL.
    """
    rows, cols = 2, 4096
    y = np.zeros((rows, cols), dtype=np.float32)
    y[:, 0] = np.float32(1.0 + 120 * EPS32)  # exactly 120 ulps of 1.0
    result = RowsSumToOne().check(_row(np.zeros((rows, cols), dtype=np.float32), y))

    assert result.verdict is Verdict.PASS
    ratio = float(result.detail.split("=")[-1])
    assert ratio == pytest.approx(120.0 / np.log2(cols), rel=1e-3)
    assert ratio < DEFAULT_THRESH < 120.0 / max(np.log2(rows), 1.0)


# --------------------------------------------------------------------------
# ShiftInvariance
# --------------------------------------------------------------------------


def test_shift_invariance_passes_on_correct_softmax():
    assert ShiftInvariance().check_group(_shift_group(_softmax)).verdict is Verdict.PASS


def test_shift_invariance_fails_on_non_invariant_kernel():
    assert ShiftInvariance().check_group(_shift_group(_naive_normalize)).verdict is Verdict.FAIL


def _overflowing_softmax_output() -> np.ndarray:
    """What exp-without-max-subtraction produces once the shift reaches overflow."""
    big = np.array([[100.0, 101.0, 102.0]], dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(big)
        y = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    assert not np.all(np.isfinite(y))
    return y


def test_shift_invariance_fails_when_only_the_shifted_output_is_non_finite():
    """The unstable softmax the relation exists to catch: exp without max-subtraction.

    Non-finite output on the *partner* is a FAIL, not an INCONCLUSIVE deferral to
    OutputsAreFinite: overflowing under a large shift while the unshifted input was
    fine *is* the shift-invariance violation, and deferring it would leave the
    relation unable to catch the one bug its shift scale was chosen for.
    """
    base_y = _softmax(np.array([[0.0, 1.0, 2.0]], dtype=np.float32))
    group = [_row(X, base_y), _row(X, _overflowing_softmax_output(), ShiftRows.name)]
    assert ShiftInvariance().check_group(group).verdict is Verdict.FAIL


def test_shift_invariance_is_inconclusive_when_the_base_output_is_non_finite():
    """A kernel that NaNs on the *unshifted* input has no invariance finding against it.

    That defect is outputs_are_finite's, and claiming it here would inflate
    shift_invariance's per-property detection count — a reported quantity. The
    asymmetry is the point: the base is the reference the relation compares to, so
    a broken base means there is nothing to compare against, not a broken relation.
    """
    shifted_y = _softmax(np.array([[0.0, 1.0, 2.0]], dtype=np.float32))
    group = [_row(X, _overflowing_softmax_output()), _row(X, shifted_y, ShiftRows.name)]
    assert ShiftInvariance().check_group(group).verdict is Verdict.INCONCLUSIVE


def test_shift_invariance_is_inconclusive_when_both_outputs_are_non_finite():
    y = _overflowing_softmax_output()
    group = [_row(X, y), _row(X, y.copy(), ShiftRows.name)]
    assert ShiftInvariance().check_group(group).verdict is Verdict.INCONCLUSIVE


def test_shift_invariance_is_inconclusive_on_an_integer_output():
    y = np.array([[0, 1, 0]], dtype=np.int64)
    group = [_row(X, y), _row(X, y, ShiftRows.name)]
    assert ShiftInvariance().check_group(group).verdict is Verdict.INCONCLUSIVE


# --------------------------------------------------------------------------
# E: everything that makes a property INCONCLUSIVE, enumerated
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CASE_PROPS)
def test_case_property_is_inconclusive_on_failed_execution(cls):
    """A non-OK status disqualifies the row *even though an output is present*.

    Status and output-presence must be independent axes. Today NumpyBackend never
    populates outputs on a failure, so a row with both a bad status and no output
    cannot tell the two apart — and a check that only looked at the output would
    pass every such test. A Phase 3 timeout with a partially written device buffer
    breaks that correlation, and this module would then judge garbage as valid.
    """
    row = _row(X, _softmax(X), status=Status.TIMEOUT)
    assert row.outputs, "the point of this test is a failed row that still has an output"
    assert cls().check(row).verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize("cls", CASE_PROPS)
def test_case_property_is_inconclusive_when_y_is_missing(cls):
    # status ok but the output name is absent: a backend contract violation, not a bug.
    row = _row(X, None)
    assert row.status == Status.OK, "the missing output must be the only defect here"
    assert cls().check(row).verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize("cls", CASE_PROPS)
def test_case_property_detail_distinguishes_bad_status_from_missing_output(cls):
    # "status=ok" is a misleading cause when the real defect is an absent output.
    missing = cls().check(_row(X, None)).detail
    failed = cls().check(_row(X, _softmax(X), status=Status.TIMEOUT)).detail
    assert OUTPUT_NAME in missing and "status" not in missing
    assert "timeout" in failed


@pytest.mark.parametrize("cls", CASE_PROPS)
def test_case_property_is_inconclusive_on_empty_output(cls):
    """Zero elements is not evidence, and it is not a defect either.

    Vacuous truth would make OutputsAreFinite/ValuesInUnitInterval report PASS, and
    RowsSumToOne would hand n=0 to residual_ratio, which rejects it with a bare
    ValueError that would abort the run.
    """
    empty = np.zeros((2, 0), dtype=np.float32)
    assert cls().check(_row(X, empty)).verdict is Verdict.INCONCLUSIVE


def test_shift_invariance_is_inconclusive_without_partner():
    rows = [_row(X, _softmax(X))]
    assert ShiftInvariance().check_group(rows).verdict is Verdict.INCONCLUSIVE


def test_shift_invariance_is_inconclusive_without_base():
    shifted = X + SHIFT
    rows = [_row(shifted, _softmax(shifted), ShiftRows.name)]
    assert ShiftInvariance().check_group(rows).verdict is Verdict.INCONCLUSIVE


def test_shift_invariance_rejects_an_empty_group():
    """An empty group is an oracle bug, not a judgeable input.

    Task 11 forms groups by grouping the replayed table on group_id, which never
    yields an empty group, and CaseGroup separately rejects one without a base. So
    this can only arrive from a coding error — and emitting a result for it would
    invent an INCONCLUSIVE that joins to no row, corrupting the very count this
    module exists to keep honest. Raising is safe here specifically because
    evaluation is offline: no hardware time is lost to a re-run.
    """
    with pytest.raises(ValueError, match="empty group"):
        ShiftInvariance().check_group([])


@pytest.mark.parametrize("failed_index", [0, 1])
def test_shift_invariance_is_inconclusive_when_either_execution_failed(failed_index):
    # Status only — the outputs stay in place, so this tests its name rather than
    # duplicating the missing-output case below.
    group = _shift_group(_softmax)
    group[failed_index].status = Status.LAUNCH_ERROR
    assert all(r.outputs for r in group)
    assert ShiftInvariance().check_group(group).verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize("missing_index", [0, 1])
def test_shift_invariance_is_inconclusive_when_either_output_is_missing(missing_index):
    group = _shift_group(_softmax)
    group[missing_index].outputs = {}
    assert ShiftInvariance().check_group(group).verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize("empty_index", [0, 1])
def test_shift_invariance_is_inconclusive_on_empty_output(empty_index):
    group = _shift_group(_softmax)
    group[empty_index].outputs = {OUTPUT_NAME: np.zeros((2, 0), dtype=np.float32)}
    assert ShiftInvariance().check_group(group).verdict is Verdict.INCONCLUSIVE


# --------------------------------------------------------------------------
# D: attribution, on every property and every verdict path
# --------------------------------------------------------------------------


def _assert_attributed(result: PropertyResult, *, case_id: str = "", group_id: str = "") -> None:
    assert isinstance(result, PropertyResult)
    assert bool(result.case_id) != bool(result.group_id), "exactly one id must be set"
    if case_id:
        assert result.case_id == case_id and result.group_id == ""
    if group_id:
        assert result.group_id == group_id and result.case_id == ""


@pytest.mark.parametrize(
    ("cls", "passing_y", "failing_y"),
    [
        (OutputsAreFinite, _softmax(X), np.full_like(X, np.nan)),
        (ValuesInUnitInterval, _softmax(X), _softmax(X) - 1.0),
        (RowsSumToOne, _softmax(X), _unnormalized(X)),
    ],
)
def test_case_property_attributes_every_verdict_to_its_case(cls, passing_y, failing_y):
    prop = cls()
    passed = prop.check(_row(X, passing_y))
    failed = prop.check(_row(X, failing_y))
    inconclusive = prop.check(_row(X, None, status=Status.LAUNCH_ERROR))

    assert (passed.verdict, failed.verdict) == (Verdict.PASS, Verdict.FAIL)
    assert inconclusive.verdict is Verdict.INCONCLUSIVE
    for result in (passed, failed, inconclusive):
        _assert_attributed(result, case_id="c-base")
        assert result.property_name == prop.name
        assert result.tier == prop.tier
        assert result.tolerance_free is prop.tolerance_free


def test_group_property_attributes_every_verdict_to_its_group():
    prop = ShiftInvariance()
    passed = prop.check_group(_shift_group(_softmax))
    failed = prop.check_group(_shift_group(_naive_normalize))
    inconclusive = prop.check_group([_row(X, _softmax(X))])

    assert (passed.verdict, failed.verdict) == (Verdict.PASS, Verdict.FAIL)
    assert inconclusive.verdict is Verdict.INCONCLUSIVE
    for result in (passed, failed, inconclusive):
        _assert_attributed(result, group_id="g0")
        assert result.property_name == prop.name


# --------------------------------------------------------------------------
# F: the unnormalized-kernel case, where two properties both fire
# --------------------------------------------------------------------------


def test_unnormalized_kernel_fails_both_independent_properties():
    """exp(x) trips both value-range and row-sum, and that is intended.

    The properties are independent predicates, not a partition of defect space:
    each holds for every correct implementation on its own, and neither implies
    the other (a uniform 0.5-everywhere output is in range but does not sum to
    one; a one-hot-with-a-negative output sums to one but is out of range). The
    detection metric counts *groups caught*, not property firings, so overlap
    costs nothing; suppressing one of them would only reduce the chance that a
    kernel disabling the other still gets caught.
    """
    y = _unnormalized(X) * 4.0
    assert ValuesInUnitInterval().check(_row(X, y)).verdict is Verdict.FAIL
    assert RowsSumToOne().check(_row(X, y)).verdict is Verdict.FAIL

    # Neither property implies the other.
    in_range_but_unnormalized = np.full_like(X, 0.5)
    assert ValuesInUnitInterval().check(_row(X, in_range_but_unnormalized)).verdict is Verdict.PASS
    assert RowsSumToOne().check(_row(X, in_range_but_unnormalized)).verdict is Verdict.FAIL

    out_of_range = np.array([[-0.5, 0.5, 1.0], [2.0, -0.5, -0.5]], dtype=np.float32)
    assert ValuesInUnitInterval().check(_row(X, out_of_range)).verdict is Verdict.FAIL
    assert RowsSumToOne().check(_row(X, out_of_range)).verdict is Verdict.PASS


# --------------------------------------------------------------------------
# layernorm properties (feature 0006)
# --------------------------------------------------------------------------


def _layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    wide = np.asarray(x, dtype=np.float64)
    centered = wide - np.mean(wide, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return (centered / np.sqrt(variance + eps)).astype(x.dtype)


#: Wide, like the layernorm task's own domain. On N(0, 1) a merely-centered output
#: already has variance near 1, which is the measurement that moved the task off it.
WIDE = np.array([[-9.0, -3.0, 4.0, 8.0], [7.0, -6.0, 2.0, -1.0]], dtype=np.float32)


def test_rows_have_zero_mean_passes_a_normalized_output():
    assert RowsHaveZeroMean().check(_row(WIDE, _layernorm(WIDE))).verdict is Verdict.PASS


def test_rows_have_zero_mean_fails_an_unsubtracted_output():
    """The criterion LAYERNORM_PROPERTIES_DETECT.

    A kernel that scales but never subtracts the mean -- the defect this property
    exists to catch, and one no reference implementation is needed to see.
    """
    wide = np.asarray(WIDE, dtype=np.float64)
    scaled_only = (wide / np.sqrt(wide.var(axis=-1, keepdims=True) + 1e-5)).astype(np.float32)
    assert RowsHaveZeroMean().check(_row(WIDE, scaled_only)).verdict is Verdict.FAIL


def test_rows_have_unit_variance_passes_a_normalized_output():
    assert RowsHaveUnitVariance().check(_row(WIDE, _layernorm(WIDE))).verdict is Verdict.PASS


def test_rows_have_unit_variance_fails_an_unscaled_output():
    # A kernel that centers but never divides: passes the mean check, fails this one.
    wide = np.asarray(WIDE, dtype=np.float64)
    centered_only = (wide - wide.mean(axis=-1, keepdims=True)).astype(np.float32)
    assert RowsHaveUnitVariance().check(_row(WIDE, centered_only)).verdict is Verdict.FAIL


def test_the_two_layernorm_properties_are_complementary():
    """Neither alone pins the normalized family; that is why the contract has both.

    Measured on the ladder: a centers-but-never-divides kernel is caught by
    unit-variance on 14 of 18 cases and by zero-mean on none of them, while a
    divides-but-never-centers kernel is caught by zero-mean on all 18 and by
    unit-variance on none. A contract carrying only one of the pair would be blind
    to exactly one of the two defects.
    """
    wide = np.asarray(WIDE, dtype=np.float64)
    centered_only = (wide - wide.mean(axis=-1, keepdims=True)).astype(np.float32)
    scaled_only = (wide / np.sqrt(wide.var(axis=-1, keepdims=True) + 1e-5)).astype(np.float32)

    assert RowsHaveZeroMean().check(_row(WIDE, centered_only)).verdict is Verdict.PASS
    assert RowsHaveUnitVariance().check(_row(WIDE, centered_only)).verdict is Verdict.FAIL
    assert RowsHaveZeroMean().check(_row(WIDE, scaled_only)).verdict is Verdict.FAIL
    assert RowsHaveUnitVariance().check(_row(WIDE, scaled_only)).verdict is Verdict.PASS


def test_unit_variance_abstains_on_a_constant_input():
    # Layernorm of a constant row is identically zero, not unit variance, because eps
    # dominates the denominator. Correct behaviour, reached at the (1,1) and (17,1)
    # rungs -- judging it would fail every correct kernel on two of nine rungs.
    constant = np.full((2, 4), 3.0, dtype=np.float32)
    y = np.zeros((2, 4), dtype=np.float32)
    assert RowsHaveUnitVariance().check(_row(constant, y)).verdict is Verdict.INCONCLUSIVE


def test_unit_variance_fails_an_all_zero_output_from_a_varying_input():
    """The abstention is keyed off the INPUT, not the output, and the gap matters.

    A kernel returning a zeroed or uninitialized buffer produces a flat output from a
    varying input. Keyed off the output, this abstained -- and every other layernorm
    property passes such a kernel (zero mean holds; shift invariance holds because
    0 == 0), so the group summarized to INCONCLUSIVE, the driver refused the run as
    an arm that established nothing, and NO scores were written at all -- discarding
    the allclose, reference and hybrid verdicts that each caught it on every group.
    """
    y = np.zeros_like(WIDE)
    assert RowsHaveUnitVariance().check(_row(WIDE, y)).verdict is Verdict.FAIL


def test_unit_variance_abstains_only_on_the_constant_rows():
    # A mixed case: one constant row, one varying. The varying row must still be
    # judged, or a single degenerate row would blind the property to the whole case.
    x = np.array([[5.0, 5.0, 5.0, 5.0], [-9.0, -3.0, 4.0, 8.0]], dtype=np.float32)
    y = np.stack([np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)])
    assert RowsHaveUnitVariance().check(_row(x, y)).verdict is Verdict.FAIL


def test_neither_layernorm_property_is_tolerance_free():
    # A float sum of n centered values is not exactly zero. Claiming otherwise would
    # inflate the tolerance-free count the project's sharpest claim rests on.
    assert RowsHaveZeroMean().tolerance_free is False
    assert RowsHaveUnitVariance().tolerance_free is False


def test_layernorm_properties_are_inconclusive_on_a_failed_execution():
    for prop in (RowsHaveZeroMean(), RowsHaveUnitVariance()):
        row = _row(WIDE, None, status=Status.LAUNCH_ERROR)
        assert prop.check(row).verdict is Verdict.INCONCLUSIVE, prop.name


def test_layernorm_properties_defer_non_finite_output():
    # One defect counted once: OutputsAreFinite owns the NaN finding.
    y = np.full((2, 4), np.nan, dtype=np.float32)
    for prop in (RowsHaveZeroMean(), RowsHaveUnitVariance()):
        assert prop.check(_row(WIDE, y)).verdict is Verdict.INCONCLUSIVE, prop.name


def test_layernorm_properties_pass_the_real_reference():
    """No false alarms anywhere on the ladder, with the margin pinned.

    This is the test that caught the original defect. The deviation of a correct
    output's variance from 1 is the reference's own ``eps`` regularization --
    ``var/(var + eps) - 1``, which grows as row variance shrinks -- not rounding. On
    N(0, 1) inputs that term reached 3.2e-5 against a 2.9e-5 budget and FAILed a
    correct reference. The task's wide input distribution is what keeps it ~46x
    below the budget, so this test fails loudly if that distribution is narrowed.
    """
    from autokernel_pbt.props.generator import Generator
    from autokernel_pbt.props.tasks import REFERENCES, TASKS

    worst = 0.0
    for group in Generator(TASKS["layernorm"].domain, seed=0).generate(9):
        for case in group.cases:
            y = REFERENCES["layernorm"](x=case.tensors["x"])
            row = ExecutionResult(case=case, outputs={OUTPUT_NAME: y})
            for prop in LAYERNORM_CASE_PROPERTIES:
                verdict = prop.check(row).verdict
                assert verdict is not Verdict.FAIL, f"{prop.name} on {case.shape}"
            variances = np.var(np.asarray(y, dtype=np.float64), axis=-1)
            judged = variances[~np.isclose(variances, 0.0, atol=1e-12)]
            if judged.size:
                worst = max(worst, float(np.max(np.abs(judged - 1.0))))

    budget = DEFAULT_THRESH * float(np.finfo(np.float32).eps) * 8
    assert worst < budget / 10, f"margin has collapsed to {budget / worst:.1f}x"


def test_layernorm_bundle_excludes_softmax_only_properties():
    # The bundles name their members explicitly. A registry-wide bundle would put
    # rows_have_zero_mean into softmax's set, where it holds for no output at all.
    softmax_names = {p.name for p in SOFTMAX_CASE_PROPERTIES}
    layernorm_names = {p.name for p in LAYERNORM_CASE_PROPERTIES}
    assert "values_in_unit_interval" not in layernorm_names
    assert "rows_sum_to_one" not in layernorm_names
    assert "rows_have_zero_mean" not in softmax_names
    assert "rows_have_unit_variance" not in softmax_names
    assert {p.name for p in LAYERNORM_GROUP_PROPERTIES} == {"shift_invariance"}
