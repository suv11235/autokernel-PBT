"""Tier-1 (portable/semantic) properties.

Tier-1 properties are pure functions of (inputs, outputs) and hold for any correct
implementation on any backend — they are the cross-backend equivalence contract the
translation workstream consumes. ``tolerance_free`` marks the ones that reach a
verdict with no numerical tolerance argument at all; that tag drives the headline
claim ("bugs found without a tolerance argument"), so it is load-bearing metadata,
not documentation.

Two scopes. A ``CaseProperty`` judges one recorded row. A ``GroupProperty`` judges a
whole case group and names, in ``requires_relation``, the metamorphic partner it
needs. Both are evaluated offline against the recorded execution table, so neither
may raise: an exception here aborts a run whose hardware time cannot be recovered.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from autokernel_pbt.props.backends.base import OUTPUT_NAME, ExecutionResult, Status
from autokernel_pbt.props.case import BASE_RELATION
from autokernel_pbt.props.relations import ShiftRows
from autokernel_pbt.props.tolerance import (
    DEFAULT_THRESH,
    ExactDtypeError,
    residual_ratio,
    within_threshold,
)
from autokernel_pbt.props.verdict import TIER_PORTABLE, PropertyResult, Verdict

# TIER_PORTABLE/TIER_BACKEND live in verdict.py, which validates tier values. Declaring
# them here too would assert the 1<->portable mapping in two files and let it drift.

# Detail strings the oracles and the report layer read back. An exact-dtype output is
# distinguishable from a genuinely unjudgeable one because the two mean different
# things for the denominator of the detection rate.
_EXACT_DTYPE_DETAIL = "exact dtype: test ratio undefined"
_EMPTY_DETAIL = "empty output: nothing to judge"
_NONFINITE_DETAIL = "non-finite output"


class _Declared(Protocol):
    """The metadata every property carries, whatever its scope."""

    name: str
    tier: int
    tolerance_free: bool


class CaseProperty(_Declared, Protocol):
    """Judges one recorded row."""

    def check(self, row: ExecutionResult) -> PropertyResult: ...


class GroupProperty(_Declared, Protocol):
    """Judges a whole case group; names the metamorphic partner it needs."""

    requires_relation: str

    def check_group(self, rows: list[ExecutionResult]) -> PropertyResult: ...


def _result(
    prop: _Declared,
    verdict: Verdict,
    detail: str = "",
    *,
    case_id: str = "",
    group_id: str = "",
) -> PropertyResult:
    """Build a result, always attributing it to the case or group it judged.

    Exactly one of case_id/group_id must be set: case properties set case_id, group
    properties set group_id. A result with neither is orphaned — ``HybridOracle``
    concatenates the declarative and reference arms and the split point is not
    recoverable from a flat list, so a missing id cannot be repaired downstream. The
    check is an assertion about this module's own call sites, which is why it raises
    rather than returning INCONCLUSIVE: it can only fire on a coding error, and the
    property bodies below are careful never to reach a verdict without an id in hand.
    """
    if bool(case_id) == bool(group_id):
        msg = (
            f"property {prop.name!r} produced an unattributed result: exactly one of "
            f"case_id/group_id must be set, got case_id={case_id!r} group_id={group_id!r}"
        )
        raise ValueError(msg)
    return PropertyResult(
        property_name=prop.name,
        tier=prop.tier,
        tolerance_free=prop.tolerance_free,
        verdict=verdict,
        detail=detail,
        case_id=case_id,
        group_id=group_id,
    )


def _usable(row: ExecutionResult) -> bool:
    """Whether the row carries an output this layer can judge at all.

    ``Status`` is a str-mixin enum, so this holds for a replayed row whose status is
    a ``Status`` member and for one carrying the bare wire string.
    """
    return row.status == Status.OK and OUTPUT_NAME in row.outputs


def _unjudgeable(y: np.ndarray) -> str:
    """Why ``y`` cannot be judged numerically, or "" if it can.

    Empty is not a defect and not evidence either: zero elements make every
    elementwise predicate vacuously true, and ``residual_ratio`` answers NaN for it,
    which ``within_threshold`` reads as a non-pass — a FAIL if taken at face value.
    Worse, a zero reduction length reaches ``residual_ratio``'s ``n=`` validation,
    which rejects it with a bare ValueError that would abort the whole evaluation.
    """
    if y.size == 0:
        return _EMPTY_DETAIL
    return ""


class OutputsAreFinite:
    """No NaN or Inf anywhere in the output.

    The most primitive portable property, and the only one that reports non-finite
    output as a defect. The others defer to it so one defect is counted once.
    """

    name = "outputs_are_finite"
    tier = TIER_PORTABLE
    tolerance_free = True

    def check(self, row: ExecutionResult) -> PropertyResult:
        case_id = row.case.case_id
        if not _usable(row):
            return _result(self, Verdict.INCONCLUSIVE, f"status={row.status}", case_id=case_id)
        y = row.outputs[OUTPUT_NAME]
        if reason := _unjudgeable(y):
            return _result(self, Verdict.INCONCLUSIVE, reason, case_id=case_id)
        if not np.all(np.isfinite(y)):
            return _result(self, Verdict.FAIL, "output contains NaN or Inf", case_id=case_id)
        return _result(self, Verdict.PASS, case_id=case_id)


class ValuesInUnitInterval:
    """Every output value lies in [0, 1]. Structural: no tolerance is consulted.

    A non-finite output is INCONCLUSIVE, not FAIL. NaN compares false against both
    bounds, so the natural reading would be "out of range" — but that is
    ``OutputsAreFinite``'s finding, and counting one defect as two would inflate the
    per-property detection numbers without catching anything extra.
    """

    name = "values_in_unit_interval"
    tier = TIER_PORTABLE
    tolerance_free = True

    def check(self, row: ExecutionResult) -> PropertyResult:
        case_id = row.case.case_id
        if not _usable(row):
            return _result(self, Verdict.INCONCLUSIVE, f"status={row.status}", case_id=case_id)
        y = row.outputs[OUTPUT_NAME]
        if reason := _unjudgeable(y):
            return _result(self, Verdict.INCONCLUSIVE, reason, case_id=case_id)
        if not np.all(np.isfinite(y)):
            return _result(self, Verdict.INCONCLUSIVE, _NONFINITE_DETAIL, case_id=case_id)
        low, high = float(np.min(y)), float(np.max(y))
        if low < 0.0 or high > 1.0:
            detail = f"range [{low:.6g}, {high:.6g}] outside [0, 1]"
            return _result(self, Verdict.FAIL, detail, case_id=case_id)
        return _result(self, Verdict.PASS, f"range=[{low:.6g}, {high:.6g}]", case_id=case_id)


class RowsSumToOne:
    """Each row of the output sums to 1, within a normalized test ratio."""

    name = "rows_sum_to_one"
    tier = TIER_PORTABLE
    tolerance_free = False

    def check(self, row: ExecutionResult) -> PropertyResult:
        case_id = row.case.case_id
        if not _usable(row):
            return _result(self, Verdict.INCONCLUSIVE, f"status={row.status}", case_id=case_id)
        y = row.outputs[OUTPUT_NAME]
        if reason := _unjudgeable(y):
            return _result(self, Verdict.INCONCLUSIVE, reason, case_id=case_id)
        if not np.all(np.isfinite(y)):
            # OutputsAreFinite reports this; a sum over NaN is not a row-sum defect.
            return _result(self, Verdict.INCONCLUSIVE, _NONFINITE_DETAIL, case_id=case_id)

        sums = np.sum(y, axis=-1)
        try:
            # n= explicitly: `sums` has shape (rows,), so the default last-axis length
            # would be the ROW COUNT, not the reduction length the log2(n) rounding
            # budget is meant to model. The two differ by orders of magnitude on a wide
            # output, and getting it wrong is silent — it only moves the pass/fail line.
            ratio = residual_ratio(sums, np.ones_like(sums), dtype=y.dtype, n=y.shape[-1])
        except ExactDtypeError:
            # An int-returning kernel can reach here (PERSISTABLE_KINDS includes "iu"),
            # and its rows may sum to one exactly. FAIL would record a correct kernel as
            # a caught bug — a false positive straight into the headline metric — and
            # letting it propagate would abort the run. Only INCONCLUSIVE is honest.
            return _result(self, Verdict.INCONCLUSIVE, _EXACT_DTYPE_DETAIL, case_id=case_id)

        if not within_threshold(ratio):
            detail = f"row-sum test ratio {ratio:.3g} >= {DEFAULT_THRESH}"
            return _result(self, Verdict.FAIL, detail, case_id=case_id)
        return _result(self, Verdict.PASS, f"ratio={ratio:.3g}", case_id=case_id)


class ShiftInvariance:
    """f(x + c) == f(x) for a per-row constant c. Needs the group's shift partner.

    Non-finite output is a FAIL here rather than a deferral to ``OutputsAreFinite``:
    a softmax without max-subtraction overflows precisely *because* of the shift, so
    overflow under a shifted input is the invariance violation, and the shift scale
    in ``ShiftRows`` was chosen to reach that regime. Deferring it would leave this
    relation unable to catch the one bug it exists for.
    """

    name = "shift_invariance"
    tier = TIER_PORTABLE
    tolerance_free = False
    # Bound to the relation class so a rename cannot leave the property looking for a
    # partner no generator produces — which would be silent, and INCONCLUSIVE forever.
    requires_relation = ShiftRows.name

    def check_group(self, rows: list[ExecutionResult]) -> PropertyResult:
        # Read the group id off any row; an empty group has none, but the result still
        # must not be orphaned, so fall back to a marker rather than an empty string.
        group_id = rows[0].case.group_id if rows else "<empty-group>"
        base = next((r for r in rows if r.case.relation == BASE_RELATION), None)
        partner = next((r for r in rows if r.case.relation == self.requires_relation), None)
        if base is None or partner is None:
            detail = f"group missing {BASE_RELATION!r} or {self.requires_relation!r} case"
            return _result(self, Verdict.INCONCLUSIVE, detail, group_id=group_id)
        if not _usable(base) or not _usable(partner):
            detail = (
                f"group contains a failed execution "
                f"(base={base.status}, partner={partner.status})"
            )
            return _result(self, Verdict.INCONCLUSIVE, detail, group_id=group_id)

        base_y = base.outputs[OUTPUT_NAME]
        shifted_y = partner.outputs[OUTPUT_NAME]
        for candidate in (base_y, shifted_y):
            if reason := _unjudgeable(candidate):
                return _result(self, Verdict.INCONCLUSIVE, reason, group_id=group_id)

        try:
            ratio = residual_ratio(shifted_y, base_y, dtype=base_y.dtype)
        except ExactDtypeError:
            return _result(self, Verdict.INCONCLUSIVE, _EXACT_DTYPE_DETAIL, group_id=group_id)

        if not within_threshold(ratio):
            detail = f"shift test ratio {ratio:.3g} >= {DEFAULT_THRESH}"
            return _result(self, Verdict.FAIL, detail, group_id=group_id)
        return _result(self, Verdict.PASS, f"ratio={ratio:.3g}", group_id=group_id)


# Name -> class, so Task 13 can build a property set from a kernel's acceptance.yaml
# by name. Keyed off each class's own ``name`` rather than a hand-written literal, so
# the registry key and the recorded ``property_name`` cannot disagree.
CASE_PROPERTY_REGISTRY: dict[str, type] = {
    cls.name: cls for cls in (OutputsAreFinite, ValuesInUnitInterval, RowsSumToOne)
}
GROUP_PROPERTY_REGISTRY: dict[str, type] = {cls.name: cls for cls in (ShiftInvariance,)}

SOFTMAX_CASE_PROPERTIES: tuple[CaseProperty, ...] = tuple(
    cls() for cls in CASE_PROPERTY_REGISTRY.values()
)
SOFTMAX_GROUP_PROPERTIES: tuple[GroupProperty, ...] = tuple(
    cls() for cls in GROUP_PROPERTY_REGISTRY.values()
)
