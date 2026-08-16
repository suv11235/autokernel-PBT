"""The three oracle arms.

This module is the experiment. Everything upstream of it exists to produce a
replayable execution table; everything here is a *strategy* for turning that table
into verdicts, and the research question is which strategy catches more bugs at
what false-positive cost. The arms form a spectrum from one strong property to many
weak ones:

``ReferenceOracle``
    One property: the output matches a trusted recomputation. Maximally strong —
    it can see any deviation, including ones no algebraic law constrains — and
    maximally brittle: it needs a reference implementation, a tolerance, and a
    dtype with a unit roundoff, and answers INCONCLUSIVE whenever it lacks one.

``DeclarativeOracle``
    Many algebraic and metamorphic laws. Each is individually weak (a wrong kernel
    can satisfy any one of them) but jointly constraining, and several need no
    tolerance argument at all — which is the headline claim this project makes.

``HybridOracle``
    A composition with precedence: run the cheap laws first, and only pay for the
    reference recompute if they found nothing. The laws act as a filter.

Two invariants hold across all three arms.

*Attribution.* Every ``PropertyResult`` carries exactly one of ``case_id`` /
``group_id``. ``HybridOracle`` concatenates two arms into one flat list, and the
split point is not recoverable from the list itself, so a result that names neither
the case nor the group it judged is orphaned for good. ``properties._result``
enforces this for the declarative arm; ``ReferenceOracle`` builds its results
directly and must therefore enforce it on *every* return path, the inconclusive one
included.

*INCONCLUSIVE is load-bearing.* It is the arm saying "I could not judge this",
which is neither a detection nor a false positive. Over-returning it deflates the
detection rate; under-returning it inflates the false-positive rate. Both corrupt
the metric the whole project reports, so each INCONCLUSIVE path below says why the
alternatives are wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from autokernel_pbt.props.backends.base import (
    OUTPUT_NAME,
    ExecutionResult,
    Status,
    kernel_inputs,
)
from autokernel_pbt.props.properties import CaseProperty, GroupProperty
from autokernel_pbt.props.tolerance import (
    DEFAULT_THRESH,
    ExactDtypeError,
    residual_ratio,
    within_threshold,
)
from autokernel_pbt.props.verdict import TIER_PORTABLE, PropertyResult, Verdict, summarize

#: The reference arm's single property. Named here rather than inlined because it is
#: also the marker that says whether ``HybridOracle`` actually consulted the
#: reference arm on a given group — see ``HybridOracle`` for why that matters.
REFERENCE_PROPERTY = "matches_reference"

# Detail strings, mirroring properties.py's vocabulary so a triage pass reading a
# mixed result list sees one language rather than two.
_EXACT_DTYPE_DETAIL = "exact dtype: test ratio undefined"
_EMPTY_DETAIL = "empty output: nothing to judge"


@runtime_checkable
class Oracle(Protocol):
    """One strategy for turning recorded executions into verdicts.

    ``name`` is what the comparison table is keyed by, so it is part of the
    contract, not a label. ``evaluate`` takes a whole case group at once — not a
    single row — because metamorphic properties are only meaningful across the
    group, and because the hybrid arm's short-circuit decision is a property of the
    group as a whole.
    """

    name: str

    def evaluate(self, rows: list[ExecutionResult]) -> list[PropertyResult]: ...


def summary(results: Iterable[PropertyResult]) -> Verdict:
    """Collapse an arm's results to one verdict.

    A thin delegation to ``summarize``, kept here so the arms and their callers share
    one combination rule by construction. Two arms that disagreed about how to fold
    FAIL and INCONCLUSIVE together would not be comparable at all, which is the one
    thing this module cannot afford.
    """
    return summarize(results)


def validate_property_set(properties: Iterable[CaseProperty | GroupProperty]) -> None:
    """Reject a property set that cannot catch what its members decline to report.

    ``ValuesInUnitInterval`` and ``RowsSumToOne`` return INCONCLUSIVE on non-finite
    output so the defect is counted once, by ``OutputsAreFinite``. That deferral
    makes them *dependent*. A set of ``{values_in_unit_interval, rows_sum_to_one}``
    without ``outputs_are_finite`` is therefore structurally incapable of catching a
    NaN-producing kernel: every member defers, every member returns INCONCLUSIVE,
    and the declarative arm records a clean miss with no error anywhere. That is a
    silent understatement of one arm in the comparison the project is built to make,
    so it is rejected at construction time — before any evaluation runs and long
    before the numbers are read.

    Membership is by ``name``, and case and group properties share one name pool: a
    group property's deferral target is a case property (``ShiftInvariance`` defers
    its base side to ``outputs_are_finite``), so validating the two scopes separately
    would reject every legitimate set.

    An *empty* set is rejected for the same reason ``_require_rows`` rejects an empty
    case group, reached through a different door: it evaluates nothing, summarizes to
    INCONCLUSIVE, and adds a group that established nothing to the denominator of the
    detection rate with no error anywhere. This is live rather than theoretical —
    Task 13 builds sets from ``acceptance.yaml`` by name, so an omitted or misspelled
    ``properties:`` key produces exactly this.
    """
    instances = list(properties)
    if not instances:
        msg = (
            "empty property set: an oracle with no properties judges nothing and "
            "summarizes to INCONCLUSIVE, silently adding a group that established "
            "nothing to the denominator of the detection rate"
        )
        raise ValueError(msg)
    names = {prop.name for prop in instances}
    for prop in instances:
        target = prop.defers_nonfinite_to
        if target and target not in names:
            msg = (
                f"{prop.name!r} defers non-finite output to {target!r}, which is not in "
                f"the property set {sorted(names)}; without it a non-finite output is "
                f"reported by nobody and the arm silently records a miss"
            )
            raise ValueError(msg)


def _require_rows(arm: str, rows: Sequence[ExecutionResult]) -> None:
    """Reject an empty case group, uniformly, in every arm.

    This mirrors ``properties.py``'s split between bad *data* (INCONCLUSIVE, because
    the hardware time behind it cannot be recovered) and a bad *call* (raise, because
    it is free to fix and would otherwise contaminate the counts). An empty group is
    the latter: ``ExecutionTable.read_groups`` groups rows by ``group_id`` and so can
    never yield an empty list, and ``CaseGroup`` separately rejects a group with no
    base case. Reaching here means the evaluation layer built the list itself and got
    it wrong.

    Returning ``[]`` would be worse than raising twice over. It summarizes to
    INCONCLUSIVE, adding a group that judged nothing to the denominator of the
    detection rate with no trace back to a cause. And it would be *inconsistent*:
    ``ShiftInvariance.check_group([])`` raises, so a declarative arm that merely
    delegated would raise or stay silent depending on whether the configured property
    set happened to include a group property — a difference in error behaviour driven
    by a config file. Checking here makes all three arms agree.
    """
    if not rows:
        msg = f"oracle {arm!r} was handed an empty case group; nothing to judge"
        raise ValueError(msg)


class ReferenceOracle:
    """One property: the output matches a trusted recomputation of it.

    The strongest arm, and the brittlest. It needs three things the declarative arm
    does not — a reference implementation, a threshold, and an output dtype that has
    a unit roundoff — and where any of them is missing it must say INCONCLUSIVE
    rather than guess. Each such path is a *reduction in this arm's measured power*,
    which is exactly what the comparison is trying to quantify, so none of them may
    be quietly rounded to PASS or FAIL.

    ``reference_fn`` is called as ``reference_fn(**kernel_inputs(case))``, so it is
    written with the kernel's own parameter names and never sees generator
    bookkeeping tensors.

    A defect in the reference is a defect in the *harness*, and this arm distinguishes
    two kinds of it, because they need opposite handling.

    An exception propagates. It is unambiguous, it cannot be confused with a finding
    about the kernel, and swallowing it would let a wholly broken reference reduce
    this arm to a silent no-op across an entire run.

    A reference that returns *unusable data* — NaN, an overflowed infinity — does not
    raise, and left unchecked it is far more dangerous than the exception.
    ``residual_ratio`` answers ``inf``, which is a FAIL, so a perfectly correct kernel
    is booked as a caught bug and the detail string names the kernel: a false positive
    in the headline metric, manufactured by the harness. The reference's output is
    therefore validated before it is compared, and a non-finite one is INCONCLUSIVE
    with a detail that says the reference is at fault. Note the asymmetry with a
    non-finite *kernel* output, which still FAILs — that one is evidence. A shape
    disagreement is a third case again and is handled differently; see ``_check``.

    The reference call is wrapped in ``np.errstate(all="ignore")`` for the same
    reason. This project sets ``filterwarnings = ["error"]``, which turns a numpy
    overflow inside the reference into a raised exception — so the same reference
    defect would abort under test config and silently FAIL under production config, in
    a project whose central claim is that arms are scored over identical executions.
    Aborting is also the wrong failure mode for an offline scoring pass: the pass is
    cheap to re-run, but it dies partway through and discards the *already paid for*
    hardware time in the rest of the table. Suppressed here and caught by the
    finiteness check below, an overflowing reference costs one INCONCLUSIVE row.

    ``n`` overrides the accumulation length the test ratio normalizes by; see
    ``_accumulation_length``.
    """

    name = "reference"

    def __init__(
        self,
        reference_fn: Callable[..., np.ndarray],
        thresh: float = DEFAULT_THRESH,
        n: int | None = None,
    ) -> None:
        self.reference_fn = reference_fn
        self.thresh = thresh
        self.n = n

    def evaluate(self, rows: list[ExecutionResult]) -> list[PropertyResult]:
        _require_rows(self.name, rows)
        return [self._check(row) for row in rows]

    def _result(self, verdict: Verdict, detail: str, case_id: str) -> PropertyResult:
        """Every result this arm emits, built in one place.

        Centralized so ``case_id`` cannot be forgotten on one branch out of five.
        The reference arm is per-row, so it always attributes to the case; there is
        no group-scoped path here to get wrong.
        """
        return PropertyResult(
            property_name=REFERENCE_PROPERTY,
            tier=TIER_PORTABLE,
            # A recomputation is compared through a numerical threshold, so this arm
            # is definitionally not tolerance-free. That is the whole contrast the
            # headline claim draws against the structural declarative properties.
            tolerance_free=False,
            verdict=verdict,
            detail=detail,
            case_id=case_id,
        )

    def _length(self, row: ExecutionResult) -> int:
        """The accumulation length the test ratio normalizes by.

        Taken from the *input*, not the output. The output's last axis is only the
        accumulation length for a full-shape elementwise kernel; for anything that
        reduces, it is the shape the error was reduced *into*, which is exactly the
        trap ``RowsSumToOne`` documents one module over. Measured on correct float32
        reductions, normalizing by the output's last axis is 18x too strict for a
        row-wise reduction of shape (2, 262144) -> (2,) and 12x too strict for a full
        512x4096 reduction -> (1,), because ``log2(2)`` and ``log2(1)`` replace
        ``log2(262144)`` and ``log2(4096)``. Over-normalizing is not a safety margin:
        it raises the pass/fail line against a correct kernel until the arm books it
        as a caught bug.

        ``Case.shape`` describes the primary input ``x``, so its last axis is the
        reduction length for this corpus's row-wise kernels — the same restriction
        ``residual_ratio``'s own default carries, moved to the side of the comparison
        where it actually holds. A contraction length that is not an input dimension
        either (GEMM's K) is not derivable here at all; that is what ``n`` on the
        constructor is for.

        A scalar input has no last axis and a zero-length one is not a valid
        accumulation, so both fall back to 1, which ``residual_ratio`` floors to a
        divisor of 1.0 — no length normalization at all. That is the conservative
        direction: it cannot manufacture a pass, and it keeps a degenerate shape from
        reaching ``_validate_n``, which rejects ``n < 1`` with a bare ValueError that
        would abort the whole scoring pass over one row.
        """
        if self.n is not None:
            return self.n
        shape = row.case.shape
        return shape[-1] if shape and shape[-1] >= 1 else 1

    def _check(self, row: ExecutionResult) -> PropertyResult:
        case_id = row.case.case_id
        # Status and output presence are independent conditions; a Phase 3 timeout
        # can leave a partially written buffer behind, so both are checked.
        if row.status != Status.OK:
            return self._result(Verdict.INCONCLUSIVE, f"status={row.status!r}", case_id)
        if OUTPUT_NAME not in row.outputs:
            detail = f"missing output {OUTPUT_NAME!r}"
            return self._result(Verdict.INCONCLUSIVE, detail, case_id)

        # atleast_1d for the same reason residual_ratio applies it: a 0-d output would
        # make `.shape[-1]` an IndexError. `single_output` normalizes this at the
        # execution boundary, so it should be unreachable — but so is a zero-size
        # output, which is guarded two lines down, and this is the same class of
        # defensiveness at the same cost.
        got = np.atleast_1d(row.outputs[OUTPUT_NAME])
        if got.size == 0:
            # Agreement over zero elements is vacuously true, not evidence. It also
            # cannot be measured: residual_ratio answers NaN, which is a non-pass and
            # so a FAIL if taken at face value.
            return self._result(Verdict.INCONCLUSIVE, _EMPTY_DETAIL, case_id)

        # See the class docstring: suppressed so that an overflowing reference is one
        # INCONCLUSIVE row rather than a run-aborting exception under test config and
        # a silent FAIL under production config.
        with np.errstate(all="ignore"):
            expected = np.atleast_1d(np.asarray(self.reference_fn(**kernel_inputs(row.case))))

        # Validate the reference before comparing against it. residual_ratio answers
        # inf for a non-finite reference, and inf is a FAIL — so an unchecked reference
        # defect books a *correct* kernel as a caught bug, with a detail naming the
        # kernel. That is a false positive manufactured by the harness, in the one
        # number this project reports. Non-finite is unambiguously the reference's
        # fault: it is computed from a recorded input by trusted code, so NaN or an
        # overflowed infinity there is never evidence about the kernel.
        if expected.dtype.kind in "fc" and not np.all(np.isfinite(expected)):
            detail = "reference output is non-finite; the reference, not the kernel"
            return self._result(Verdict.INCONCLUSIVE, detail, case_id)

        # A shape disagreement is deliberately NOT treated the same way, because it is
        # symmetric: nothing here knows the correct output shape independently, so
        # "reference is wrong" and "kernel returned the wrong shape" are the same
        # observation. Calling it INCONCLUSIVE would permanently blind the strong arm
        # to wrong-shaped output, which is a common and serious real kernel bug and
        # exactly what this arm exists to catch. The two error sources are also very
        # differently detectable: a mis-written reference disagrees on *every* row, so
        # it shows up as a 100% failure rate including against the reference kernel
        # itself, whereas the non-finite case above is data-dependent and intermittent
        # (reachable on ~9.5% of groups at the default ShiftRows shift scale) and so
        # would hide inside a plausible-looking detection rate. FAIL is therefore the
        # right verdict; what the harness owes the reader is a detail that names both
        # shapes and does not pretend to know which side is wrong.
        if expected.shape != got.shape:
            detail = (
                f"shape mismatch: output {got.shape} vs reference {expected.shape} "
                f"(one of the two is wrong; this arm cannot tell which)"
            )
            return self._result(Verdict.FAIL, detail, case_id)

        try:
            # dtype= comes from the recorded output, not the reference: the reference
            # may be computed at higher precision, and the rounding budget belongs to
            # the dtype the kernel actually produced.
            ratio = residual_ratio(
                got, expected, dtype=got.dtype, n=self._length(row)
            )
        except ExactDtypeError:
            # PERSISTABLE_KINDS is "biuf", so int and bool outputs genuinely arrive
            # here. A test ratio is undefined for them: FAIL would book a possibly
            # correct integer kernel as a caught bug — a false positive injected
            # straight into the headline metric — and letting the exception escape
            # would abort a run whose hardware time cannot be recovered.
            return self._result(Verdict.INCONCLUSIVE, _EXACT_DTYPE_DETAIL, case_id)

        if not within_threshold(ratio, self.thresh):
            detail = f"reference test ratio {ratio:.3g} >= {self.thresh}"
            return self._result(Verdict.FAIL, detail, case_id)
        return self._result(Verdict.PASS, f"ratio={ratio:.3g}", case_id)


class DeclarativeOracle:
    """Many individually weak laws, evaluated at their natural scopes.

    Case properties judge one row each; group properties judge the whole group once,
    because a metamorphic relation is a statement about a base and its partner
    together and has no meaning applied to either alone.

    The property set is validated at construction rather than at first use, so a
    misconfigured set fails before any evaluation runs — the failure mode it guards
    against is silent, and a silent miss discovered after the numbers are reported is
    not discovered at all.
    """

    name = "declarative"

    def __init__(
        self,
        case_properties: Iterable[CaseProperty],
        group_properties: Iterable[GroupProperty] = (),
    ) -> None:
        self.case_properties = tuple(case_properties)
        self.group_properties = tuple(group_properties)
        validate_property_set([*self.case_properties, *self.group_properties])

    def evaluate(self, rows: list[ExecutionResult]) -> list[PropertyResult]:
        _require_rows(self.name, rows)
        results = [prop.check(row) for row in rows for prop in self.case_properties]
        results.extend(prop.check_group(rows) for prop in self.group_properties)
        return results


class HybridOracle:
    """Laws first as a cheap filter, the reference recompute only if they find nothing.

    The precedence is the point. A reference recompute costs a full evaluation of the
    kernel's semantics in the harness; the declarative laws are elementwise scans and
    one reduction. So when the laws already reached FAIL, the reference arm can add
    nothing to the verdict and is skipped — that saving is the reason this arm exists
    and is a measured result, not an implementation detail.

    INCONCLUSIVE does *not* short-circuit. It means the laws established nothing at
    all, which is precisely when the reference arm is the only thing that could reach
    a verdict; skipping there would throw away detections rather than cost.

    The saving is real but *conditional*, and the condition belongs next to the
    number. Measured, the reference recompute is the dominant half of oracle time and
    its share grows with size: 33% at (8, 8), 61% at (64, 256), 75% at (512, 4096). But
    it is only skipped on groups the declarative arm already failed, so the run-level
    saving is (declarative FAIL rate) x that share — and in a false-positive study,
    which scores *correct* kernels, the declarative arm never fails and the saving is
    exactly zero. A cost-per-bug figure quoted without that conditioning overstates
    the arm on precisely the workload where it does nothing.

    KNOWN BIAS, deliberately accepted. Because of the short-circuit, this arm's
    reference-arm coverage is *conditional*: on a declarative-FAIL group it never
    learns what the reference arm would have said. For the headline detection-rate
    table that is harmless — the group was caught either way — but any per-property
    attribution analysis, and any cost-per-bug figure, must exclude the groups where
    the reference arm never ran, or it will compare a full-coverage arm against a
    partial one. Nothing extra is recorded for this, because nothing needs to be: a
    ``matches_reference`` result is present exactly when the reference arm was
    consulted, so the conditioning is recoverable from the result list itself. A
    dedicated flag would be a second source of truth for the same fact, and could
    disagree with it.
    """

    name = "hybrid"

    def __init__(self, declarative: DeclarativeOracle, reference: ReferenceOracle) -> None:
        self.declarative = declarative
        self.reference = reference

    def evaluate(self, rows: list[ExecutionResult]) -> list[PropertyResult]:
        _require_rows(self.name, rows)
        results = self.declarative.evaluate(rows)
        if summary(results) is Verdict.FAIL:
            return results
        return [*results, *self.reference.evaluate(rows)]
