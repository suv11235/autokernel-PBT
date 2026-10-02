"""The six checks a certificate must pass, and the certificate record.

A proof the language's checker accepts is necessary and nowhere near sufficient, because the agent
wrote everything the proof is about. Each check below closes one way of getting a proof accepted
without proving adherence, and each is paired, in the tests, with an attempt that cheats in exactly
that way and must be rejected by that check alone.

Agent-supplied readings are data, not harness code: one that raises, returns something other than
a boolean, or writes into its inputs fails its own check and never aborts the others.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from autokernel_pbt.formal.attempt import Attempt, ProofCheck, Readings
from autokernel_pbt.formal.comparisons import Comparisons
from autokernel_pbt.formal.ground import CaseRecord, Ground, realize

#: Samples each axiom's reading is checked on. An axiom false only on a set of measure zero slips
#: through; one false on any positive fraction of its domain is caught with overwhelming odds.
AXIOM_SAMPLES = 64

_NO_READING = "no executable reading, so it fails closed"


class Check(str, Enum):
    """The six checks, in the order they run."""

    PROOF = "proof"
    SPEC_SOUNDNESS = "spec_soundness"
    SPEC_COMPLETENESS = "spec_completeness"
    MODEL_FIDELITY = "model_fidelity"
    NON_VACUITY = "non_vacuity"
    AXIOMS = "axioms"

    # As Verdict and Status do: format() of a str-mixin Enum gives the value on 3.10/3.11 and the
    # name on 3.12+, against a declared requires-python >= 3.10.
    __str__ = str.__str__


@dataclass(frozen=True)
class Finding:
    check: Check
    passed: bool
    detail: str


@dataclass(frozen=True)
class Certificate:
    """What the six checks found for one attempt."""

    language: str
    kernel_id: str
    findings: tuple[Finding, ...]

    @property
    def failed(self) -> frozenset[Check]:
        return frozenset(f.check for f in self.findings if not f.passed)

    @property
    def certified(self) -> bool:
        # All six must have run, not merely none failed: a record missing a check is not a
        # certificate that passed it.
        return {f.check for f in self.findings} == set(Check) and not self.failed


def certify(attempt: Attempt, ground: Ground) -> Certificate:
    """Judge one attempt against its ground. Raises only if the ground cannot judge."""
    cases = realize(ground)
    findings = (
        _proof(attempt.proof),
        _spec_soundness(attempt.readings, cases),
        _spec_completeness(attempt.readings, cases),
        _model_fidelity(attempt.readings, cases),
        _non_vacuity(attempt, cases),
        _axioms(attempt, ground.primitives),
    )
    return Certificate(attempt.language, attempt.kernel_id, findings)


def _holds(reading: Callable[..., object], *args: object) -> tuple[bool | None, str]:
    """Run a reading that should answer yes or no. ``None`` means it did neither."""
    try:
        answer = reading(*args)
    except Exception as exc:  # noqa: BLE001 - agent code failing is a finding, not an abort
        return None, f"raised {type(exc).__name__}: {exc}"
    if not isinstance(answer, (bool, np.bool_)):
        return None, f"returned {type(answer).__name__}, not a bool"
    return bool(answer), ""


def _proof(proof: ProofCheck) -> Finding:
    if not proof.accepted:
        return Finding(Check.PROOF, False, "the language's checker rejected the proof")
    if proof.escape_hatches:
        hatches = sorted(proof.escape_hatches)
        return Finding(Check.PROOF, False, f"escape hatch(es) in the proof: {hatches}")
    return Finding(Check.PROOF, True, "accepted by the checker, with no escape hatch")


def _spec_soundness(readings: Readings | None, cases: tuple[CaseRecord, ...]) -> Finding:
    if readings is None:
        return Finding(Check.SPEC_SOUNDNESS, False, f"the spec has {_NO_READING}")
    rejected = []
    for case in cases:
        cmp = Comparisons.for_output(case.inputs, case.reference.dtype)
        holds, why = _holds(readings.spec, case.inputs, case.reference, cmp)
        if holds is None:
            return Finding(Check.SPEC_SOUNDNESS, False, f"spec reading {why} on {case.case_id}")
        if not holds:
            rejected.append(case.case_id)
    if rejected:
        detail = (
            f"spec rejects the reference on {len(rejected)} of {len(cases)} cases, "
            f"first {rejected[0]}"
        )
        return Finding(Check.SPEC_SOUNDNESS, False, detail)
    detail = f"spec accepts the reference on all {len(cases)} cases"
    return Finding(Check.SPEC_SOUNDNESS, True, detail)


def _spec_completeness(readings: Readings | None, cases: tuple[CaseRecord, ...]) -> Finding:
    if readings is None:
        return Finding(Check.SPEC_COMPLETENESS, False, f"the spec has {_NO_READING}")
    accepted: dict[str, list[str]] = {}
    judged = 0
    for case in cases:
        for name, output in case.hidden.items():
            cmp = Comparisons.for_output(case.inputs, output.dtype)
            if cmp.close(output, case.reference):
                # Right here — a one-column softmax is 1.0 at any temperature — so there is
                # nothing for the spec to reject on this case.
                continue
            judged += 1
            holds, why = _holds(readings.spec, case.inputs, output, cmp)
            if holds is None:
                detail = f"spec reading {why} on {case.case_id} ({name})"
                return Finding(Check.SPEC_COMPLETENESS, False, detail)
            if holds:
                accepted.setdefault(name, []).append(case.case_id)
    if accepted:
        which = ", ".join(f"{name} on {len(ids)}" for name, ids in sorted(accepted.items()))
        detail = f"spec accepts wrong output from {which} case(s)"
        return Finding(Check.SPEC_COMPLETENESS, False, detail)
    detail = f"spec rejects every hidden kernel on all {judged} cases where it is wrong"
    return Finding(Check.SPEC_COMPLETENESS, True, detail)


def _model_fidelity(readings: Readings | None, cases: tuple[CaseRecord, ...]) -> Finding:
    if readings is None:
        return Finding(Check.MODEL_FIDELITY, False, f"the model has {_NO_READING}")
    diverged = []
    for case in cases:
        if case.kernel is None:
            detail = (
                f"the kernel under proof failed on {case.case_id}; there is nothing to be "
                f"faithful to"
            )
            return Finding(Check.MODEL_FIDELITY, False, detail)
        try:
            output = np.asarray(readings.model(case.inputs))
            cmp = Comparisons.for_output(case.inputs, case.kernel.dtype)
            same = cmp.close(output, case.kernel)
        except Exception as exc:  # noqa: BLE001 - agent code failing is a finding, not an abort
            detail = f"model reading raised {type(exc).__name__}: {exc} on {case.case_id}"
            return Finding(Check.MODEL_FIDELITY, False, detail)
        if not same:
            diverged.append(case.case_id)
    if diverged:
        detail = (
            f"model diverges from the kernel on {len(diverged)} of {len(cases)} cases, "
            f"first {diverged[0]}"
        )
        return Finding(Check.MODEL_FIDELITY, False, detail)
    detail = f"model matches the kernel on all {len(cases)} cases"
    return Finding(Check.MODEL_FIDELITY, True, detail)


def _non_vacuity(attempt: Attempt, cases: tuple[CaseRecord, ...]) -> Finding:
    if not attempt.proof.statement_has_shape:
        detail = (
            "the theorem is not of the required shape: for all inputs, hypotheses imply "
            "spec(inputs, model(inputs))"
        )
        return Finding(Check.NON_VACUITY, False, detail)
    if attempt.readings is None:
        return Finding(Check.NON_VACUITY, False, f"the hypotheses have {_NO_READING}")
    excluded = []
    for case in cases:
        holds, why = _holds(attempt.readings.hypotheses, case.inputs)
        if holds is None:
            return Finding(Check.NON_VACUITY, False, f"hypotheses reading {why} on {case.case_id}")
        if not holds:
            excluded.append(case.case_id)
    if excluded:
        detail = (
            f"hypotheses exclude {len(excluded)} of {len(cases)} cases, first {excluded[0]}: "
            f"the theorem says nothing about them"
        )
        return Finding(Check.NON_VACUITY, False, detail)
    return Finding(Check.NON_VACUITY, True, f"hypotheses hold on all {len(cases)} cases")


def _axioms(attempt: Attempt, primitives: frozenset[str]) -> Finding:
    if attempt.proof.derives_false:
        detail = "False is derivable from the axioms within the bounded attempt"
        return Finding(Check.AXIOMS, False, detail)
    rng = np.random.default_rng(0)
    for axiom in attempt.axioms:
        beyond = axiom.mentions - primitives
        if beyond:
            detail = (
                f"axiom {axiom.name!r} mentions {sorted(beyond)}, beyond the declared "
                f"primitives {sorted(primitives)}"
            )
            return Finding(Check.AXIOMS, False, detail)
        if axiom.holds_on is None:
            return Finding(Check.AXIOMS, False, f"axiom {axiom.name!r} has {_NO_READING}")
        for _ in range(AXIOM_SAMPLES):
            holds, why = _holds(axiom.holds_on, rng)
            if holds is None:
                return Finding(Check.AXIOMS, False, f"axiom {axiom.name!r} reading {why}")
            if not holds:
                return Finding(Check.AXIOMS, False, f"axiom {axiom.name!r} is false on a sample")
    detail = (
        f"{len(attempt.axioms)} axiom(s), each about declared primitives only and true on "
        f"{AXIOM_SAMPLES} samples"
    )
    return Finding(Check.AXIOMS, True, detail)
