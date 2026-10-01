"""The admission gate for agent-authored mutants.

An agent asked for a specific fault class returns something plausible; whether it is
actually that fault, actually broken, or actually runnable is not guaranteed. Each
way it can be wrong corrupts a different number, and none of them announces itself:

* a mutant that is secretly **correct** enters the detection denominator as a bug
  nobody can catch, lowering every arm's rate for free -- the most dangerous of the
  three, because nothing downstream looks wrong;
* one broken in a **different class** than intended corrupts per-class rates while
  the total stays plausible;
* one broken **catastrophically** makes every arm INCONCLUSIVE, at which point the
  driver refuses the run and nothing is recorded at all.

The gate addresses the first and third. It deliberately does NOT address the second;
see `Mutant.intended_class`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from autokernel_pbt.props.backends.base import (
    OUTPUT_NAME,
    ExecutionResult,
    Status,
    kernel_inputs,
)
from autokernel_pbt.props.tolerance import DEFAULT_THRESH, ExactDtypeError, residual_ratio


@dataclass(frozen=True)
class Rejection:
    """Why a candidate was refused, kept rather than discarded.

    The rejection rate is a finding: it says what proportion of an agent's attempts
    at a named fault class are not that fault, which is a fact about code-generating
    models and costs nothing extra to collect.
    """

    reason: str
    groups_broken: int
    groups_judgeable: int
    #: Groups that produced an output the gate could not compare -- an exact output
    #: dtype, or a non-finite reference -- and that no comparable row covered. Kept
    #: apart from ``groups_judgeable`` so "never compared" cannot read as "matched".
    groups_unverifiable: int = 0


#: Why a group that ran could still not be compared. Fixed phrases, because the
#: rejection reason is tallied as a finding and a reason assembled from free text
#: cannot be counted.
_EXACT_DTYPE = "the output dtype is exact, so no test ratio is defined for it"
_NONFINITE_REFERENCE = (
    "the reference output is non-finite, a defect of the reference and not of the candidate"
)


def _accumulation_length(row: ExecutionResult) -> int:
    """The reduction length the test ratio normalizes by: the INPUT's last axis.

    The same rule as ``ReferenceOracle._length``, because the gate is only honest if
    it agrees with the arm it admits candidates for. The output's last axis is the
    shape the error was reduced *into*: for a (2, 1024) -> (2,) row sum it is 2, which
    makes the threshold 10x stricter than the arm's and admits as "broken" a kernel
    the arm calls correct. A scalar or zero-length input falls back to 1, which
    ``residual_ratio`` floors to no length normalization at all.
    """
    shape = row.case.shape
    return shape[-1] if shape and shape[-1] >= 1 else 1


def admit(rows: list[ExecutionResult], *, reference_fn: Callable[..., Any]) -> bool | Rejection:
    """`True` if the candidate belongs in the corpus, else a `Rejection`.

    Two criteria, and no more:

    * **broken somewhere** -- it differs from the reference beyond tolerance on at
      least one case group;
    * **judgeable somewhere** -- at least one group ran to `Status.OK` AND could be
      compared, so the arms have something to judge rather than abstaining everywhere.

    Notably absent: any requirement that the candidate AGREE with the reference
    somewhere. A kernel wrong on every group is an ordinary bug that should score a
    detection rate of 1.0, and demanding agreement would reject valid mutants for
    being too easy to catch -- exactly backwards.

    A group can run and still not be comparable, for either of the reasons the
    reference arm abstains on: an exact output dtype has no test ratio, and a
    non-finite reference output is a harness defect, never evidence about the
    candidate. Such a group is UNVERIFIABLE -- neither judgeable nor broken -- and is
    counted as such, so a rejection can never say "matches the reference" about a
    comparison it did not make. A wrong output shape is checked before either: it is
    evidence that needs no test ratio, as it is for the reference arm.
    """
    judgeable: set[str] = set()
    broken: set[str] = set()
    unverifiable: dict[str, str] = {}

    for row in rows:
        if row.status != Status.OK or OUTPUT_NAME not in row.outputs:
            continue
        group_id = row.case.group_id
        got = np.atleast_1d(row.outputs[OUTPUT_NAME])
        with np.errstate(all="ignore"):
            expected = np.atleast_1d(np.asarray(reference_fn(**kernel_inputs(row.case))))
        if expected.dtype.kind in "fc" and not np.all(np.isfinite(expected)):
            unverifiable.setdefault(group_id, _NONFINITE_REFERENCE)
            continue
        if got.shape != expected.shape:
            judgeable.add(group_id)
            broken.add(group_id)
            continue
        try:
            ratio = residual_ratio(
                got, expected, dtype=got.dtype, n=_accumulation_length(row)
            )
        except ExactDtypeError:
            unverifiable.setdefault(group_id, _EXACT_DTYPE)
            continue
        judgeable.add(group_id)
        if not np.isfinite(ratio) or ratio >= DEFAULT_THRESH:
            broken.add(group_id)

    # A group with one comparable row is judged by it; only groups no row could
    # cover count as unverifiable.
    uncovered = {gid: why for gid, why in unverifiable.items() if gid not in judgeable}

    if not judgeable:
        if uncovered:
            causes = [c for c in (_EXACT_DTYPE, _NONFINITE_REFERENCE) if c in uncovered.values()]
            return Rejection(
                reason="not verifiable on any group: " + "; and ".join(causes),
                groups_broken=0,
                groups_judgeable=0,
                groups_unverifiable=len(uncovered),
            )
        return Rejection(
            reason="not judgeable on any group: every case failed to produce an output",
            groups_broken=0,
            groups_judgeable=0,
        )
    if not broken:
        return Rejection(
            reason="not broken on any group: it matches the reference within tolerance",
            groups_broken=0,
            groups_judgeable=len(judgeable),
            groups_unverifiable=len(uncovered),
        )
    return True
