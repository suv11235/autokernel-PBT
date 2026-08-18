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


def admit(rows: list[ExecutionResult], *, reference_fn: Callable[..., Any]) -> bool | Rejection:
    """`True` if the candidate belongs in the corpus, else a `Rejection`.

    Two criteria, and no more:

    * **broken somewhere** -- it differs from the reference beyond tolerance on at
      least one case group;
    * **judgeable somewhere** -- at least one group ran to `Status.OK`, so the arms
      have something to judge rather than abstaining everywhere.

    Notably absent: any requirement that the candidate AGREE with the reference
    somewhere. A kernel wrong on every group is an ordinary bug that should score a
    detection rate of 1.0, and demanding agreement would reject valid mutants for
    being too easy to catch -- exactly backwards.
    """
    judgeable: set[str] = set()
    broken: set[str] = set()

    for row in rows:
        if row.status != Status.OK or OUTPUT_NAME not in row.outputs:
            continue
        judgeable.add(row.case.group_id)
        got = np.atleast_1d(row.outputs[OUTPUT_NAME])
        with np.errstate(all="ignore"):
            expected = np.atleast_1d(np.asarray(reference_fn(**kernel_inputs(row.case))))
        if got.shape != expected.shape:
            broken.add(row.case.group_id)
            continue
        try:
            ratio = residual_ratio(got, expected, dtype=got.dtype, n=got.shape[-1])
        except ExactDtypeError:
            # An exact-dtype output has no test ratio. Treated as unbroken here
            # rather than guessed: admitting it on a technicality would put an
            # unverified candidate into the denominator.
            continue
        if not np.isfinite(ratio) or ratio >= DEFAULT_THRESH:
            broken.add(row.case.group_id)

    if not judgeable:
        return Rejection(
            reason="not judgeable on any group: every case failed to produce an output",
            groups_broken=len(broken),
            groups_judgeable=0,
        )
    if not broken:
        return Rejection(
            reason="not broken on any group: it matches the reference within tolerance",
            groups_broken=0,
            groups_judgeable=len(judgeable),
        )
    return True
