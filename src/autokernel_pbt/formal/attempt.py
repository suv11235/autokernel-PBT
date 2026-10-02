"""An attempt, as the harness sees it: what a language adapter extracted from the agent's work.

The agent writes a spec, a model of the kernel, hypotheses, a theorem and a proof in some formal
language, and none of it is Python. An adapter runs the language's checker and supplies the
*executable readings* — the same definitions, evaluated on concrete inputs — through which the
language-neutral checks test everything the proof takes for granted. An ``Attempt`` is everything
those checks may read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from autokernel_pbt.formal.comparisons import Comparisons

Inputs = Mapping[str, np.ndarray]

#: Does this output meet the spec for these inputs? Equality goes through the harness's ``cmp``.
SpecReading = Callable[[Inputs, np.ndarray, Comparisons], bool]
#: What the agent says the kernel computes.
ModelReading = Callable[[Inputs], np.ndarray]
#: Does the theorem claim anything about these inputs?
HypothesesReading = Callable[[Inputs], bool]


@dataclass(frozen=True)
class Readings:
    """The executable reading of an attempt's spec, model and hypotheses.

    They must come from the same definitions the proof is about — never a twin the agent wrote
    separately, which would let it test one thing and prove another. Guaranteeing that is the
    adapter's obligation (feature 0010); here a reading is only its shape.
    """

    spec: SpecReading
    model: ModelReading
    hypotheses: HypothesesReading


@dataclass(frozen=True)
class Axiom:
    """An assumption the proof rests on beyond the language's base library.

    ``mentions`` is every identifier the axiom's statement refers to, as the adapter parsed it.
    ``holds_on`` draws a sample, evaluates the axiom under its intended reading, and says whether
    it held. ``None`` means the axiom has no reading, and an axiom nobody can test fails closed.
    """

    name: str
    mentions: frozenset[str]
    holds_on: Callable[[np.random.Generator], bool] | None


@dataclass(frozen=True)
class ProofCheck:
    """What the language's own tooling found, as the adapter reports it.

    ``statement_has_shape`` is the one thing the harness fixes about the theorem: for all inputs,
    hypotheses imply spec(inputs, model(inputs)). Without it the checks could not find the spec,
    the model and the hypotheses inside the statement.
    """

    accepted: bool
    escape_hatches: tuple[str, ...]
    statement_has_shape: bool
    derives_false: bool


@dataclass(frozen=True)
class Attempt:
    """One agent attempt at one kernel in one language. No readings fails closed."""

    language: str
    kernel_id: str
    readings: Readings | None
    axioms: tuple[Axiom, ...]
    proof: ProofCheck
