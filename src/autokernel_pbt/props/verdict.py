"""Three-valued verdicts and per-property attribution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    """Execution verdict: PASS, FAIL, or INCONCLUSIVE.

    A str subclass so json.dumps and Parquet persistence store the wire value
    ("pass", "fail", "inconclusive") rather than raising TypeError or storing
    the enum name. Without this, Phase 2's per-property attribution in the
    execution table would require custom serialization.
    """

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"

    # Without these, `str()` and `format()` of a str-mixin Enum return the wire
    # value on 3.10/3.11 but the repr-style name ("Verdict.PASS") on 3.12+.
    # This ensures `f"{verdict}"` and `str(verdict)` behave consistently across
    # the supported range (requires-python = ">=3.10").
    __str__ = str.__str__
    __format__ = str.__format__


# Tier vocabulary: the two tiers that properties use in Phase 1
TIER_PORTABLE = 1   # pure functions of inputs+outputs
TIER_BACKEND = 2    # needs execution telemetry
VALID_TIERS = frozenset({TIER_PORTABLE, TIER_BACKEND})


@dataclass(frozen=True)
class PropertyResult:
    """One property's verdict on one case or group.

    ``tier`` and ``tolerance_free`` are recorded per result so detection can be
    reported split by property tier and by whether a tolerance argument was needed.
    This per-property attribution is essential for the headline claim:
    "bugs found without a tolerance argument."

    The caller (Task 10 and 11) populates ``case_id`` and ``group_id`` during
    evaluation to establish which input was judged. This association ensures
    later phases can recover what each result judged without re-evaluating.
    """

    property_name: str
    tier: int
    tolerance_free: bool
    verdict: Verdict
    detail: str = ""
    case_id: str = ""   # set by case properties; "" for group properties
    group_id: str = ""  # set by group properties; "" for case properties

    def __post_init__(self) -> None:
        # Validate tier is in the set of valid tiers
        if self.tier not in VALID_TIERS:
            msg = f"tier must be in {sorted(VALID_TIERS)}, got {self.tier}"
            raise ValueError(msg)


def summarize(results: Iterable[PropertyResult]) -> Verdict:
    """Combine multiple property verdicts into a single test outcome.

    Precedence (strict order):
    1. FAIL dominates: any single property failure fails the whole test.
    2. Empty or wholly INCONCLUSIVE sets are INCONCLUSIVE, never PASS.
       This is mandatory for the false-positive metric: a property set that
       checked nothing has not established correctness.
    3. All other cases (only PASS verdicts) return PASS.

    Accepts any Iterable (including generators, which are not Sequences).
    """
    # Materialize to list to allow multiple iterations
    results_list = list(results)

    # FAIL dominates
    if any(r.verdict is Verdict.FAIL for r in results_list):
        return Verdict.FAIL

    # Empty or wholly INCONCLUSIVE is INCONCLUSIVE, never PASS
    if not results_list or any(r.verdict is Verdict.INCONCLUSIVE for r in results_list):
        return Verdict.INCONCLUSIVE

    return Verdict.PASS
