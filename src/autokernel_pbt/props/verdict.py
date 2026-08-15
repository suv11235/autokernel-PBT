"""Three-valued verdicts and per-property attribution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    """Execution verdict: PASS, FAIL, or INCONCLUSIVE.

    A str subclass to prepare for persistence to Parquet/JSON in Phase 2,
    where per-property attribution will be stored in the execution table.
    Without the str mixin, rendering behavior differs across Python 3.10+
    versions (3.10/3.11 render the value, 3.12+ renders the name).
    Matching Status.py's pattern ensures consistent string representation.
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


# Tier values that properties and results use. Exactly two tiers in Phase 1:
# 1 = portable/semantic, pure functions of inputs+outputs
# 2 = backend-specific, needs execution telemetry
VALID_TIERS = frozenset({1, 2})


@dataclass(frozen=True)
class PropertyResult:
    """One property's verdict on one case or group.

    ``tier`` and ``tolerance_free`` are recorded per result so detection can be
    reported split by property tier and by whether a tolerance argument was needed.
    This per-property attribution is essential for the headline claim:
    "bugs found without a tolerance argument."
    """

    property_name: str
    tier: int
    tolerance_free: bool
    verdict: Verdict
    detail: str = ""

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

    Accepts any Iterable (including generators, which are not Sequences);
    converts to list internally to allow multiple iterations.
    """
    # Convert to list to allow multiple iterations (needed for two any() calls)
    results_list = list(results) if not isinstance(results, list) else results

    # FAIL dominates
    if any(r.verdict is Verdict.FAIL for r in results_list):
        return Verdict.FAIL

    # Empty or wholly INCONCLUSIVE is INCONCLUSIVE, never PASS
    if not results_list or any(r.verdict is Verdict.INCONCLUSIVE for r in results_list):
        return Verdict.INCONCLUSIVE

    return Verdict.PASS
