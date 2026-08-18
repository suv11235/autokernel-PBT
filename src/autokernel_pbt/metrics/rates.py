"""The metrics, computed from the recorded tables alone.

No oracle, no kernel, no generator is in the loop here. That is the whole point: a
recorded run is a reusable dataset only if a rate can be derived from it months later
on a machine with none of those things available.

THE UNIT IS THE CASE GROUP, not the result. This is settled and measured rather than
assumed: per-result and per-group rates differ 0.222 against 0.778 for the same 14
detections, because arms emit different numbers of results per group -- the reference
arm one per recorded case, the declarative arm one per case per case-property plus one
per group-property. A per-result rate is therefore a weighted average whose weights
belong to the arm rather than to the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autokernel_pbt.props.scores import ArmScores
from autokernel_pbt.props.verdict import PropertyResult, Verdict, summarize


@dataclass(frozen=True)
class ArmRates:
    """One arm's rates over one recorded run."""

    arm: str
    groups_scored: int
    groups_failed: int
    groups_inconclusive: int
    detection_rate: float
    #: Groups failed by at least one property that needed no tolerance, over groups
    #: scored. The project's sharpest claim -- "bugs found without a tolerance
    #: argument" -- and it CANNOT be inferred from `detection_rate`: a group failed
    #: only by a tolerance-bearing property must not count toward it.
    tolerance_free_detection_rate: float
    #: Index of the first failing group in generation order, or None. Generation order
    #: is a pure function of (seed, index), so this is reproducible.
    cases_to_first_failure: int | None


def _by_group(results: list[PropertyResult]) -> dict[str, list[PropertyResult]]:
    groups: dict[str, list[PropertyResult]] = {}
    for result in results:
        # Every persisted score row carries a group_id; `ScoreTable` refuses one
        # without. Reading it directly rather than falling back keeps a malformed
        # table loud instead of silently mis-binned.
        groups.setdefault(result.group_id, []).append(result)
    return groups


def arm_rates(arm: ArmScores) -> ArmRates:
    """Rates for one arm. Pure; touches nothing but the results it is handed."""
    groups = _by_group(arm.results)
    order = list(groups)
    verdicts = {gid: summarize(rs) for gid, rs in groups.items()}

    failed = [gid for gid in order if verdicts[gid] is Verdict.FAIL]
    inconclusive = sum(1 for gid in order if verdicts[gid] is Verdict.INCONCLUSIVE)
    tolerance_free = sum(
        1
        for gid in failed
        if any(r.verdict is Verdict.FAIL and r.tolerance_free for r in groups[gid])
    )
    n = len(order)
    return ArmRates(
        arm=arm.arm,
        groups_scored=n,
        groups_failed=len(failed),
        groups_inconclusive=inconclusive,
        detection_rate=len(failed) / n if n else 0.0,
        tolerance_free_detection_rate=tolerance_free / n if n else 0.0,
        cases_to_first_failure=order.index(failed[0]) if failed else None,
    )


def rates_from_run(run_dir: Path | str) -> dict[str, ArmRates]:
    """Every arm's rates for a recorded run, arm name -> rates.

    Goes through `driver.read_run`, which refuses a scores/rows pair that is not
    about the same corpus. Reading `scores.parquet` directly would happily compute a
    rate from another run's verdicts, since case ids are a pure function of
    (seed, index) and would join perfectly.
    """
    from autokernel_pbt.props.driver import read_run

    _, arms = read_run(run_dir)
    return {arm.arm: arm_rates(arm) for arm in arms}
