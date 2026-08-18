"""Metric definitions, computed from artifacts alone."""

from __future__ import annotations

import numpy as np

from autokernel_pbt.metrics.rates import arm_rates, rates_from_run
from autokernel_pbt.props.scores import ArmScores
from autokernel_pbt.props.verdict import PropertyResult, Verdict


def _r(prop, verdict, group, tolerance_free=False):
    return PropertyResult(prop, 1, tolerance_free, verdict, group_id=group)


def test_detection_rate_counts_groups_not_results():
    """The criterion DETECTION_IS_KEYED_BY_GROUP.

    Measured on this corpus: per-result and per-group rates differ 0.222 against
    0.778 for the same 14 detections, because arms emit different numbers of results
    per group. A per-result rate is a weighted average whose weights belong to the
    arm rather than to the kernel.
    """
    arm = ArmScores(arm="declarative", elapsed_s=0.0, results=[
        _r("a", Verdict.FAIL, "g0"), _r("b", Verdict.PASS, "g0"),
        _r("a", Verdict.PASS, "g1"), _r("b", Verdict.PASS, "g1"),
    ])
    rates = arm_rates(arm)
    assert rates.groups_scored == 2
    assert rates.groups_failed == 1
    assert rates.detection_rate == 0.5


def test_tolerance_free_detection_has_its_own_numerator():
    """The criterion TOLERANCE_FREE_DETECTION_IS_SEPARATE.

    "Bugs found without a tolerance argument" is the project's sharpest claim, so it
    cannot be inferred from the overall rate: a group failed only by a
    tolerance-bearing property must not count toward it.
    """
    arm = ArmScores(arm="declarative", elapsed_s=0.0, results=[
        _r("free", Verdict.FAIL, "g0", tolerance_free=True),
        _r("ratio", Verdict.FAIL, "g1", tolerance_free=False),
    ])
    rates = arm_rates(arm)
    assert rates.detection_rate == 1.0
    assert rates.tolerance_free_detection_rate == 0.5


def test_a_group_failed_only_by_a_tolerance_bearing_property_is_excluded():
    # The same group carrying both kinds of result must count once, and only because
    # of the tolerance-free one.
    arm = ArmScores(arm="declarative", elapsed_s=0.0, results=[
        _r("ratio", Verdict.FAIL, "g0", tolerance_free=False),
        _r("free", Verdict.PASS, "g0", tolerance_free=True),
    ])
    assert arm_rates(arm).tolerance_free_detection_rate == 0.0


def test_cases_to_first_failure_is_the_first_failing_group_index():
    arm = ArmScores(arm="reference", elapsed_s=0.0, results=[
        _r("p", Verdict.PASS, "g0"), _r("p", Verdict.PASS, "g1"),
        _r("p", Verdict.FAIL, "g2"),
    ])
    assert arm_rates(arm).cases_to_first_failure == 2


def test_cases_to_first_failure_is_none_when_nothing_failed():
    arm = ArmScores(arm="reference", elapsed_s=0.0, results=[_r("p", Verdict.PASS, "g0")])
    assert arm_rates(arm).cases_to_first_failure is None


def test_inconclusive_groups_are_not_counted_as_detections():
    # A group nobody could judge is not a caught bug. Counting it would inflate every
    # arm's rate by the crash rate, which is the quantity the metric isolates.
    arm = ArmScores(arm="reference", elapsed_s=0.0, results=[
        _r("p", Verdict.INCONCLUSIVE, "g0"), _r("p", Verdict.FAIL, "g1"),
    ])
    rates = arm_rates(arm)
    assert rates.groups_failed == 1
    assert rates.groups_inconclusive == 1
    assert rates.detection_rate == 0.5


def test_rates_are_computed_from_the_tables_alone(tmp_path, repo_root):
    """The criterion METRICS_COME_FROM_ARTIFACTS_ALONE.

    A recorded run is a reusable dataset only if a rate can be derived from it with
    no oracle, kernel or generator available. This drives a real run and then reads
    the rate back off the two Parquet files.
    """
    from autokernel_pbt.props.driver import run_task
    from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference

    def unnormalized(x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        return np.exp(shifted).astype(x.dtype)

    run_dir = tmp_path / "run"
    run_task(task=SOFTMAX, kernel=unnormalized, reference_fn=softmax_reference,
             run_dir=run_dir, repo_root=repo_root,
             n_groups=len(SOFTMAX.domain.shapes), seed=42,
             kernel_id="unnormalized", kernel_is_broken=True)

    table = rates_from_run(run_dir)
    assert set(table) == {"allclose", "reference", "declarative", "hybrid"}
    # The measured ladder deflation: 7 of 9, the other two being the single-column
    # rungs where an unnormalized softmax is genuinely correct.
    assert table["declarative"].detection_rate == 7 / 9
    assert table["declarative"].groups_scored == 9
