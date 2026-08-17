"""All four arms over one recorded run.

The comparison this project reports needs the field's default (``allclose``) scored
on the same executions as the strengthened reference ratio, plus the declarative set
and the hybrid composition. This module asserts the driver actually produces all
four, and two things about how they treat a kernel that did not run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autokernel_pbt.props.driver import ARM_NAMES, read_run, run_task
from autokernel_pbt.props.scores import ScoreTable
from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference
from autokernel_pbt.props.verdict import Verdict, summarize

pytestmark = pytest.mark.integration

SEED = 42

#: The whole ladder. Anything smaller makes ``Generator`` warn that a boundary shape
#: will never be exercised, and this project turns warnings into errors.
ALL_SHAPES = len(SOFTMAX.domain.shapes)

#: The shape the partially-crashing kernel below refuses to run on. Any ladder shape
#: works; this one is a non-degenerate rung, so the *other* groups still produce real
#: verdicts and the run is not refused for establishing nothing.
CRASH_SHAPE = (3, 7)


def correct_softmax(x: np.ndarray) -> np.ndarray:
    """Correct, and deliberately not bit-identical to ``softmax_reference``.

    It accumulates in the input's own float32 rather than widening to float64, so the
    tolerance-bearing arms actually consult their thresholds rather than being handed
    a residual of exactly zero.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(x.dtype)


def unnormalized_softmax(x: np.ndarray) -> np.ndarray:
    """Broken: forgets the division by the row sum.

    Genuinely *correct* on the two single-column rungs, where softmax is identically
    1.0 for any implementation — the known ladder deflation, not an oracle miss.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(shifted).astype(x.dtype)


def crashes_on_one_shape(x: np.ndarray) -> np.ndarray:
    """Raises on exactly one ladder rung, and is correct everywhere else.

    Deliberately *partial*. A kernel that crashed on every case would make every arm
    wholly INCONCLUSIVE, and the driver refuses that outright — an arm that
    establishes nothing anywhere is a wiring bug, not a result. Crashing on one rung
    keeps the run legitimate while still producing rows that no arm can judge, which
    is the state under test.
    """
    if x.shape == CRASH_SHAPE:
        msg = "kernel exploded"
        raise RuntimeError(msg)
    return correct_softmax(x)


def drive(run_dir: Path, kernel, kernel_id: str, repo_root: Path) -> Path:
    run_task(
        task=SOFTMAX,
        kernel=kernel,
        reference_fn=softmax_reference,
        run_dir=run_dir,
        repo_root=repo_root,
        n_groups=ALL_SHAPES,
        seed=SEED,
        kernel_id=kernel_id,
    )
    return run_dir


def test_four_arms_score_one_recorded_run(tmp_path: Path, repo_root: Path):
    """The criterion FOUR_ARMS_SCORE_ONE_RUN."""
    run_dir = drive(tmp_path / "run", correct_softmax, "correct_softmax", repo_root)
    assert {arm.arm for arm in ScoreTable(run_dir).read()} == set(ARM_NAMES)


def test_every_arm_catches_an_unnormalized_softmax(tmp_path: Path, repo_root: Path):
    # The comparison is only meaningful if each arm can fail; an arm that never
    # reaches FAIL on a plainly broken kernel is not measuring anything.
    run_dir = drive(tmp_path / "run", unnormalized_softmax, "unnormalized", repo_root)
    for arm in ScoreTable(run_dir).read():
        verdicts = [r.verdict for r in arm.results]
        assert Verdict.FAIL in verdicts, f"{arm.arm} did not catch an unnormalized softmax"


def test_allclose_and_reference_agree_on_a_correct_kernel(tmp_path: Path, repo_root: Path):
    """The comparison the fourth arm exists to enable.

    On a correct kernel the strengthened log2(n)-normalized ratio and the field's
    plain ``allclose`` must both pass. A disagreement here would mean the reference
    arm's calibration is off in one direction or the other, which is exactly the
    objection carrying this arm is meant to answer with evidence.
    """
    run_dir = drive(tmp_path / "run", correct_softmax, "correct_softmax", repo_root)
    by_arm = {arm.arm: arm for arm in ScoreTable(run_dir).read()}
    for name in ("allclose", "reference"):
        assert Verdict.FAIL not in {r.verdict for r in by_arm[name].results}, name


def test_launch_error_is_inconclusive_in_every_arm(tmp_path: Path, repo_root: Path):
    """The criterion LAUNCH_ERROR_IS_NEVER_A_CAUGHT_BUG.

    A kernel that did not run is not a detection. Booking a crash as FAIL would
    inflate every arm's rate by the crash rate, which is precisely the quantity the
    metric is supposed to isolate.

    The kernel crashes on one rung only, so the other groups still establish
    verdicts and the run is legitimate; the assertion is about the crashed group.
    """
    run_dir = drive(tmp_path / "run", crashes_on_one_shape, "crashes", repo_root)
    rows, arms = read_run(run_dir)

    crashed = {row.case.group_id for row in rows if row.case.shape == CRASH_SHAPE}
    assert crashed, "no group used the crash shape, so this test would pass vacuously"

    for arm in arms:
        by_group: dict[str, list] = {}
        for result in arm.results:
            by_group.setdefault(result.group_id, []).append(result)
        for group_id in crashed:
            verdict = summarize(by_group[group_id])
            assert verdict is Verdict.INCONCLUSIVE, (
                f"{arm.arm} scored a crashed group as {verdict}, not INCONCLUSIVE"
            )


def test_a_wholly_crashing_kernel_is_refused_rather_than_scored(
    tmp_path: Path, repo_root: Path
):
    """The complement of the test above, and the reason it crashes on one rung only.

    Every arm INCONCLUSIVE everywhere is indistinguishable in the artifacts from an
    honest arm that caught nothing, and would persist as a detection rate of 0.0 with
    nothing saying so. The driver refuses it.
    """

    def always_crashes(x: np.ndarray) -> np.ndarray:
        msg = "kernel exploded"
        raise RuntimeError(msg)

    with pytest.raises(ValueError, match="established nothing anywhere"):
        drive(tmp_path / "run", always_crashes, "always_crashes", repo_root)
