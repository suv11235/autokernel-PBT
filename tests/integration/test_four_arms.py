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
from autokernel_pbt.props.tasks import REFERENCES, SOFTMAX, TASKS, softmax_reference
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


def test_allclose_and_reference_agree_on_a_correct_softmax(tmp_path: Path, repo_root: Path):
    """The two tolerance-bearing arms agree on softmax -- and ONLY on softmax.

    Scoped to softmax deliberately. An earlier version of this test was named for
    "a correct kernel" and read as a general claim, which is false: on layernorm the
    same two arms disagree on 5 of 9 groups, because a fixed `atol` cannot cope with
    an output centered on zero. See
    ``test_allclose_false_positives_on_a_correct_layernorm_kernel``.

    Softmax's absolute error scales with each element's own magnitude, so agreement
    here says the reference arm's calibration is not *looser* than the field default
    on that task. It says nothing about any other task.
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


# --------------------------------------------------------------------------- #
# The allclose arm's layernorm false positives
# --------------------------------------------------------------------------- #


def correct_layernorm(x: np.ndarray) -> np.ndarray:
    """Correct, accumulating in the input's own float32 rather than widening."""
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    variance = (centered * centered).mean(axis=-1, keepdims=True)
    return (centered / np.sqrt(variance + np.float32(1e-5))).astype(x.dtype)


def drive_layernorm(run_dir: Path, kernel, kernel_id: str, repo_root: Path) -> Path:
    run_task(
        task=TASKS["layernorm"],
        kernel=kernel,
        reference_fn=REFERENCES["layernorm"],
        run_dir=run_dir,
        repo_root=repo_root,
        n_groups=len(TASKS["layernorm"].domain.shapes),
        seed=SEED,
        kernel_id=kernel_id,
    )
    return run_dir


def test_allclose_false_positives_on_a_correct_layernorm_kernel(
    tmp_path: Path, repo_root: Path
):
    """A measured property of the field's default oracle, pinned so it stays visible.

    This is NOT a defect to be tuned away. `AllcloseOracle` is carried precisely
    because it is the untuned default the kernel literature uses, and changing its
    tolerances would delete the comparison it exists to provide.

    The mechanism: `atol=1e-8` is ~12x below float32 eps (1.19e-7), and a layernorm
    output is centered on zero by construction, so near-zero elements are guaranteed
    in every row. At an element whose true value is 3.6e-4 the budget is
    `rtol*3.6e-4 + atol = 1.4e-8`, an order of magnitude below the 4.2e-8 deviation a
    correct float32 kernel produces. Softmax is immune because its absolute error
    scales with each element's own magnitude, which is why the softmax-only agreement
    test does not catch this.

    Measured on seed 42: allclose FAILs 5 of 9 groups (6 of 18 cases) while the
    reference, declarative and hybrid arms pass all 9. That is a false-positive rate
    of 0.556 for the field's default on a correct kernel, and it is one of the
    sharper results this arm was added to produce.
    """
    run_dir = drive_layernorm(tmp_path / "run", correct_layernorm, "correct_ln", repo_root)
    _, arms = read_run(run_dir)

    by_arm = {}
    for arm in arms:
        groups: dict[str, list] = {}
        for result in arm.results:
            groups.setdefault(result.group_id, []).append(result)
        by_arm[arm.arm] = sum(
            summarize(results) is Verdict.FAIL for results in groups.values()
        )

    assert by_arm["allclose"] > 0, (
        "allclose no longer false-positives on layernorm; if its tolerances were "
        "tuned, the arm has stopped being the untuned default it exists to represent"
    )
    for name in ("reference", "declarative", "hybrid"):
        assert by_arm[name] == 0, f"{name} false-positived on a correct layernorm kernel"


def test_the_strengthened_reference_arm_beats_the_field_default_here(
    tmp_path: Path, repo_root: Path
):
    """The comparison the fourth arm was added to make, on the rung where it bites.

    The reference arm's log2(n)-normalized test ratio is scale-invariant, so it is
    unmoved by the near-zero elements that break a fixed `atol`. Softmax cannot show
    this -- both arms agree there -- so without layernorm the strengthened baseline
    would look like an unfalsifiable claim.
    """
    run_dir = drive_layernorm(tmp_path / "run", correct_layernorm, "correct_ln", repo_root)
    _, arms = read_run(run_dir)
    verdicts = {
        arm.arm: {r.verdict for r in arm.results} for arm in arms
    }
    assert Verdict.FAIL in verdicts["allclose"]
    assert Verdict.FAIL not in verdicts["reference"]


def test_a_recorded_run_can_be_regenerated_from_disk(tmp_path: Path, repo_root: Path):
    """The offline-shrinker path, end to end, with no generator handed in.

    Reads a recorded run back, takes the spec off a row, and rebuilds that group from
    the task's domain alone -- which is what a shrinker months later would do. The
    rebuilt tensors must be bitwise identical to the recorded ones, or a minimized
    reproducer would describe a case the run never executed.
    """
    from autokernel_pbt.props.generator import Generator
    from autokernel_pbt.props.table import ExecutionTable

    run_dir = drive(tmp_path / "run", correct_softmax, "correct_softmax", repo_root)
    rows = ExecutionTable(run_dir).read()

    recorded = next(r for r in rows if r.case.relation == "base")
    assert recorded.case_spec is not None, "the recorded run carries no recipe"

    rebuilt = Generator(SOFTMAX.domain, seed=recorded.case_spec.seed).group_from_spec(
        recorded.case_spec
    )
    assert rebuilt.group_id == recorded.case.group_id
    assert np.array_equal(rebuilt.base.tensors["x"], recorded.case.tensors["x"])
