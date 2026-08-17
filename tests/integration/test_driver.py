"""The driver, end to end: one composition of generate -> execute -> persist -> score.

``assert_replay_fairness`` in ``test_record_replay.py`` states the limit this module
exists to close: a test that passes one dict to two arms proves those two arms agree,
not that a *driver* hands both arms the same corpus. There was no driver to ask. There
is one now, and the first test below asks it.

Three things here are assertions about the artifacts rather than about objects in
memory, and that is deliberate — the research claim is that a recorded run is scorable
months later with no hardware:

* the detection rate is computed by reading ``rows.parquet`` and ``scores.parquet`` and
  nothing else, with no oracle, no kernel and no generator in the loop;
* the declarative arm's property set is read back *off disk* and compared against a
  contract file that was edited on the way in;
* the join between the two files is attacked five ways — a mistyped case key, a
  mistyped group key, an arm that covers only part of the table, an arm that
  establishes nothing anywhere, and a scores file belonging to a different run —
  because ``scores.py`` enumerates exactly those and hands the verification here.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.contract import CONTRACT_FILENAME, KERNEL_TASKS_DIR
from autokernel_pbt.props.driver import ARM_NAMES, arm_order, read_run, run_task
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.oracle import REFERENCE_PROPERTY, DeclarativeOracle, ReferenceOracle
from autokernel_pbt.props.properties import RowsSumToOne, ShiftInvariance
from autokernel_pbt.props.scores import SCORES_FILE, ArmScores, ScoreTable
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference
from autokernel_pbt.props.verdict import PropertyResult, Verdict, summarize

pytestmark = pytest.mark.integration

SEED = 42

#: The whole ladder. Anything smaller makes ``Generator`` warn that a boundary shape
#: will never be exercised, and this project turns warnings into errors — so the
#: default is not a performance choice, it is the only warning-free one.
ALL_SHAPES = len(SOFTMAX.domain.shapes)


# --------------------------------------------------------------------------- #
# Kernels under test
# --------------------------------------------------------------------------- #


def correct_softmax(x: np.ndarray) -> np.ndarray:
    """Correct, and deliberately not bit-identical to ``softmax_reference``.

    It accumulates in the input's own float32 rather than widening to float64, so the
    reference arm's threshold is actually consulted rather than being handed a residual
    of exactly zero. See ``test_record_replay.correct_softmax``, which makes the same
    choice for the same reason.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(x.dtype)


def unnormalized_softmax(x: np.ndarray) -> np.ndarray:
    """Broken: forgets the division by the row sum.

    Deterministic and data-independent, so "was it caught" is a statement about the
    oracles rather than about a lucky draw — except on the two single-column rungs,
    where it is *genuinely correct* (see ``_LADDER_SHAPES``) and both arms rightly pass.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(shifted).astype(x.dtype)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def drive(
    run_dir: Path,
    kernel: Callable[..., np.ndarray],
    kernel_id: str,
    kernel_is_broken: bool | None,
    repo_root: Path,
    n_groups: int = ALL_SHAPES,
    seed: int = SEED,
) -> Path:
    run_task(
        task=SOFTMAX,
        kernel=kernel,
        reference_fn=softmax_reference,
        run_dir=run_dir,
        repo_root=repo_root,
        n_groups=n_groups,
        seed=seed,
        kernel_id=kernel_id,
        kernel_is_broken=kernel_is_broken,
    )
    return run_dir


def group_verdicts(run_dir: Path, arm_name: str) -> dict[str, Verdict]:
    """One verdict per recorded group for one arm, from the two Parquet files only.

    THE UNIT OF ANALYSIS IS THE GROUP, and this is what reading at that unit costs now
    that every score row carries ``group_id``: a ``setdefault`` and a ``summarize``, with
    no case-to-group join and no knowledge of which module wrote the file. An earlier
    version of this helper did the join by hand — the same indirection a real consumer
    would have had to reinvent, and the divergence this task exists to end.

    ``read_run`` rather than two bare reads: a rate is a statement about both files at
    once, and pairing them is the only place a scores file from another run is visible.
    """
    rows, arms = read_run(run_dir)
    by_arm = {arm.arm: arm for arm in arms}
    per_group: dict[str, list[PropertyResult]] = {
        row.case.group_id: [] for row in rows
    }
    for result in by_arm[arm_name].results:
        per_group[result.group_id].append(result)
    return {gid: summarize(results) for gid, results in per_group.items()}


def fail_rate(run_dir: Path, arm_name: str) -> float:
    verdicts = group_verdicts(run_dir, arm_name)
    assert verdicts, "no groups scored; a rate over nothing is not a rate"
    return sum(v is Verdict.FAIL for v in verdicts.values()) / len(verdicts)


def property_names(run_dir: Path, arm_name: str) -> set[str]:
    arms = {arm.arm: arm for arm in ScoreTable(run_dir).read()}
    return {result.property_name for result in arms[arm_name].results}


# --------------------------------------------------------------------------- #
# ONE_DRIVER
# --------------------------------------------------------------------------- #


def test_both_arms_score_the_same_recorded_corpus(
    tmp_path: Path, monkeypatch, repo_root: Path
):
    """Acceptance criterion ONE_DRIVER.

    The obligation ``assert_replay_fairness`` documents and explicitly cannot discharge:
    "the two arms were handed the same groups" is unreachable in a test that passes one
    dict to both. Here the corpus is whatever the driver chose, read back off disk, and
    the claim is that both arms judged *that* corpus and all of it.

    Two phases, and the second is what makes the first non-vacuous.

    **Identity of coverage.** Every id each arm judged is a recorded id, and every
    recorded group has a verdict from every arm. Set equality, not containment, in both
    directions: containment alone would pass for an arm that scored one group of nine,
    the partial-coverage attack ``scores.py`` warns yields a plausible wrong denominator.

    **Identity of the rows themselves.** Ids are a function of ``(seed, index)`` and
    nothing else, so a driver that quietly regenerated and re-executed the corpus for the
    second arm would emit exactly the ids phase one demands while scoring different
    bytes. The only way to tell is to make the read path return something no kernel here
    produces — a NaN output — and require *both* arms to have seen it. An arm scoring any
    corpus other than the one that single read returned still passes the clean kernel and
    is caught.
    """
    run_dir = drive(tmp_path / "run", correct_softmax, "correct_softmax", False, repo_root)

    rows = ExecutionTable(run_dir).read()
    assert rows, "the driver recorded nothing; there is no corpus to share"
    recorded_cases = {row.case.case_id for row in rows}
    recorded_groups = {row.case.group_id for row in rows}

    arms = ScoreTable(run_dir).read()
    # Every arm the driver claims to run, not a hard-coded pair: adding a fifth arm
    # must extend this test rather than slip past it.
    assert {arm.arm for arm in arms} == set(ARM_NAMES)

    for arm in arms:
        judged_cases = {r.case_id for r in arm.results if r.case_id}
        judged_groups = {r.group_id for r in arm.results}
        assert judged_cases <= recorded_cases, f"{arm.arm} judged a case that was never recorded"
        assert judged_groups <= recorded_groups, f"{arm.arm} judged a group that was never recorded"
        # Every row carries the unit of analysis, so the rollup below needs no join.
        assert all(r.group_id for r in arm.results), (
            f"{arm.arm} persisted a verdict with no group_id; it cannot be rolled up"
        )
        # The corpus is shared only if it is also *whole* for each arm.
        assert set(group_verdicts(run_dir, arm.arm)) == recorded_groups, (
            f"{arm.arm} left part of the recorded table unscored"
        )

    # Both arms reached a verdict on every group, and agree on this clean kernel. The
    # agreement is not the criterion — it is what makes the coverage claim non-vacuous,
    # since an arm answering INCONCLUSIVE everywhere would also "cover" every group.
    reference = group_verdicts(run_dir, ReferenceOracle.name)
    declarative = group_verdicts(run_dir, DeclarativeOracle.name)
    assert set(reference.values()) == {Verdict.PASS}, reference
    assert set(declarative.values()) == {Verdict.PASS}, declarative

    # Phase two: one perturbed read, and both arms must show it.
    original = ExecutionTable.read_groups

    def read_groups(self: ExecutionTable) -> dict[str, list[Any]]:
        groups = original(self)
        for rows in groups.values():
            for row in rows:
                row.outputs = {
                    name: np.full_like(array, np.nan) for name, array in row.outputs.items()
                }
        return groups

    monkeypatch.setattr(ExecutionTable, "read_groups", read_groups)
    perturbed = drive(tmp_path / "perturbed", correct_softmax, "correct_softmax", False, repo_root)
    for arm in (ReferenceOracle.name, DeclarativeOracle.name):
        assert fail_rate(perturbed, arm) == 1.0, (
            f"{arm} did not score the rows the single read returned; it scored some other "
            f"corpus, so the arms are not provably sharing one"
        )


def test_the_kernel_is_executed_exactly_once_per_case(tmp_path: Path, repo_root: Path):
    """D: both arms replay one execution; neither re-runs the kernel.

    The negative claim needs a counter rather than an absence of evidence, so the kernel
    counts its own invocations and is compared against the number of persisted rows.

    Equality against a row count is only sound because a double execution cannot inflate
    both sides of it: ``ExecutionTable.write`` rejects duplicate ``case_id``s outright
    (they would share a payload file), so re-running a case adds an invocation and no
    row, and the equality breaks rather than tracking.
    """
    calls = {"n": 0}

    def counting(x: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return correct_softmax(x)

    run_dir = drive(tmp_path / "run", counting, "counting", False, repo_root)

    rows = ExecutionTable(run_dir).read()
    assert rows
    assert calls["n"] == len(rows), "the kernel ran a different number of times than there are rows"

    # And scoring, which happened inside run_task after the table was written, added
    # nothing: the two arms between them judged every row without a single re-execution.
    scored_cases = {
        r.case_id for arm in ScoreTable(run_dir).read() for r in arm.results if r.case_id
    }
    assert scored_cases == {row.case.case_id for row in rows}
    assert calls["n"] == len(rows)


def test_oracle_time_is_recorded_for_every_arm(tmp_path: Path, repo_root: Path):
    """Cost-per-bug is bugs over seconds, and the seconds are not re-derivable."""
    run_dir = drive(tmp_path / "run", correct_softmax, "correct_softmax", False, repo_root)
    arms = ScoreTable(run_dir).read()
    assert arms
    for arm in arms:
        assert arm.elapsed_s > 0.0, f"{arm.arm} recorded no elapsed time"


# --------------------------------------------------------------------------- #
# DETECTION_IS_DERIVABLE
# --------------------------------------------------------------------------- #


def test_detection_rate_is_computable_from_disk(tmp_path: Path, repo_root: Path):
    """Acceptance criterion DETECTION_IS_DERIVABLE: a real number, from artifacts alone.

    Two runs are recorded — one known-broken kernel, one known-correct — and every
    number below is then computed by reading ``rows.parquet`` and ``scores.parquet``.
    No oracle, kernel, generator or backend is consulted after ``run_task`` returns,
    which is the property that makes an expensive hardware run reusable.

    THE THRESHOLD IS NOT LAZINESS. ``_LADDER_SHAPES`` documents that two of the nine
    rungs are single-column, and softmax on a one-column input is exactly 1.0 for any
    implementation — so on those two groups this kernel is *genuinely correct* and both
    arms are right to pass it. The honest expectation is therefore 7/9, not 9/9, and
    asserting equality with 1.0 would be asserting a bug. The bound is stated as
    "at least the seven discriminating groups" so the test still fails if a detection
    is lost, and the blind spot is asserted separately below rather than hidden inside
    a slack threshold.
    """
    broken = drive(
        tmp_path / "broken", unnormalized_softmax, "unnormalized_softmax", True, repo_root
    )
    correct = drive(tmp_path / "correct", correct_softmax, "correct_softmax", False, repo_root)

    # Ground truth comes off the execution table, not from this test's local knowledge:
    # that is the join the kernel-identity columns exist to support.
    broken_rows = ExecutionTable(broken).read()
    correct_rows = ExecutionTable(correct).read()
    assert broken_rows and correct_rows
    assert all(row.kernel_is_broken is True for row in broken_rows)
    assert all(row.kernel_id == "unnormalized_softmax" for row in broken_rows)
    assert all(row.kernel_is_broken is False for row in correct_rows)
    assert all(row.kernel_id == "correct_softmax" for row in correct_rows)

    shapes = {row.case.group_id: tuple(row.case.shape) for row in broken_rows}
    discriminating = {gid for gid, shape in shapes.items() if shape[-1] > 1}
    degenerate = set(shapes) - discriminating
    assert discriminating and degenerate, shapes
    expected = len(discriminating) / len(shapes)

    for arm in (ReferenceOracle.name, DeclarativeOracle.name):
        detection = fail_rate(broken, arm)
        false_positive = fail_rate(correct, arm)
        assert detection >= expected, f"{arm} lost a detection: {detection} < {expected}"
        assert false_positive == 0.0, f"{arm} manufactured a false positive: {false_positive}"

        # The blind spot, asserted rather than absorbed into the threshold: the misses
        # are exactly the single-column groups, where the kernel is not broken at all.
        verdicts = group_verdicts(broken, arm)
        missed = {gid for gid, verdict in verdicts.items() if verdict is not Verdict.FAIL}
        assert missed == degenerate, (
            f"{arm} missed groups other than the documented single-column blind spot: {missed}"
        )


# --------------------------------------------------------------------------- #
# JOIN_IS_VERIFIED
# --------------------------------------------------------------------------- #


def _ghost_case(results: list[PropertyResult]) -> list[PropertyResult]:
    """A single mistyped ``case_id``: ``scores.py``'s second documented attack.

    Only one result is retargeted, and a case-scoped one whose group is still covered by
    its siblings — so coverage stays complete and this saboteur is reachable *only* by
    the unknown-id check. Retargeting a whole group would let the coverage check catch it
    too, and a saboteur caught by two guards certifies neither.
    """
    for index, result in enumerate(results):
        if result.case_id:
            return [*results[:index], replace(result, case_id="ghost-case"), *results[index + 1 :]]
    raise AssertionError("no case-scoped result to retarget; the saboteur is inert")


def _ghost_group(results: list[PropertyResult]) -> list[PropertyResult]:
    """A group-scoped verdict pointing at a group that was never recorded.

    Same construction as ``_ghost_case`` and for the same reason: the group keeps its
    case-scoped results, so coverage is untouched and only the unknown-id check can fire.
    """
    for index, result in enumerate(results):
        if result.group_id:
            return [
                *results[:index],
                replace(result, group_id="ghost-group"),
                *results[index + 1 :],
            ]
    raise AssertionError("no group-scoped result to retarget; the saboteur is inert")


def _all_inconclusive(results: list[PropertyResult]) -> list[PropertyResult]:
    """An arm that names every group and establishes nothing on any of them.

    Every id is real and coverage is complete, so all three join checks pass. What it
    reports is 0.0 detection and 0.0 false positives with full apparent confidence —
    indistinguishable in the artifacts from an honest arm that caught nothing. This is
    what a property set whose members all defer, or a reference that failed on every
    row, actually looks like.
    """
    return [replace(result, verdict=Verdict.INCONCLUSIVE) for result in results]


def _partial_coverage() -> Callable[[list[PropertyResult]], list[PropertyResult]]:
    """Score the first group and abstain on the rest: the third documented attack.

    Every id it emits is real, so both unknown-id checks pass and the denominator would
    silently become one. This is the saboteur that made the coverage check necessary.
    """
    state = {"groups": 0}

    def sabotage(results: list[PropertyResult]) -> list[PropertyResult]:
        state["groups"] += 1
        return results if state["groups"] == 1 else []

    return sabotage


#: Each saboteur paired with the message its own guard must produce. Pairing is what
#: makes these pin individual guards rather than "something raised": the unique-catcher
#: invariant this repo learned three times over. The first element is a *factory*
#: because the coverage saboteur is stateful across groups and must be built per test.
JOIN_SABOTEURS = {
    "score_names_an_unrecorded_case": (
        lambda: _ghost_case,
        r"case_id\(s\) that no recorded row carries",
    ),
    "score_names_an_unrecorded_group": (
        lambda: _ghost_group,
        r"group_id\(s\) that no recorded row carries",
    ),
    "scoring_covers_only_part_of_the_table": (
        _partial_coverage,
        r"scored only part of the recorded table",
    ),
    "the_arm_establishes_nothing_anywhere": (
        lambda: _all_inconclusive,
        r"summarizes to INCONCLUSIVE on every one of",
    ),
}


@pytest.mark.parametrize("name", sorted(JOIN_SABOTEURS))
def test_scores_that_do_not_join_the_recorded_rows_are_refused(
    tmp_path: Path, monkeypatch, repo_root: Path, name: str
):
    """Acceptance criterion JOIN_IS_VERIFIED, one attack at a time.

    ``scores.py`` enumerates the ways two individually valid Parquet files assert a rate
    nobody computed, and states that only the driver — which holds both tables at once —
    can refuse them. Each saboteur here is one of those ways, injected into the arm that
    would produce it, and each is matched against the message of the guard that must be
    its *sole* catcher.

    The refusal must also come *before* the write. A scores file left on disk beside a
    raised exception is precisely the artifact the criterion exists to prevent: it reads
    clean forever after, and nothing in it records that the run was rejected.
    """
    factory, expected = JOIN_SABOTEURS[name]
    sabotage = factory()
    original = DeclarativeOracle.evaluate

    def evaluate(self, rows):
        return sabotage(original(self, rows))

    monkeypatch.setattr(DeclarativeOracle, "evaluate", evaluate)

    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match=expected):
        drive(run_dir, correct_softmax, "correct_softmax", False, repo_root)

    assert not (run_dir / SCORES_FILE).exists(), (
        "the driver wrote a scores file it had already decided does not join"
    )
    # The execution table survives: the executions were paid for and are still valid;
    # it is the scoring pass, which is free to re-run, that was refused.
    assert ExecutionTable(run_dir).read(), "the refusal destroyed the recorded run"


def test_scores_from_another_run_are_refused(tmp_path: Path, repo_root: Path):
    """The attack ``JOIN_IS_VERIFIED`` names first and no key check can see.

    Case ids are a pure function of ``(seed, index)``, so the *correct* kernel's scores
    dropped onto the *broken* kernel's table join perfectly: every case id is recorded,
    every group is covered, nothing is malformed. The pair simply describes two
    different runs — and it reads as a detection rate of zero against a kernel the
    execution table itself labels broken.

    The damage is measured first, deliberately: the refusal is only worth having if the
    thing it refuses would otherwise have produced a plausible wrong number.
    """
    broken = drive(
        tmp_path / "broken", unnormalized_softmax, "unnormalized_softmax", True, repo_root
    )
    correct = drive(tmp_path / "correct", correct_softmax, "correct_softmax", False, repo_root)
    assert fail_rate(broken, DeclarativeOracle.name) > 0.0

    shutil.copy(correct / SCORES_FILE, broken / SCORES_FILE)

    # What the swap would have reported, read the naive way: two separate reads, every
    # key joining, and a rate of zero on a kernel recorded as broken.
    rows = ExecutionTable(broken).read()
    swapped = {arm.arm: arm for arm in ScoreTable(broken).read()}
    assert all(row.kernel_is_broken is True for row in rows)
    assert {r.case_id for r in swapped[DeclarativeOracle.name].results if r.case_id} <= {
        row.case.case_id for row in rows
    }
    assert not any(
        r.verdict is Verdict.FAIL for r in swapped[DeclarativeOracle.name].results
    ), "the swapped scores were not clean, so this proves nothing"

    with pytest.raises(ValueError, match="different runs"):
        read_run(broken)


def test_scores_that_carry_no_corpus_identity_are_refused(tmp_path: Path, repo_root: Path):
    """"Cannot tell" must not read as "fine".

    A scores file written by something other than the driver — or by a build predating
    the column — pairs with any table at all. Blanking the stamp is how that looks from
    the reader's side.
    """
    run_dir = drive(tmp_path / "run", correct_softmax, "correct_softmax", False, repo_root)
    table = pq.read_table(run_dir / SCORES_FILE)
    blanked = table.set_column(
        table.schema.get_field_index("corpus_fingerprint"),
        "corpus_fingerprint",
        pa.array([""] * table.num_rows, type=pa.string()),
    )
    pq.write_table(blanked, run_dir / SCORES_FILE)

    with pytest.raises(ValueError, match="cannot be paired"):
        read_run(run_dir)


def test_a_clean_run_is_not_refused(tmp_path: Path, repo_root: Path):
    """The join guards' negative control.

    Refusals that fired on everything would satisfy every test above while making the
    driver useless. This asserts the honest run reaches the write and pairs on read.
    """
    run_dir = drive(tmp_path / "run", correct_softmax, "correct_softmax", False, repo_root)
    assert (run_dir / SCORES_FILE).exists()
    rows, arms = read_run(run_dir)
    assert rows and arms
    assert ExecutionTable(run_dir).corpus_fingerprint() == ScoreTable(run_dir).corpus_fingerprint()


def test_a_run_with_no_scores_yet_pairs_without_complaint(tmp_path: Path, repo_root: Path):
    """Recorded but not scored is a legitimate state, not a mismatch.

    Recording is the expensive half and may finish long before any arm runs; a pairing
    check that refused it would make the artifacts unreadable between the two steps.
    """
    run_dir = tmp_path / "run"
    backend = NumpyBackend()
    ExecutionTable(run_dir).write(
        [
            backend.run(correct_softmax, case)
            for group in Generator(SOFTMAX.domain, SEED).generate(ALL_SHAPES)
            for case in group.cases
        ]
    )
    rows, arms = read_run(run_dir)
    assert rows and arms == []


# --------------------------------------------------------------------------- #
# CONTRACT_DRIVES_THE_ARM
# --------------------------------------------------------------------------- #


def _contract_path(root: Path) -> Path:
    return root / KERNEL_TASKS_DIR / SOFTMAX.task_id / CONTRACT_FILENAME


def test_the_declarative_arm_comes_from_the_contract(tmp_path: Path, repo_root: Path):
    """Acceptance criterion CONTRACT_DRIVES_THE_ARM: the file is load-bearing.

    ``test_contract.py`` already shows that *an oracle built from a contract* loses a
    detection when a criterion is deleted. What it cannot show is that the driver builds
    its arm that way rather than from a hardcoded tuple — and until this test,
    ``contract.py`` was imported by nothing but its own tests, so the spec-as-oracle path
    was unexercised in production.

    So the edit is made to a copy of the whole ``kernels/`` tree and the driver is
    pointed at that root. Nothing else changes. The assertion is made against the
    *persisted* scores, because what the driver held in memory is not the artifact
    anyone will read.
    """
    full = drive(tmp_path / "full", unnormalized_softmax, "unnormalized_softmax", True, repo_root)
    assert RowsSumToOne.name in property_names(full, DeclarativeOracle.name)
    assert fail_rate(full, DeclarativeOracle.name) > 0.0

    trimmed_root = tmp_path / "trimmed_root"
    shutil.copytree(repo_root / "kernels", trimmed_root / "kernels")
    path = _contract_path(trimmed_root)
    document = yaml.safe_load(path.read_text())
    document["criteria"] = [
        c for c in document["criteria"] if c["check"]["property"] != RowsSumToOne.name
    ]
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    trimmed = drive(
        tmp_path / "trimmed",
        unnormalized_softmax,
        "unnormalized_softmax",
        True,
        trimmed_root,
    )
    scored = property_names(trimmed, DeclarativeOracle.name)
    assert RowsSumToOne.name not in scored, "the driver ignored the contract"
    # The rest of the contract still ran, so this is a removed criterion rather than a
    # broken arm — and the remaining laws are the ones that genuinely cannot see this
    # bug, which is why the detection goes to zero.
    assert ShiftInvariance.name in scored
    assert fail_rate(trimmed, DeclarativeOracle.name) == 0.0, (
        "the weakened contract still caught the bug; the removed criterion was not what "
        "was doing the catching, so this test proves nothing"
    )

    # The reference arm is untouched by the contract and must still catch it — otherwise
    # the run above differs from the first in more than the one deleted criterion.
    assert REFERENCE_PROPERTY in property_names(trimmed, ReferenceOracle.name)
    assert fail_rate(trimmed, ReferenceOracle.name) > 0.0


def test_an_unwired_kernel_id_is_a_type_error():
    """``kernel_id`` is keyword-only with no default, deliberately.

    A default of ``""`` would make an unwired call record rows whose ground truth is
    unstated, and the loss is silent and unrecoverable: nothing downstream can tell an
    unlabelled row from a labelled one after the run is over.
    """
    with pytest.raises(TypeError, match="kernel_id"):
        run_task(  # type: ignore[call-arg]
            task=SOFTMAX,
            kernel=correct_softmax,
            reference_fn=softmax_reference,
            run_dir="unused",
            repo_root=Path("unused"),
            n_groups=1,
            seed=SEED,
        )


def test_arm_scores_are_written_once_per_arm(tmp_path: Path, repo_root: Path):
    """``ArmScores`` requires one entry per arm per write; two would merge on read."""
    run_dir = drive(tmp_path / "run", correct_softmax, "correct_softmax", False, repo_root)
    arms: list[ArmScores] = ScoreTable(run_dir).read()
    names = [arm.arm for arm in arms]
    assert len(names) == len(set(names)) == len(ARM_NAMES), names




# --------------------------------------------------------------------------- #
# Arm evaluation order (feature 0006)
# --------------------------------------------------------------------------- #


def test_arm_order_varies_with_seed():
    """The criterion ARM_ORDER_IS_RANDOMIZED.

    ``elapsed_s`` is order-biased: the arm that runs second inherits everything the
    first warmed. Under a fixed order that bias is systematic, so a cost-per-bug
    comparison would partly measure position. This does not make one run's timing
    fair -- only repetition does -- but it stops the bias favouring the same arm
    every time.
    """
    orders = {tuple(arm_order(seed)) for seed in range(20)}
    assert len(orders) > 1, "arm order is identical across every seed"


def test_arm_order_is_a_permutation_of_every_arm():
    # A shuffle that dropped or duplicated an arm would silently change what the
    # experiment measures, and the coverage check would not see it.
    for seed in range(10):
        assert sorted(arm_order(seed)) == sorted(ARM_NAMES)


def test_arm_order_is_deterministic_for_a_seed():
    # A run must replay: same seed, same order, or two runs of "the same" experiment
    # are not comparable even in principle.
    assert arm_order(3) == arm_order(3)
