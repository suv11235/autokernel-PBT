"""The one composition of the pipeline: generate, execute, persist, score.

Phase 1 assembled this twice in test files and the two copies had already drifted. This
module is the single composition, and three of its choices are load-bearing rather than
incidental.

*The arms score the table, not the run.* Every arm is handed rows read back from
``rows.parquet``, never the in-memory ``ExecutionResult`` list that produced it. The
research claim is that a recorded run can be scored months later on a machine with no
device attached; the replay path is therefore the path that must be exercised on every
run, not a second path validated once and then bypassed by the driver that matters.

*The corpus is shared by construction.* One generation, one execution, one read, both
arms. ``assert_replay_fairness`` in ``tests/integration/test_record_replay.py`` states
plainly that "the two arms were handed the same groups" is unreachable in a test that
passes one dict to two arms, and hands the obligation here.

*The declarative arm is built from the task's contract*, never from a property tuple
written in this file. Writing the spec is writing the oracle — that is what makes
authoring cost a measurable quantity — and it is what stops the driver silently applying
softmax's laws to a task that does not obey them. Until this module existed,
``contract.py`` was imported by nothing but its own tests.

THE UNIT OF ANALYSIS IS THE GROUP, and the driver's guarantee is what makes that
computable; see ``run_task`` for the statement and ``_verify_join`` for the enforcement.

WHAT IS NOT DONE HERE: metrics. This feature makes the numbers derivable, not derived.
The driver writes two joined tables and refuses to write a pair that does not join; the
rate itself is computed by whoever reads them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from autokernel_pbt.props.backends.base import ExecutionResult
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.contract import (
    CONTRACT_FILENAME,
    KERNEL_TASKS_DIR,
    load_contract,
    oracle_from_contract,
)
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.oracle import Oracle, ReferenceOracle
from autokernel_pbt.props.scores import ArmScores, ScoreTable
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import Task


def contract_path(task: Task, repo_root: Path) -> Path:
    """Where ``task``'s acceptance contract lives under ``repo_root``.

    ``KERNEL_TASKS_DIR`` and ``CONTRACT_FILENAME`` are imported from ``contract.py``
    rather than restated. A second copy of the layout here could disagree with the
    loader's own ``contract_paths`` walk, and the failure would be a driver that reads a
    file the project's contract inventory does not know exists.

    ``repo_root`` is a parameter for the same reason ``contract_paths`` takes one: the
    package is installed from ``src/`` and an installed copy has no repo above it, so a
    root derived from ``__file__`` would be right under test and wrong in a wheel.
    """
    return repo_root / KERNEL_TASKS_DIR / task.task_id / CONTRACT_FILENAME


def _verify_join(recorded: list[ExecutionResult], scored: list[ArmScores]) -> None:
    """Refuse a scores/rows pair that does not join. The obligation ``scores.py`` assigns.

    Neither file carries a run identity, so nothing inside either module can tell that a
    ``scores.parquet`` belongs to the ``rows.parquet`` beside it. Only the driver holds
    both tables at once, and only before the write is the refusal free. Three checks,
    each corresponding to one of the attacks ``scores.py`` enumerates, and each raising a
    message that names which one fired — a shared message would let one guard's saboteur
    silently certify another's.

    1. Every ``case_id`` a score carries is a recorded case. Catches a mistyped id and a
       scores file copied in from another run, both of which orphan silently: the
       affected rows drop out of the numerator and the denominator is computed anyway.
    2. Every ``group_id`` a score carries is a recorded group. Separate from (1) because
       a result carries exactly one of the two, so a check on cases alone is blind to
       every group-scoped verdict — which is where the metamorphic detections live.
    3. Every arm covered every recorded group. This is the one the other two cannot see:
       an arm that scored three groups of nine emits nothing but valid ids and yields a
       rate over a denominator of three. Coverage is asserted per *arm*, not over the
       union of arms, because a rate is computed per arm and a fully-covering reference
       arm would otherwise mask a partially-covering declarative one.

    Coverage is measured in groups, not cases, and that is a deliberate limit rather than
    an oversight: a contract of only group properties emits no ``case_id`` at all, so
    requiring per-case coverage would reject a legitimate arm. See ``run_task`` for why
    the group is the unit anyway.
    """
    group_of_case = {row.case.case_id: row.case.group_id for row in recorded}
    recorded_groups = {row.case.group_id for row in recorded}

    for arm in scored:
        unknown_cases = sorted(
            {result.case_id for result in arm.results if result.case_id} - group_of_case.keys()
        )
        if unknown_cases:
            msg = (
                f"arm {arm.arm!r} produced scores carrying case_id(s) that no recorded row "
                f"carries: {unknown_cases}. The scores do not join the execution table, so "
                f"every rate computed from the pair would silently drop them"
            )
            raise ValueError(msg)

        unknown_groups = sorted(
            {result.group_id for result in arm.results if result.group_id} - recorded_groups
        )
        if unknown_groups:
            msg = (
                f"arm {arm.arm!r} produced scores carrying group_id(s) that no recorded row "
                f"carries: {unknown_groups}. The scores do not join the execution table, so "
                f"every rate computed from the pair would silently drop them"
            )
            raise ValueError(msg)

        covered = {
            result.group_id or group_of_case[result.case_id]
            for result in arm.results
            if result.case_id or result.group_id
        }
        unscored = sorted(recorded_groups - covered)
        if unscored:
            msg = (
                f"arm {arm.arm!r} scored only part of the recorded table; {len(unscored)} of "
                f"{len(recorded_groups)} recorded group(s) have no verdict from it: "
                f"{unscored}. Every rate would be computed over the covered subset alone, "
                f"with nothing in either file saying so"
            )
            raise ValueError(msg)


def run_task(
    *,
    task: Task,
    kernel: Callable[..., np.ndarray],
    reference_fn: Callable[..., np.ndarray],
    run_dir: Path | str,
    repo_root: Path,
    n_groups: int,
    seed: int,
    kernel_id: str,
    kernel_is_broken: bool | None = None,
) -> None:
    """Record one kernel's executions for one task, then score them with both arms.

    Writes two joined artifacts into ``run_dir`` and nothing else:
    ``rows.parquet`` + ``tensors/`` (the executions) and ``scores.parquet`` (the
    verdicts, one row per ``PropertyResult``, with the arm that produced it and what that
    arm cost).

    THE UNIT OF ANALYSIS IS THE CASE GROUP. This is a guarantee, not an observation, and
    it is the one thing a reader of the artifacts must know that the schema does not say.

    The arms emit different numbers of results per group: the reference arm one per
    recorded case, the declarative arm one per case per case-property plus one per
    group-property. A rate computed per *result* is therefore a weighted average whose
    weights are a property of the arm rather than of the kernel — measured on this
    corpus, the same declarative arm reports 0.778 rolled up per group and 0.519 per
    result, and only the first of those is comparable against the reference arm. So:

    * what the driver guarantees is that **every arm reaches a verdict on every recorded
      group**, enforced by ``_verify_join`` before anything is written;
    * a rate over these artifacts must be computed by joining each score to its group
      (``group_id`` directly, or ``case_id`` through ``rows.parquet``), folding the
      group's results with ``verdict.summarize``, and counting groups.

    Deliberately NOT recorded as a column. A ``unit`` column on ``scores.parquet`` would
    be a field describing how a *reader* should aggregate, written by the producer, and
    it would be the only such field in either schema — the rows are per-result and stay
    per-result. Adding it belongs with the metric computation this feature explicitly
    defers, where it can be introduced with a consumer that reads it; a column nothing
    reads is a claim nothing checks.

    ``kernel_id`` is keyword-only with no default so an unwired call is a ``TypeError``
    rather than a run whose rows silently carry ``""``. ``kernel_is_broken`` does default,
    to ``None`` — "not stated", which is distinct from ``False`` ("stated correct") and
    must stay so: collapsing them enlarges the correct-kernel denominator of the
    false-positive rate.

    Raises ``ValueError`` before writing any scores if the verdicts do not join the
    recorded rows; the execution table is left intact, because the executions are the
    part that cost hardware time and the scoring pass is free to re-run.
    """
    groups = Generator(task.domain, seed).generate(n_groups)
    backend = NumpyBackend()

    results: list[ExecutionResult] = []
    for group in groups:
        for case in group.cases:
            result = backend.run(kernel, case)
            # Ground truth is stamped here rather than inside the backend: it is a fact
            # about the experiment, not about the execution, and a backend that knew it
            # could in principle let it influence what it records.
            result.kernel_id = kernel_id
            result.kernel_is_broken = kernel_is_broken
            results.append(result)
    ExecutionTable(run_dir).write(results)

    # The replayed corpus. Read once, scored by every arm, so "both arms saw the same
    # rows" is true by construction rather than by comparison — and so that in-memory
    # interference between arms stays *detectable* (a fresh read per arm would give each
    # arm its own throwaway copy to scribble on).
    recorded = ExecutionTable(run_dir).read_groups()

    declarative = oracle_from_contract(load_contract(contract_path(task, repo_root)))
    arms: list[Oracle] = [ReferenceOracle(reference_fn), declarative]

    scored: list[ArmScores] = []
    for oracle in arms:
        # perf_counter, not process_time: cost-per-bug is wall-clock, which is what a
        # reader comparing an arm's cost against a hardware run's cost needs.
        start = time.perf_counter()
        arm_results = [result for rows in recorded.values() for result in oracle.evaluate(rows)]
        elapsed = time.perf_counter() - start
        # `oracle.name`, not a literal: the arm name is the key every downstream
        # comparison groups by, and a literal here could disagree with the class that
        # produced the results.
        scored.append(ArmScores(arm=oracle.name, elapsed_s=elapsed, results=arm_results))

    # Verified against the very rows the arms were handed, not a fresh read of the
    # table. A second read would verify the join for a corpus that may differ from the
    # scored one — which is the "execution table rewritten after scoring" attack, run by
    # the driver against itself.
    _verify_join([row for rows in recorded.values() for row in rows], scored)
    ScoreTable(run_dir).write(scored)
