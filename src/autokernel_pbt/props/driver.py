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
from dataclasses import replace
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
from autokernel_pbt.props.oracle import AllcloseOracle, HybridOracle, Oracle, ReferenceOracle
from autokernel_pbt.props.scores import ArmScores, ScoreTable
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import Task
from autokernel_pbt.props.verdict import PropertyResult, Verdict, summarize

#: The four arms, in canonical order: the field's default, the strengthened
#: reference, the declarative set, and their composition. This is the order they are
#: *persisted* in -- ``run_task`` sorts by it before writing, so two runs are
#: diffable -- while ``arm_order`` decides the order they are *evaluated* in.
ARM_NAMES = ("allclose", "reference", "declarative", "hybrid")


def arm_order(seed: int) -> list[str]:
    """The order to evaluate arms in for this run.

    ``elapsed_s`` is order-biased: the arm that runs second inherits everything the
    first one warmed. Under a *fixed* order that bias is systematic, so any
    cost-per-bug comparison between arms partly measures position rather than cost.
    With two arms that was already true; with four it is worse, because the last arm
    would always enjoy the most warming.

    Randomizing per run does not make a single run's timing fair — only repeated
    timing with a reported spread does, and that belongs to the metrics phase where
    the number is consumed. What it buys is that the bias no longer favours the same
    arm every time, so averaging across runs converges on cost instead of on
    position. See ``run_task``'s docstring for the measured magnitudes, which sit
    near the clock's noise floor.

    Derived from a seed rather than from entropy, because a run must replay: the same
    seed must give the same order, or two runs of "the same" experiment are not
    comparable even in principle.

    ``run_task`` takes that seed as its own ``arm_order_seed`` parameter, separate
    from the corpus seed, and the separation is the point. The experiment that
    consumes this randomization is repeated timing of *one* corpus; driving both from
    one seed would make every repetition evaluate the arms identically and leave the
    position bias fully systematic for exactly that measurement, while varying the
    corpus seed to shake the order would confound arm position with the inputs.
    """
    rng = np.random.default_rng([seed, 0xA6])
    return [ARM_NAMES[index] for index in rng.permutation(len(ARM_NAMES))]


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


def _keyed_by_group(
    results: list[PropertyResult], group_of_case: dict[str, str]
) -> list[PropertyResult]:
    """Stamp each case-scoped verdict with the group its case belongs to.

    The oracle layer's invariant is that a ``PropertyResult`` carries *exactly one* of
    ``case_id``/``group_id``: ``HybridOracle`` concatenates two arms into one flat list
    and the scope of a result is not otherwise recoverable from it. The *persisted*
    invariant is different — every score row carries ``group_id``, and ``case_id``
    refines it — because a reader of the artifacts has no way to perform the case-to-
    group join otherwise, and the group is the unit at which arms are comparable.

    The driver is the only layer that can bridge the two: it holds the recorded rows,
    which are where the mapping lives. Doing it in the oracles instead would mean
    changing the invariant that keeps the hybrid arm's split point recoverable, for the
    benefit of a file the oracles never see.

    A case id with no recorded row is left alone rather than raising here; that is
    ``_verify_join``'s finding to report, and reporting it from two places would let one
    guard's saboteur certify the other's.
    """
    return [
        replace(result, group_id=group_of_case[result.case_id])
        if result.case_id and not result.group_id and result.case_id in group_of_case
        else result
        for result in results
    ]


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

        # `.get` rather than `[]`: an unknown case id is the check above's finding, and
        # a KeyError raised from here would report it in the wrong guard's name.
        covered = {result.group_id or group_of_case.get(result.case_id, "") for result in arm.results}
        covered.discard("")
        unscored = sorted(recorded_groups - covered)
        if unscored:
            msg = (
                f"arm {arm.arm!r} scored only part of the recorded table; {len(unscored)} of "
                f"{len(recorded_groups)} recorded group(s) have no verdict from it: "
                f"{unscored}. Every rate would be computed over the covered subset alone, "
                f"with nothing in either file saying so"
            )
            raise ValueError(msg)

        # Coverage is necessary and not sufficient: an arm can name every group and
        # still have established nothing on any of them. Wholly-INCONCLUSIVE is what a
        # misconfigured arm looks like — a property set whose members all defer, a
        # reference that raised on every row — and it is indistinguishable in the
        # artifacts from an honest arm that simply caught nothing, while reporting 0.0
        # detection and 0.0 false positives with full confidence. Like `ScoreTable`'s
        # "arm has no results", this is a wiring bug rather than data, so it raises.
        #
        # KNOWN LIMIT: an arm that abstains on *some* groups is not caught here, and
        # cannot be — abstention is a legitimate, load-bearing answer (see
        # `oracle.py`), so only the degenerate all-or-nothing case is decidable.
        by_group: dict[str, list[PropertyResult]] = {}
        for result in arm.results:
            by_group.setdefault(result.group_id or group_of_case[result.case_id], []).append(
                result
            )
        # `by_group` empty means the arm judged nothing at all, which the coverage check
        # above already reported for any non-empty table; `all()` over nothing is True,
        # so without this the empty-table case would be blamed on the wrong guard.
        if by_group and all(
            summarize(results) is Verdict.INCONCLUSIVE for results in by_group.values()
        ):
            msg = (
                f"arm {arm.arm!r} summarizes to INCONCLUSIVE on every one of "
                f"{len(by_group)} recorded group(s); it established nothing anywhere, "
                f"which is a misconfigured arm rather than a result, and would be "
                f"persisted as a detection rate of 0.0 with nothing saying so"
            )
            raise ValueError(msg)


def read_run(run_dir: Path | str) -> tuple[list[ExecutionResult], list[ArmScores]]:
    """Both of a run's tables, refusing a pair that is not about the same corpus.

    THIS IS THE READ PATH FOR ANYONE COMPUTING A NUMBER. Reading the two tables
    separately is always possible and is sometimes right — but a rate is a statement
    about both at once, and the pairing is exactly what neither table can check alone.

    Case ids are a pure function of ``(seed, index)``, so a ``scores.parquet`` from a
    *correct-kernel* run dropped beside a *broken-kernel* ``rows.parquet`` joins on every
    key and reports 0.0 detection against a kernel the table labels broken. Nothing about
    that file is malformed; it is simply about a different run. The corpus fingerprint
    (see ``table.new_corpus_fingerprint``) is what makes it detectable, and this is where
    it is checked.

    An unstamped file — ``""`` on either side — is refused rather than waved through:
    it means a table written by a build that predates the column, or scores written by
    something other than the driver, and in both cases "cannot tell" must not read as
    "fine".

    Returns ``(rows, arms)``. A run with a table but no scores yet is not an error and
    returns ``(rows, [])``; a run with neither returns ``([], [])``.
    """
    table = ExecutionTable(run_dir)
    scores = ScoreTable(run_dir)
    rows = table.read()
    arms = scores.read()
    if not arms:
        return rows, arms

    recorded = table.corpus_fingerprint()
    judged = scores.corpus_fingerprint()
    if not recorded or not judged:
        msg = (
            f"run {Path(run_dir)} has scores that cannot be paired with its execution "
            f"table: corpus fingerprint is {recorded!r} on the rows and {judged!r} on the "
            f"scores. An unstamped file may belong to any run, so it is refused rather "
            f"than assumed to belong to this one"
        )
        raise ValueError(msg)
    if recorded != judged:
        msg = (
            f"run {Path(run_dir)} pairs scores for corpus {judged} with an execution "
            f"table for corpus {recorded}; these are different runs. Case ids are a pure "
            f"function of (seed, index), so the two join cleanly on every key and would "
            f"report a rate about neither run"
        )
        raise ValueError(msg)
    return rows, arms


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
    arm_order_seed: int | None = None,
) -> None:
    """Record one kernel's executions for one task, then score them with both arms.

    Writes two joined artifacts into ``run_dir`` and nothing else:
    ``rows.parquet`` + ``tensors/`` (the executions) and ``scores.parquet`` (the
    verdicts, one row per ``PropertyResult``, with the arm that produced it and what that
    arm cost).

    THE UNIT OF ANALYSIS IS THE CASE GROUP, and every score row is keyed by it — that
    is the guarantee, expressed as a key rather than as a note a reader may never see.

    The arms emit different numbers of results per group: the reference arm one per
    recorded case, the declarative arm one per case per case-property plus one per
    group-property. A rate computed per *result* is therefore a weighted average whose
    weights belong to the arm rather than to the kernel. Measured on the ladder corpus
    against ``unnormalized_softmax``, both arms find 14 FAILs, and:

    ======================  ==========  ===========  ==========
    rollup                  reference   declarative  comparable
    ======================  ==========  ===========  ==========
    per group (9 groups)    0.778       0.778        yes
    per case (18 cases)     0.778       0.778        yes, here
    per result              0.778       0.222        no
    ======================  ==========  ===========  ==========

    Note what this corpus does *not* say: per-case is equally comparable on it, because
    every arm here happens to judge every case. The group is nonetheless the unit the
    driver guarantees, and the reason is structural rather than numerical — a contract
    of only group properties emits no ``case_id`` at all, so a per-case guarantee would
    reject a legitimate arm, while a group-scoped verdict always exists. So:

    * ``_verify_join`` enforces that **every arm reaches a verdict on every recorded
      group** before anything is written;
    * ``_keyed_by_group`` stamps every case-scoped verdict with its group, so the correct
      rollup is ``GROUP BY group_id`` over ``scores.parquet`` alone — no join back to
      ``rows.parquet``, and no knowledge of this docstring.

    What is deliberately NOT added is a ``unit`` column naming the intended aggregation:
    that is a producer instructing a reader, and it would leave the reader without the
    key needed to follow the instruction. Recording the key makes the instruction
    unnecessary.

    ``kernel_id`` is keyword-only with no default so an unwired call is a ``TypeError``
    rather than a run whose rows silently carry ``""``. ``kernel_is_broken`` does default,
    to ``None`` — "not stated", which is distinct from ``False`` ("stated correct") and
    must stay so: collapsing them enlarges the correct-kernel denominator of the
    false-positive rate.

    ``elapsed_s`` IS NOT YET A FAIR COST DENOMINATOR, and a single run's value must not
    be quoted as one. It is well-defined — wall-clock seconds bracketing exactly
    ``oracle.evaluate``, with the contract load, the table read, the group keying and the
    join verification all outside it, symmetrically for both arms — but it is
    *order-biased*: the arm that runs second benefits from everything the first arm
    warmed. Measured over 40 paired trials on this corpus, the reference/declarative
    median ratio is 0.73 with the reference arm first and 0.70 with it second; an
    independent review measured 0.80 against 0.67 on other hardware. The direction is
    stable, the magnitude is not, and at ~0.5 ms per arm the whole measurement sits near
    the clock's noise floor.

    Arm order is randomized per run as of feature 0006 — see ``arm_order`` — so that
    bias is no longer *systematic*. A usable cost-per-bug figure still needs repeated
    timing with a reported spread, and that belongs to the metrics phase, which is
    where the number would be consumed; note also that at ~0.5 ms per arm the whole
    measurement sits near the clock's noise floor, so randomization buys correctness
    of method rather than precision. What this feature guarantees is only that the
    seconds are *recorded* — they are the one quantity in these artifacts that
    re-scoring cannot recover, because a second measurement is a different
    machine-minute.

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
            # Denormalized from the group onto each of its rows, so a recorded run is
            # regenerable offline without the generator that made it.
            result.case_spec = group.spec
            results.append(result)
    ExecutionTable(run_dir).write(results)

    # The replayed corpus. Read once, scored by every arm, so "both arms saw the same
    # rows" is true by construction rather than by comparison — and so that in-memory
    # interference between arms stays *detectable* (a fresh read per arm would give each
    # arm its own throwaway copy to scribble on).
    table = ExecutionTable(run_dir)
    recorded = table.read_groups()
    rows = [row for group_rows in recorded.values() for row in group_rows]
    group_of_case = {row.case.case_id: row.case.group_id for row in rows}

    # Built before the clock starts, and outside every arm's measurement, so the
    # contract load is not charged to the declarative arm's cost.
    declarative = oracle_from_contract(load_contract(contract_path(task, repo_root)))
    reference = ReferenceOracle(reference_fn)
    # allclose and reference are handed the *same* reference implementation, so every
    # difference between their verdicts is the comparison method and nothing else.
    # That is the whole reason the field's default is carried alongside the
    # strengthened ratio rather than described in prose.
    by_name: dict[str, Oracle] = {
        "allclose": AllcloseOracle(reference_fn),
        "reference": reference,
        "declarative": declarative,
        "hybrid": HybridOracle(declarative=declarative, reference=reference),
    }
    # NOT `seed`. The corpus seed and the arm-order seed are separate parameters
    # because the experiment that consumes the randomization is *repeated timing of
    # one corpus*: passing the corpus seed here would make every repetition of
    # seed=42 evaluate the arms in the identical order, leaving the position bias
    # fully systematic for exactly that measurement. Varying the corpus seed instead
    # would confound arm position with the inputs, so averaging would converge on
    # cost and on a different corpus at once. Defaults to `seed` so a single run is
    # still reproducible from its seed alone.
    order = arm_order(seed if arm_order_seed is None else arm_order_seed)
    arms: list[Oracle] = [by_name[name] for name in order]

    scored: list[ArmScores] = []
    for oracle in arms:
        # perf_counter, not process_time: cost-per-bug is wall-clock, which is what a
        # reader comparing an arm's cost against a hardware run's cost needs. See the
        # docstring for why a single run's value is not yet a fair denominator.
        start = time.perf_counter()
        arm_results = [result for group_rows in recorded.values() for result in oracle.evaluate(group_rows)]
        elapsed = time.perf_counter() - start
        # Keying happens after the clock stops: it is bookkeeping the driver does, not
        # work the arm did, and charging it to the arm would bias the very comparison
        # the timing exists to support.
        # `oracle.name`, not a literal: the arm name is the key every downstream
        # comparison groups by, and a literal here could disagree with the class that
        # produced the results.
        scored.append(
            ArmScores(
                arm=oracle.name,
                elapsed_s=elapsed,
                results=_keyed_by_group(arm_results, group_of_case),
            )
        )

    # Verified against the very rows the arms were handed, not a fresh read of the
    # table. A second read would verify the join for a corpus that may differ from the
    # scored one — which is the "execution table rewritten after scoring" attack, run by
    # the driver against itself.
    # Sorted into ARM_NAMES order *after* the clock has stopped, alongside the group
    # keying that is likewise excluded from the timing. Without this the on-disk arm
    # order is whatever arm_order() chose, so two runs of the same task under
    # different seeds produce score files whose rows are permuted for reasons
    # unrelated to the data, and any diff or fingerprint of the two sees noise.
    scored.sort(key=lambda arm: ARM_NAMES.index(arm.arm))
    _verify_join(rows, scored)
    # The identity is read back off the table rather than minted here, so the scores can
    # only ever claim the corpus that is actually on disk beside them.
    ScoreTable(run_dir).write(scored, corpus_fingerprint=table.corpus_fingerprint())
