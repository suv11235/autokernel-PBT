"""End-to-end record/replay: generate once, execute once, persist, score offline.

This module holds ``REPLAY_FAIRNESS``, the acceptance criterion the whole
batch-first architecture exists to satisfy. The research claim is that three
oracle strategies are compared over *byte-identical* executions; if the choice of
oracle could influence the inputs a kernel saw, the comparison would confound
generation with checking and the headline result would be unfalsifiable.

The guarantee is structural, not aspirational: oracles are handed rows read back
from a persisted table and have no reference to the generator or the backend at
all. These tests assert that the structure actually delivers it, and that the
assertion is not vacuous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, ExecutionResult, Status
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.oracle import (
    REFERENCE_PROPERTY,
    DeclarativeOracle,
    HybridOracle,
    ReferenceOracle,
    summary,
)
from autokernel_pbt.props.properties import (
    SOFTMAX_CASE_PROPERTIES,
    SOFTMAX_GROUP_PROPERTIES,
    OutputsAreFinite,
    RowsSumToOne,
    ShiftInvariance,
    ValuesInUnitInterval,
)
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import REFERENCES, TASKS
from autokernel_pbt.props.verdict import Verdict

pytestmark = pytest.mark.integration

SEED = 20240815

#: The relu task declares ``relations = ()``, so its groups hold a base case and
#: nothing else, and its outputs are unbounded above. Neither
#: ``values_in_unit_interval`` nor ``rows_sum_to_one`` is a law relu obeys, and
#: ``shift_invariance`` has no partner to find — see
#: ``test_relu_uses_only_the_properties_relu_actually_obeys`` for why passing the
#: softmax set here would be an error that hides as an INCONCLUSIVE.
RELU_CASE_PROPERTIES = (OutputsAreFinite(),)


# --------------------------------------------------------------------------- #
# Kernels under test
# --------------------------------------------------------------------------- #


def correct_softmax(x: np.ndarray) -> np.ndarray:
    """A correct softmax that is *not* the reference implementation.

    It accumulates in float64 and casts back, so the recorded output differs from
    ``softmax_reference`` in the last bits. A kernel that were literally the
    reference would make the reference arm's comparison vacuous (ratio exactly 0)
    and would prove nothing about the threshold.
    """
    wide = x.astype(np.float64)
    shifted = wide - np.max(wide, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(x.dtype)


def unnormalized_softmax(x: np.ndarray) -> np.ndarray:
    """Broken: forgets the division by the row sum.

    Chosen as the primary broken kernel because it is deterministic and
    data-independent — ``exp(x - max(x))`` lies in (0, 1] for every input, so
    ``values_in_unit_interval`` passes and ``rows_sum_to_one`` fails on every row
    of every shape. That makes "which arm caught it, and via which property" an
    assertion about the oracles rather than about a lucky draw.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(shifted).astype(x.dtype)


def naive_softmax(x: np.ndarray) -> np.ndarray:
    """Broken: no max-subtraction, so it overflows on a shifted row.

    Two details are load-bearing and neither is stylistic.

    ``np.errstate`` is fidelity. This models a real device kernel, which produces
    ``inf`` and returns; it does not raise. Without the suppression, numpy's
    overflow RuntimeWarning becomes an exception under this project's
    ``filterwarnings = ["error"]``, the backend classifies the row as
    ``launch_error``, and every property returns INCONCLUSIVE — so the test would
    exercise the failed-execution path instead of the numerical-instability path
    it is named for.

    The accumulation stays in the input's own dtype. Widening to float64 first
    would move the overflow point from ``x > 88.7`` to ``x > 709``, which
    ``ShiftRows``' float32-calibrated shift scale never reaches — the kernel would
    then be *numerically fine* and the test would pass while asserting nothing.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        exp = np.exp(x)
        return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(x.dtype)


def nan_relu(x: np.ndarray) -> np.ndarray:
    """Broken: returns NaN wherever the input is negative, instead of zero."""
    return np.where(x > 0, x, np.nan).astype(x.dtype)


def correct_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, np.zeros((), dtype=x.dtype))


# --------------------------------------------------------------------------- #
# The pipeline under test
# --------------------------------------------------------------------------- #


class CountingBackend:
    """A real ``NumpyBackend`` that records how many executions it performed.

    Replay's whole promise is that scoring costs no hardware. That is only
    checkable if something counts executions, so the count is taken here rather
    than inferred.
    """

    def __init__(self) -> None:
        self.inner = NumpyBackend()
        self.name = self.inner.name
        self.calls = 0

    def run(self, kernel: Callable[..., np.ndarray], case: Any) -> ExecutionResult:
        self.calls += 1
        return self.inner.run(kernel, case)


def record(
    task_id: str,
    kernel: Callable[..., np.ndarray],
    run_dir: Path,
    n_groups: int | None = None,
    seed: int = SEED,
) -> tuple[ExecutionTable, CountingBackend]:
    """Run the real pipeline once: generate -> execute -> persist.

    Returns a *fresh* reader over the persisted run, never the in-memory results.
    Everything downstream of this function sees only what survived the table,
    which is exactly the constraint the architecture claims to impose on oracles.

    ``n_groups`` defaults to the number of ladder shapes so every boundary shape
    is exercised; a smaller value is what ``Generator`` warns about, and this
    project turns warnings into errors.
    """
    task = TASKS[task_id]
    if n_groups is None:
        n_groups = len(task.domain.shapes)
    groups = Generator(task.domain, seed).generate(n_groups)
    backend = CountingBackend()
    results = [backend.run(kernel, case) for group in groups for case in group.cases]
    ExecutionTable(run_dir).write(results)
    return ExecutionTable(run_dir), backend


def fingerprint(tensors: dict[str, np.ndarray]) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """Identity of a tensor set at the byte level.

    ``np.array_equal`` is deliberately not used anywhere in this module. It
    reports True for a float32 array and a float64 array holding the same values,
    and for arrays whose shapes merely broadcast — so an arm silently rescoring a
    promoted copy of the inputs would pass a fairness test built on it. dtype,
    shape and ``tobytes()`` together admit no such reading.
    """
    return {
        name: (array.dtype.str, tuple(array.shape), array.tobytes())
        for name, array in sorted(tensors.items())
    }


def table_fingerprint(table: ExecutionTable) -> dict[str, dict[str, Any]]:
    return {row.case.case_id: fingerprint(row.case.tensors) for row in table.read()}


def reference_oracle() -> ReferenceOracle:
    """The reference arm for softmax.

    Constructed with no ``n`` override on purpose. ``ReferenceOracle._length``
    takes the accumulation length from ``row.case.shape[-1]``, and softmax reduces
    along the last axis, so the default *is* the right length here; an override
    would be a second source of truth for a value already recorded on the case.
    ``test_reference_arm_length_matches_the_softmax_reduction_axis`` pins that.
    """
    return ReferenceOracle(REFERENCES["softmax"])


def declarative_oracle() -> DeclarativeOracle:
    return DeclarativeOracle(SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES)


def arm_verdicts(oracle: Any, table: ExecutionTable) -> dict[str, Verdict]:
    return {
        group_id: summary(oracle.evaluate(rows))
        for group_id, rows in table.read_groups().items()
    }


def failing_properties(oracle: Any, table: ExecutionTable) -> set[str]:
    return {
        result.property_name
        for rows in table.read_groups().values()
        for result in oracle.evaluate(rows)
        if result.verdict is Verdict.FAIL
    }


# --------------------------------------------------------------------------- #
# REPLAY_FAIRNESS
# --------------------------------------------------------------------------- #


def rows_fingerprint(groups: dict[str, list[ExecutionResult]]) -> dict[str, Any]:
    return {
        row.case.case_id: fingerprint(row.case.tensors)
        for rows in groups.values()
        for row in rows
    }


def test_both_arms_see_byte_identical_inputs(tmp_path: Path):
    """Acceptance criterion REPLAY_FAIRNESS.

    Three claims, in increasing strength.

    First, the table is a fixed point: reading it twice yields byte-identical
    input tensors. On its own this could pass trivially, so it is stated with
    dtype, shape and raw bytes rather than value equality —
    ``test_the_fairness_assertion_can_actually_fail`` pins that the comparison can
    tell a perturbed table from an intact one.

    Second, and this is what the research claim actually needs: the two arms are
    handed *the same row objects*, not two equivalent reads. Fairness is not "the
    arms saw equal-looking data", it is "there was one execution and both arms
    scored it". Passing the identical objects makes any divergence impossible by
    construction rather than by comparison, and makes in-memory mutation
    detectable — a fresh read per arm would hide it, since each arm would scribble
    on its own throwaway copy.

    Third, neither arm perturbs what it was handed. The reference arm recomputes
    softmax from every recorded input and the declarative arm scans and reduces
    over outputs; either could plausibly write in place. The shared rows are
    fingerprinted after each arm, and the table is re-read from disk at the end to
    cover an arm that reached past its arguments to the ledger itself.
    """
    table, _ = record("softmax", correct_softmax, tmp_path / "run")

    first = table_fingerprint(table)
    second = table_fingerprint(ExecutionTable(tmp_path / "run"))
    assert first, "the recorded run is empty; there is nothing to be fair about"
    assert first.keys() == second.keys()
    for case_id in first:
        assert first[case_id] == second[case_id], f"inputs differ between reads for {case_id}"

    # One read, shared by both arms. Everything below scores these exact objects.
    shared = table.read_groups()
    before = rows_fingerprint(shared)
    assert before == first, "the shared read differs from the table on disk"

    reference_verdicts = {
        group_id: summary(reference_oracle().evaluate(rows)) for group_id, rows in shared.items()
    }
    after_reference = rows_fingerprint(shared)
    assert after_reference == before, "the reference arm mutated the rows it was handed"

    declarative_verdicts = {
        group_id: summary(declarative_oracle().evaluate(rows))
        for group_id, rows in shared.items()
    }
    after_declarative = rows_fingerprint(shared)
    assert after_declarative == before, "the declarative arm mutated the rows it was handed"

    # Both arms must actually have judged every recorded group, or "same inputs"
    # is a statement about rows nobody scored.
    assert reference_verdicts.keys() == declarative_verdicts.keys() == shared.keys()
    assert reference_verdicts and declarative_verdicts

    assert table_fingerprint(ExecutionTable(tmp_path / "run")) == before, (
        "scoring changed the persisted table"
    )


def test_the_fairness_assertion_can_actually_fail(tmp_path: Path):
    """Guard against a vacuous REPLAY_FAIRNESS.

    ``fingerprint`` is the comparison the criterion above is built on. If it
    could not distinguish a perturbed table from an intact one, the criterion
    would assert nothing. Three perturbations that ``np.array_equal`` would miss
    or under-report are injected here: a one-ulp value change, a dtype promotion,
    and a reshape. Each must be visible.
    """
    table, _ = record("softmax", correct_softmax, tmp_path / "run")
    rows = table.read()
    original = fingerprint(rows[0].case.tensors)

    x = rows[0].case.tensors["x"]

    perturbed = x.copy()
    flat = perturbed.reshape(-1)
    flat[0] = np.nextafter(flat[0], np.float32(np.inf))
    assert fingerprint({"x": perturbed}) != {"x": original["x"]}

    promoted = x.astype(np.float64)
    assert np.array_equal(promoted, x), "the promotion must be invisible to value equality"
    assert fingerprint({"x": promoted}) != {"x": original["x"]}

    reshaped = x.reshape(1, -1)
    assert fingerprint({"x": reshaped}) != {"x": original["x"]}


# --------------------------------------------------------------------------- #
# A correct kernel, a broken kernel
# --------------------------------------------------------------------------- #


def test_correct_kernel_passes_both_arms(tmp_path: Path):
    table, _ = record("softmax", correct_softmax, tmp_path / "run")

    reference = arm_verdicts(reference_oracle(), table)
    declarative = arm_verdicts(declarative_oracle(), table)

    assert reference and declarative
    assert set(reference.values()) == {Verdict.PASS}, reference
    assert set(declarative.values()) == {Verdict.PASS}, declarative


def group_shapes(table: ExecutionTable) -> dict[str, tuple[int, ...]]:
    return {row.case.group_id: tuple(row.case.shape) for row in table.read()}


def test_unnormalized_softmax_is_caught_by_both_arms(tmp_path: Path):
    """A missing normalization, and exactly which property sees it.

    ``rows_sum_to_one`` is the declarative arm's finding; ``values_in_unit_interval``
    must *not* fire, because ``exp(x - max)`` genuinely lies in (0, 1]. Asserting the
    non-firing matters as much as the firing: a property that failed here would be
    reporting a bug it cannot actually see.

    Single-column shapes are excluded, and the exclusion is the interesting part.
    On an ``(m, 1)`` input every row's max is its only element, so
    ``exp(x - max) == 1`` and the row already sums to one — this kernel is not
    merely undetected there, it is *correct* there, and so is agreed to be correct
    by both arms. The degenerate rungs of the ladder are therefore blind to a
    whole class of normalization bug, which is a fact about the corpus rather than
    about the oracles; it is asserted below so it cannot be mistaken for a
    detection gap in an arm.
    """
    table, _ = record("softmax", unnormalized_softmax, tmp_path / "run")
    shapes = group_shapes(table)
    discriminating = {gid for gid, shape in shapes.items() if shape[-1] > 1}
    degenerate = set(shapes) - discriminating
    assert discriminating and degenerate, shapes

    declarative = declarative_oracle()
    verdicts = arm_verdicts(declarative, table)
    assert {verdicts[gid] for gid in discriminating} == {Verdict.FAIL}
    assert {verdicts[gid] for gid in degenerate} == {Verdict.PASS}

    failed = failing_properties(declarative, table)
    assert RowsSumToOne.name in failed
    assert ValuesInUnitInterval.name not in failed
    assert OutputsAreFinite.name not in failed

    reference = reference_oracle()
    reference_verdicts = arm_verdicts(reference, table)
    assert {reference_verdicts[gid] for gid in discriminating} == {Verdict.FAIL}
    assert {reference_verdicts[gid] for gid in degenerate} == {Verdict.PASS}
    assert failing_properties(reference, table) == {REFERENCE_PROPERTY}


def test_naive_softmax_is_caught_by_the_metamorphic_relation(tmp_path: Path):
    """The bug the ``shift_rows`` relation exists for.

    A softmax without max-subtraction is mathematically shift-invariant and only
    breaks once ``exp`` overflows, which needs ``x > 88.7`` in float32 — a regime
    unit-scale inputs never reach. It is the *shifted partner*, drawn at the
    dtype's overflow scale, that exposes it. Overflow is data-dependent, so this
    runs several groups per shape to make the detection a property of the ladder
    rather than of one draw.
    """
    task = TASKS["softmax"]
    table, _ = record(
        "softmax", naive_softmax, tmp_path / "run", n_groups=4 * len(task.domain.shapes)
    )

    declarative = declarative_oracle()
    verdicts = arm_verdicts(declarative, table)
    assert Verdict.FAIL in verdicts.values(), verdicts
    failed = failing_properties(declarative, table)
    assert ShiftInvariance.name in failed, failed

    # Every base case is a plain unit-scale input, on which this kernel is
    # numerically fine — so the detection genuinely comes from the relation and
    # not from the kernel being broken everywhere.
    base_rows = [row for row in table.read() if row.case.relation == "base"]
    assert base_rows
    for row in base_rows:
        assert row.status is Status.OK
        assert np.all(np.isfinite(row.outputs[OUTPUT_NAME]))


# --------------------------------------------------------------------------- #
# D: the relu path
# --------------------------------------------------------------------------- #


def test_relu_uses_only_the_properties_relu_actually_obeys(tmp_path: Path):
    """Why the relu task does not reuse the softmax property set.

    Relu's outputs are unbounded above and its rows do not sum to one, and its
    domain declares no relations, so ``shift_invariance`` can never find a
    partner. Passing ``SOFTMAX_GROUP_PROPERTIES`` here would not raise — it would
    return INCONCLUSIVE forever, adding groups that established nothing to the
    denominator of the detection rate. That silent degradation is asserted below
    so the choice is recorded as a measurement rather than a preference.
    """
    table, _ = record("relu", correct_relu, tmp_path / "run")

    misapplied = DeclarativeOracle(SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES)
    for rows in table.read_groups().values():
        results = misapplied.evaluate(rows)
        shift = [r for r in results if r.property_name == ShiftInvariance.name]
        assert len(shift) == 1
        assert shift[0].verdict is Verdict.INCONCLUSIVE
        assert "missing" in shift[0].detail
    # And the case-level half is not merely inconclusive, it is actively wrong.
    assert ValuesInUnitInterval.name in failing_properties(misapplied, table)

    # The set relu actually gets: one law, which relu really does obey.
    correct_arm = DeclarativeOracle(RELU_CASE_PROPERTIES)
    assert set(arm_verdicts(correct_arm, table).values()) == {Verdict.PASS}


def test_relu_broken_kernel_is_caught_by_its_own_property_set(tmp_path: Path):
    table, _ = record("relu", nan_relu, tmp_path / "run")
    arm = DeclarativeOracle(RELU_CASE_PROPERTIES)
    assert Verdict.FAIL in arm_verdicts(arm, table).values()
    assert failing_properties(arm, table) == {OutputsAreFinite.name}


def test_relu_reference_arm_scores_the_relu_table(tmp_path: Path):
    """The relu rung is a full task, not a softmax variant: it has its own reference.

    Worth its own test because relu's reference returns the input's dtype by
    construction (see ``relu_reference``), and ``ReferenceOracle`` FAILs on a
    shape-or-dtype-driven mismatch without being able to say which side is wrong —
    so a reference that silently promoted would look like a broken kernel on every
    single row.
    """
    clean, _ = record("relu", correct_relu, tmp_path / "clean")
    assert set(arm_verdicts(ReferenceOracle(REFERENCES["relu"]), clean).values()) == {Verdict.PASS}

    broken, _ = record("relu", nan_relu, tmp_path / "broken")
    verdicts = arm_verdicts(ReferenceOracle(REFERENCES["relu"]), broken)
    assert Verdict.FAIL in verdicts.values(), verdicts


# --------------------------------------------------------------------------- #
# Replay proper
# --------------------------------------------------------------------------- #


def test_a_third_oracle_scores_the_table_with_no_re_execution(tmp_path: Path):
    """The architectural payoff: a new arm costs zero hardware.

    The hybrid arm is built *after* the run is on disk, from a reader that never
    saw the generator or the kernel. Both the backend's execution count and the
    kernel's call count are frozen before scoring and must be unchanged after —
    the negative claim needs a counter, not an absence of evidence.
    """
    calls = {"kernel": 0}

    def counted(x: np.ndarray) -> np.ndarray:
        calls["kernel"] += 1
        return correct_softmax(x)

    _, backend = record("softmax", counted, tmp_path / "run")
    executions_after_run = backend.calls
    kernel_calls_after_run = calls["kernel"]
    assert executions_after_run == kernel_calls_after_run > 0

    replayed = ExecutionTable(tmp_path / "run")
    hybrid = HybridOracle(declarative_oracle(), reference_oracle())
    verdicts = arm_verdicts(hybrid, replayed)

    assert verdicts
    assert set(verdicts.values()) == {Verdict.PASS}
    assert backend.calls == executions_after_run
    assert calls["kernel"] == kernel_calls_after_run


def test_hybrid_consults_the_reference_arm_only_when_the_laws_pass(tmp_path: Path):
    """The short-circuit, observed over a replayed table rather than asserted in theory.

    The condition is the declarative arm's verdict on that group, not the kernel's
    overall health: on a broken kernel's degenerate single-column groups the laws
    pass, and there the reference recompute is still paid for. That conditioning
    is exactly what ``HybridOracle``'s docstring warns must not be dropped from a
    cost-per-bug figure, so it is asserted in both directions.
    """
    clean, _ = record("softmax", correct_softmax, tmp_path / "run-clean")
    broken, _ = record("softmax", unnormalized_softmax, tmp_path / "run-broken")

    declarative = declarative_oracle()
    hybrid = HybridOracle(declarative_oracle(), reference_oracle())

    for rows in clean.read_groups().values():
        names = {r.property_name for r in hybrid.evaluate(rows)}
        assert REFERENCE_PROPERTY in names

    skipped = 0
    for rows in broken.read_groups().values():
        laws_failed = summary(declarative.evaluate(rows)) is Verdict.FAIL
        names = {r.property_name for r in hybrid.evaluate(rows)}
        assert (REFERENCE_PROPERTY not in names) is laws_failed
        skipped += laws_failed
    assert skipped, "no group short-circuited; the saving being measured is zero"


def test_reference_arm_length_matches_the_softmax_reduction_axis(tmp_path: Path):
    """B: confirm the default ``n`` is the right one for this corpus.

    ``ReferenceOracle`` derives the accumulation length from ``row.case.shape[-1]``.
    Softmax reduces along the last axis, so the default is correct and no override
    is needed — but "correct by default" is exactly the kind of thing that stays
    true only until a shape changes, so it is pinned rather than assumed.
    """
    table, _ = record("softmax", correct_softmax, tmp_path / "run")
    oracle = reference_oracle()
    assert oracle.n is None
    rows = table.read()
    assert rows
    for row in rows:
        # The recorded case shape is the input's, and the input's last axis is the
        # number of terms the softmax denominator accumulates.
        assert oracle._length(row) == row.case.shape[-1]
        assert row.outputs[OUTPUT_NAME].shape == row.case.shape


def test_every_ladder_shape_is_exercised_and_recorded(tmp_path: Path):
    """The ladder is the recall mechanism; a shape that never runs cannot catch anything."""
    task = TASKS["softmax"]
    table, _ = record("softmax", correct_softmax, tmp_path / "run")
    recorded = {tuple(row.case.shape) for row in table.read()}
    assert set(task.domain.shapes) <= recorded
    assert all(len(shape) == 2 for shape in task.domain.shapes)
