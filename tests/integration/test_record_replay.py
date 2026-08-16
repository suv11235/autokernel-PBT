"""End-to-end record/replay: generate once, execute once, persist, score offline.

This module holds ``REPLAY_FAIRNESS``, the acceptance criterion the whole
batch-first architecture exists to satisfy. The research claim is that three
oracle strategies are compared over *byte-identical* executions; if the choice of
oracle could influence what a later arm scores, the comparison would confound
generation with checking and the headline result would be unfalsifiable.

The guarantee is structural, not aspirational: oracles are handed rows read back
from a persisted table and have no reference to the generator or the backend at
all. These tests assert that the structure actually delivers it, and — via the
saboteur suite at the bottom — that the assertion has the *coverage* to notice
when it does not.

Coverage is the word to hold onto. An earlier version of this module fingerprinted
``case.tensors`` and nothing else, which sounds like the right thing to protect:
"the arms see identical inputs". It is not. The arms score ``outputs``. An arm that
rewrote ``row.outputs['y']`` in place left every input byte untouched, passed the
criterion, and flipped 7 of 9 groups from FAIL to not-FAIL for the arm that ran
after it. The fixed point of that lesson is ``row_fingerprint``: the unit of
fairness is the whole recorded row, not the half of it named "input".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

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
from autokernel_pbt.props.table import ExecutionTable, _json_safe
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
    """A correct softmax, computed the way a real device kernel computes it.

    Deliberately *not* the reference. It accumulates in the input's own float32
    rather than widening to float64, so its result differs from
    ``softmax_reference`` in the last bits and the reference arm's threshold is
    actually exercised — measured test ratios run 0.098 to 0.248 against
    ``DEFAULT_THRESH = 30.0``. A kernel that widened to float64 would be
    bit-identical to the reference, every ratio would be exactly 0, and every
    clean path in this module — this test, the fairness test's reference arm, the
    third-oracle test, the hybrid test's clean half — would pass without the
    comparison having been performed at all.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(x.dtype)


def unnormalized_softmax(x: np.ndarray) -> np.ndarray:
    """Broken: forgets the division by the row sum.

    Chosen as the primary broken kernel because it is deterministic and
    data-independent — ``exp(x - max(x))`` lies in (0, 1] for every input, so
    ``values_in_unit_interval`` passes and ``rows_sum_to_one`` fails on every row
    of every non-degenerate shape. That makes "which arm caught it, and via which
    property" an assertion about the oracles rather than about a lucky draw.
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
) -> tuple[ExecutionTable, CountingBackend, list[ExecutionResult]]:
    """Run the real pipeline once: generate -> execute -> persist.

    Returns a *fresh* reader over the persisted run, never the in-memory results.
    Everything downstream of this function sees only what survived the table,
    which is exactly the constraint the architecture claims to impose on oracles.

    The in-memory ``results`` are returned alongside, and this is not a
    convenience. They are the only witness that lives outside the read path.
    Without one, nothing here can distinguish "the table holds what was executed"
    from "the read path is self-consistent" — a ``read()`` that corrupted every
    row identically on every call is a fixed point, invisible to any number of
    re-reads, and it can destroy every detection in the run.

    Note that the *generated* groups would be a weaker witness than these, not a
    complementary one: ``NumpyBackend`` stores the very same ``Case`` object on
    its result, so comparing generated cases against executed ones compares an
    object with itself. ``results`` carries the outputs too, which is the half
    that matters.

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
    return ExecutionTable(run_dir), backend, results


def fingerprint(tensors: dict[str, np.ndarray]) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """Identity of a tensor set at the byte level.

    ``np.array_equal`` is deliberately not used anywhere in this module. It
    reports True for a float32 array and a float64 array holding the same values,
    and for arrays whose shapes merely broadcast — so an arm silently rescoring a
    promoted copy would pass a fairness test built on it. dtype, shape and
    ``tobytes()`` together admit no such reading.
    """
    return {
        name: (array.dtype.str, tuple(array.shape), array.tobytes())
        for name, array in sorted(tensors.items())
    }


def canonical_telemetry(telemetry: dict[str, Any]) -> Any:
    """Telemetry as it survives the ledger, which is not bitwise.

    ``table.py`` documents that telemetry is JSON-encoded and therefore lossy by
    design: a tuple returns as a list, a non-string key as a string. Comparing raw
    dicts across the persistence boundary would fail on that documented behaviour
    rather than on corruption, so both sides are pushed through the same encoding
    first. This keeps telemetry *witnessed* — an arm that clobbers it is still
    caught — while not asserting a fidelity the ledger never promised.

    ``_json_safe`` is imported from the ledger rather than approximated here, and
    the private name is the point: this must be the *same* encoder, not a
    plausible one. An earlier version passed ``default=str``, which stringifies a
    numpy scalar where ``_json_safe`` calls ``.item()`` on it. A Phase 3 backend
    reporting ``{"sm_occupancy": np.float32(0.75)}`` — precisely the case
    ``_json_safe``'s docstring was written for, and one ``test_table.py``
    certifies as supported — would then have failed this criterion on a run with
    zero corruption, reporting a fairness violation that never happened. A false
    failure here is worse than a missed one: it would be debugged as the thing it
    is not.
    """
    return json.loads(json.dumps(telemetry, sort_keys=True, default=_json_safe))


def row_fingerprint(row: ExecutionResult) -> tuple[Any, ...]:
    """Identity of a whole recorded execution.

    Every field an oracle can read is covered, because every field an oracle can
    read is a field an oracle could corrupt for the next arm. This module has now
    learned that lesson three times, each time by finding a field it had not
    thought of, so the list is enumerated with what each one costs:

    * ``case.metadata()`` — the deadliest and the least obvious. It is not inert
      bookkeeping: ``ShiftInvariance.check_group`` locates the base and its
      partner *by* ``case.relation``, so an arm relabelling one row costs 14/14
      detections on a kernel caught only by shift-invariance, and
      ``ReferenceOracle._length`` reads ``case.shape[-1]``, so rewriting the shape
      moves the pass/fail line — masking real detections or manufacturing false
      ones — for 8/14.
    * ``outputs`` — what the arms actually score. An arm that renormalized
      ``outputs['y']`` in place, a plausible "repair the row" bug, left every
      input byte pristine and flipped 7 of 9 groups from FAIL to not-FAIL.
    * ``status`` and ``error`` — these gate every property's usable/unusable
      branch, so forging them turns detections into INCONCLUSIVE wholesale.
    * ``telemetry`` — what tier-2 properties read in Phase 3, canonicalized
      because the ledger's JSON round trip is documented-lossy.

    KNOWN AND DELIBERATELY UNCOVERED: row *order* within a group. ``rows_fingerprint``
    keys by ``case_id``, so an arm reversing ``rows`` in place is invisible here.
    That is latent rather than live — no property in the current set is
    order-sensitive; base and partner are found by relation, not by index — and
    closing it would mean asserting an ordering the table does not promise. If an
    order-sensitive property is ever added, this is the assertion that must grow.
    """
    return (
        row.case.metadata(),
        fingerprint(row.case.tensors),
        fingerprint(row.outputs),
        row.status,
        row.error,
        canonical_telemetry(row.telemetry),
    )


def rows_fingerprint(groups: dict[str, list[ExecutionResult]]) -> dict[str, tuple[Any, ...]]:
    return {row.case.case_id: row_fingerprint(row) for rows in groups.values() for row in rows}


def table_fingerprint(table: ExecutionTable) -> dict[str, tuple[Any, ...]]:
    return {row.case.case_id: row_fingerprint(row) for row in table.read()}


def executed_rows(results: list[ExecutionResult]) -> dict[str, tuple[Any, ...]]:
    """What the backend actually produced, before persistence touched it."""
    return {result.case.case_id: row_fingerprint(result) for result in results}


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
        group_id: summary(oracle.evaluate(rows)) for group_id, rows in table.read_groups().items()
    }


def failing_properties(oracle: Any, table: ExecutionTable) -> set[str]:
    return {
        result.property_name
        for rows in table.read_groups().values()
        for result in oracle.evaluate(rows)
        if result.verdict is Verdict.FAIL
    }


def group_shapes(table: ExecutionTable) -> dict[str, tuple[int, ...]]:
    return {row.case.group_id: tuple(row.case.shape) for row in table.read()}


# --------------------------------------------------------------------------- #
# REPLAY_FAIRNESS
# --------------------------------------------------------------------------- #


def assert_replay_fairness(run_dir: Path) -> None:
    """The criterion itself, factored out so saboteurs can be run against it.

    Four claims, in increasing strength.

    **Fidelity.** The persisted rows — whole rows, not just their inputs — are
    what the backend executed. This is the only claim with a witness outside the
    read path, and it is what makes the rest non-circular: a ``read()`` that
    corrupted every row identically on every call would satisfy every
    self-comparison below. Witnessing only inputs here is not a weaker version of
    this claim, it is a different and much smaller one: a read path that repaired
    ``outputs`` would leave the inputs untouched and destroy every detection in
    the run.

    **Stability.** The table is a fixed point: reading it twice yields identical
    rows. Stated with dtype, shape and raw bytes rather than value equality —
    ``test_the_fairness_assertion_discriminates`` pins that the comparison can
    tell a perturbed row from an intact one.

    **Shared identity.** The two arms are handed *the same row objects*, not two
    equivalent reads. Fairness is not "the arms saw equal-looking data", it is
    "there was one execution and both arms scored it". Passing the identical
    objects makes divergence impossible by construction rather than by
    comparison, and makes in-memory corruption detectable — a fresh read per arm
    would hide it, since each arm would scribble on its own throwaway copy.

    **Non-interference.** Neither arm perturbs what it was handed, in any field.
    The rows are fingerprinted whole after each arm, and the table is re-read from
    disk at the end to cover an arm reaching past its arguments to the ledger.

    One limit, stated so it is not mistaken for a guarantee: "the two arms were
    handed the same groups" is unreachable here *by construction of this test*,
    which passes one dict to both. The system-level obligation — that a real
    driver does not hand arm A one corpus and arm B another — is inherited by Task
    13, which writes that driver. This test cannot discharge it.
    """
    table, _, results = record("softmax", correct_softmax, run_dir)

    executed = executed_rows(results)
    assert executed, "the recorded run is empty; there is nothing to be fair about"
    assert table_fingerprint(table) == executed, "the persisted rows are not what was executed"

    first = table_fingerprint(table)
    second = table_fingerprint(ExecutionTable(run_dir))
    assert first.keys() == second.keys()
    for case_id in first:
        assert first[case_id] == second[case_id], f"rows differ between reads for {case_id}"

    # One read, shared by both arms. Everything below scores these exact objects.
    shared = table.read_groups()
    before = rows_fingerprint(shared)
    assert before == first, "the shared read differs from the table on disk"

    reference_verdicts = {
        group_id: summary(reference_oracle().evaluate(rows)) for group_id, rows in shared.items()
    }
    assert rows_fingerprint(shared) == before, "the reference arm mutated the rows it was handed"

    declarative_verdicts = {
        group_id: summary(declarative_oracle().evaluate(rows)) for group_id, rows in shared.items()
    }
    assert rows_fingerprint(shared) == before, "the declarative arm mutated the rows it was handed"

    # Both arms must actually have judged every recorded group, or "same rows" is
    # a statement about rows nobody scored.
    assert reference_verdicts.keys() == declarative_verdicts.keys() == shared.keys()
    assert reference_verdicts and declarative_verdicts

    assert table_fingerprint(ExecutionTable(run_dir)) == before, (
        "scoring changed the persisted table"
    )


def test_both_arms_see_byte_identical_inputs(tmp_path: Path):
    """Acceptance criterion REPLAY_FAIRNESS. See ``assert_replay_fairness``."""
    assert_replay_fairness(tmp_path / "run")


def test_the_fairness_assertion_discriminates(tmp_path: Path):
    """Guard that ``fingerprint`` can tell perturbed bytes from intact ones.

    This covers ``fingerprint``'s *discrimination*. The saboteur suite below
    covers its *coverage* — which fields it is applied to — and that is a
    genuinely different property: the original version of this module had perfect
    discrimination over a set of fields that omitted ``outputs``, and so certified
    a run in which one arm rewrote what the next arm scored.
    """
    table, _, _ = record("softmax", correct_softmax, tmp_path / "run")
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
# Saboteurs: does the criterion have the coverage it claims?
# --------------------------------------------------------------------------- #


def _corrupt_input_in_place(row: ExecutionResult) -> None:
    row.case.tensors["x"] = row.case.tensors["x"] * np.float32(1.0000001)


def _promote_input_dtype(row: ExecutionResult) -> None:
    row.case.tensors.update({k: v.astype(np.float64) for k, v in row.case.tensors.items()})


def _corrupt_output_in_place(row: ExecutionResult) -> None:
    if OUTPUT_NAME in row.outputs:
        y = row.outputs[OUTPUT_NAME]
        if y.size:
            writable = np.array(y, copy=True)
            writable.reshape(-1)[0] = np.nextafter(
                writable.reshape(-1)[0], np.array(np.inf, dtype=y.dtype)
            )
            row.outputs[OUTPUT_NAME] = writable


def _repair_outputs(row: ExecutionResult) -> None:
    """The realistic bug: an arm that "helpfully" renormalizes the row it scores.

    This is the one that motivated the whole fix. It leaves every input byte
    untouched, so an input-only fingerprint certifies the run — while a later arm
    scores a normalized ``y`` and its detections evaporate.
    """
    if OUTPUT_NAME in row.outputs:
        y = np.asarray(row.outputs[OUTPUT_NAME], dtype=np.float64)
        total = np.sum(y, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            row.outputs[OUTPUT_NAME] = (y / total).astype(row.outputs[OUTPUT_NAME].dtype)


def _replace_outputs(row: ExecutionResult) -> None:
    row.outputs = {name: np.zeros_like(array) for name, array in row.outputs.items()}


def _clobber_telemetry(row: ExecutionResult) -> None:
    row.telemetry = {"backend": "forged", "wall_ms": 0.0}


def _clobber_status(row: ExecutionResult) -> None:
    row.status = Status.TIMEOUT
    row.error = "forged"


def _relabel_relation(row: ExecutionResult) -> None:
    """Rewrite ``case.relation``. The costliest corruption found so far.

    ``ShiftInvariance.check_group`` finds the base and its partner *by* relation,
    so relabelling one row costs 14/14 detections on a kernel that only that
    property catches — and it touches not a single tensor byte.
    """
    row.case = replace(row.case, relation="base")


def _rewrite_shape(row: ExecutionResult) -> None:
    """Rewrite ``case.shape``, which ``ReferenceOracle._length`` reads.

    The recorded shape sets the ``log2(n)`` divisor of the test ratio. Inflating
    it masks real detections; shrinking it manufactures false ones. Measured cost
    of this one: 8/14.
    """
    rows_, cols = row.case.shape[0], row.case.shape[-1]
    row.case = replace(row.case, shape=(rows_, cols * 2))


def _drift_one_ulp(row: ExecutionResult) -> None:
    x = np.array(row.case.tensors["x"], copy=True)
    if x.size:
        flat = x.reshape(-1)
        flat[0] = np.nextafter(flat[0], np.float32(np.inf))
    row.case.tensors["x"] = x


def _every_read(corrupt: Callable[[ExecutionResult], None]):
    """A read-path saboteur that corrupts identically on every call.

    The fixed-point attacks: no amount of re-reading and comparing can see them,
    so only the executed-rows fidelity witness catches them.
    """

    def factory():
        def apply(rows: list[ExecutionResult]) -> None:
            for row in rows:
                corrupt(row)

        return apply

    return factory


def _from_read(index: int, corrupt: Callable[[ExecutionResult], None]):
    """A read path that corrupts only from the ``index``-th call onward.

    This models a *non-deterministic* read — the realistic shape of a caching bug,
    a partial write, or a racing reader — and it is what pins the criterion's
    remaining assertions. Each of stability, shared-read-equals-disk and the
    post-scoring re-read was individually deletable with zero test failures,
    because every saboteur written before this one was caught by an earlier
    assertion and none modelled a table that changes under you.

    Coupled, deliberately, to the exact read sequence in ``assert_replay_fairness``:
    (1) fidelity, (2) ``first``, (3) ``second``, (4) ``shared``, (5) the final
    re-read. Adding a read there will mis-target these, and they will say so —
    each asserts the *message* it expects, so a drifted index fails loudly rather
    than passing against the wrong assertion.
    """

    def factory():
        state = {"reads": 0}

        def apply(rows: list[ExecutionResult]) -> None:
            state["reads"] += 1
            if state["reads"] < index:
                return
            for row in rows:
                corrupt(row)

        return apply

    return factory


#: Corruptions injected into an *arm*, i.e. one oracle sabotaging the rows the
#: next oracle will score. Every one must break the criterion. Only the first two
#: were caught by the original inputs-only fingerprint; the rest are the fields
#: this module discovered it had forgotten, one review at a time.
ARM_SABOTEURS = {
    "arm_corrupts_input_in_place": _corrupt_input_in_place,
    "arm_promotes_input_dtype": _promote_input_dtype,
    "arm_corrupts_output_in_place": _corrupt_output_in_place,
    "arm_repairs_outputs": _repair_outputs,
    "arm_replaces_outputs": _replace_outputs,
    "arm_clobbers_telemetry": _clobber_telemetry,
    "arm_clobbers_status": _clobber_status,
    "arm_relabels_relation": _relabel_relation,
    "arm_rewrites_shape": _rewrite_shape,
}

#: The arms the criterion actually runs. Saboteurs are injected into *each* in
#: turn, rather than only into the one that happens to run first. Injecting into
#: the first arm alone leaves the second arm's non-interference assertion wholly
#: unexercised — deleting that line kept the suite green — and asserting the arm
#: order instead would pin an incidental detail while still not testing the line.
SABOTAGED_ARMS = {
    "reference": ReferenceOracle,
    "declarative": DeclarativeOracle,
}

#: Corruptions injected into the *read path*, each paired with the assertion that
#: must catch it. Pairing is what makes these pin individual lines: without it a
#: saboteur caught by an earlier assertion silently certifies a later one it never
#: reached, which is how three assertions here stayed deletable through two
#: reviews.
READ_SABOTEURS = {
    "read_drifts_one_ulp": (_every_read(_drift_one_ulp), "not what was executed"),
    "read_promotes_dtype": (_every_read(_promote_input_dtype), "not what was executed"),
    "read_repairs_outputs": (_every_read(_repair_outputs), "not what was executed"),
    "read_replaces_outputs": (_every_read(_replace_outputs), "not what was executed"),
    "read_drifts_after_the_fidelity_check": (
        _from_read(3, _drift_one_ulp),
        "differ between reads",
    ),
    "read_drifts_only_for_the_shared_read": (
        _from_read(4, _drift_one_ulp),
        "shared read differs from the table on disk",
    ),
    "read_drifts_only_on_the_final_reread": (
        _from_read(5, _drift_one_ulp),
        "scoring changed the persisted table",
    ),
}


@pytest.mark.parametrize("arm", sorted(SABOTAGED_ARMS))
@pytest.mark.parametrize("name", sorted(ARM_SABOTEURS))
def test_an_arm_that_corrupts_rows_is_caught(tmp_path: Path, monkeypatch, name: str, arm: str):
    """Each saboteur must break REPLAY_FAIRNESS, from either arm, and be blamed on it.

    Sabotaging the first arm is the higher-damage case — everything it corrupts is
    then scored by the second arm as if it were the recorded execution — but it is
    not the only one that must be caught.

    ``match`` pins *attribution*, not merely detection, and that is what keeps both
    assertions alive. The declarative arm's check runs after both arms over the
    same shared objects, so it subsumes the reference arm's: with a bare
    ``pytest.raises`` the reference assertion was deletable with zero failures —
    the same defect the previous round fixed for the declarative arm, relocated
    rather than closed. Requiring the message to name the sabotaged arm makes each
    assertion the only one that can satisfy its own cases.
    """
    corrupt = ARM_SABOTEURS[name]
    cls = SABOTAGED_ARMS[arm]
    original = cls.evaluate

    def evaluate(self, rows):
        for row in rows:
            corrupt(row)
        return original(self, rows)

    monkeypatch.setattr(cls, "evaluate", evaluate)
    with pytest.raises(AssertionError, match=f"the {arm} arm mutated"):
        assert_replay_fairness(tmp_path / "run")


@pytest.mark.parametrize("name", sorted(READ_SABOTEURS))
def test_a_read_path_that_corrupts_rows_is_caught(tmp_path: Path, monkeypatch, name: str):
    """Each read-path saboteur must break the criterion, at its named assertion.

    The four fixed-point saboteurs are caught by the fidelity witness; the three
    late-onset ones are each caught by exactly one of the assertions that no
    earlier saboteur reached.
    """
    factory, expected = READ_SABOTEURS[name]
    corrupt = factory()
    original = ExecutionTable.read

    def read(self):
        rows = original(self)
        corrupt(rows)
        return rows

    monkeypatch.setattr(ExecutionTable, "read", read)
    with pytest.raises(AssertionError, match=expected):
        assert_replay_fairness(tmp_path / "run")


def test_numpy_typed_telemetry_does_not_manufacture_a_fairness_violation(
    tmp_path: Path, monkeypatch
):
    """A clean run must stay clean when a backend reports numpy-typed telemetry.

    ``table.py`` supports this deliberately — ``_json_safe`` exists for it and
    ``test_table.py`` certifies it — so a Phase 3 backend reporting an
    ``np.float32`` occupancy counter is a *correct* backend. If this module's
    telemetry canonicalizer disagreed with the ledger's encoder, that correct
    backend would fail REPLAY_FAIRNESS with no corruption anywhere, and the
    failure would be investigated as a fairness violation that never happened.
    """
    original = NumpyBackend._telemetry

    def telemetry(self, start):
        recorded = original(self, start)
        recorded["sm_occupancy"] = np.float32(0.75)
        return recorded

    monkeypatch.setattr(NumpyBackend, "_telemetry", telemetry)
    assert_replay_fairness(tmp_path / "run")


def test_a_repairing_arm_would_actually_destroy_detections(tmp_path: Path):
    """Why the saboteur suite is not paranoia: the damage, measured.

    ``_repair_outputs`` is scored against a genuinely broken kernel here. Without
    the criterion above catching it, an arm running first would silently convert
    the arm running second into one that finds nothing — and the comparison the
    project reports would be between a real oracle and a sabotaged one.
    """
    table, _, _ = record("softmax", unnormalized_softmax, tmp_path / "run")
    declarative = declarative_oracle()

    honest = arm_verdicts(declarative, table)
    honest_failures = {gid for gid, v in honest.items() if v is Verdict.FAIL}
    assert honest_failures, "the broken kernel must be detected before sabotage"

    repaired = table.read_groups()
    for rows in repaired.values():
        for row in rows:
            _repair_outputs(row)
    after = {gid: summary(declarative.evaluate(rows)) for gid, rows in repaired.items()}
    still_failing = {gid for gid, v in after.items() if v is Verdict.FAIL}

    assert still_failing < honest_failures, (
        "the repair must destroy detections to be worth guarding"
    )


# --------------------------------------------------------------------------- #
# A correct kernel, a broken kernel
# --------------------------------------------------------------------------- #


def test_correct_kernel_passes_both_arms(tmp_path: Path):
    table, _, _ = record("softmax", correct_softmax, tmp_path / "run")

    reference = arm_verdicts(reference_oracle(), table)
    declarative = arm_verdicts(declarative_oracle(), table)

    assert reference and declarative
    assert set(reference.values()) == {Verdict.PASS}, reference
    assert set(declarative.values()) == {Verdict.PASS}, declarative


def test_the_reference_arm_actually_exercises_its_threshold(tmp_path: Path):
    """A passing reference arm must be passing on a real comparison, not on ratio 0.

    If the kernel under test were bit-identical to the reference, every ratio
    would be exactly 0 and every clean reference-arm path in this module would be
    vacuous — passing without the threshold ever having been consulted. So every
    non-degenerate row is required to produce a strictly positive ratio, well
    inside the threshold (measured: 0.033 to 0.248 against 30.0).

    Single-column rows are exempt and, once again, for a real reason: softmax of
    an ``(m, 1)`` input is exactly 1.0 in any precision, so kernel and reference
    agree bitwise there and a zero ratio is the correct answer rather than a sign
    of vacuity.
    """
    table, _, _ = record("softmax", correct_softmax, tmp_path / "run")
    oracle = reference_oracle()
    ratios = {}
    for rows in table.read_groups().values():
        for row, result in zip(rows, oracle.evaluate(rows)):
            assert result.verdict is Verdict.PASS
            assert result.detail.startswith("ratio=")
            ratios[row.case.case_id] = (
                tuple(row.case.shape),
                float(result.detail.removeprefix("ratio=")),
            )
    assert ratios
    discriminating = [r for shape, r in ratios.values() if shape[-1] > 1]
    assert discriminating
    assert min(discriminating) > 0.0, (
        "the kernel is bit-identical to the reference; the arm's threshold is never consulted"
    )
    assert max(discriminating) < oracle.thresh


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
    by both arms. See ``_LADDER_SHAPES`` for what that costs the reported numbers.
    """
    table, _, _ = record("softmax", unnormalized_softmax, tmp_path / "run")
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
    dtype's overflow scale, that exposes it.

    The seed is fixed, so this is deterministic rather than sampled. The margin is
    also wide: the measured per-row corruption rate is 3.34% over the 248 rows run
    here, so P(no detection) is about 2e-4, and 60 of 60 trial seeds detect.
    """
    task = TASKS["softmax"]
    table, _, _ = record(
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
    table, _, _ = record("relu", correct_relu, tmp_path / "run")

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
    table, _, _ = record("relu", nan_relu, tmp_path / "run")
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
    clean, _, _ = record("relu", correct_relu, tmp_path / "clean")
    assert set(arm_verdicts(ReferenceOracle(REFERENCES["relu"]), clean).values()) == {Verdict.PASS}

    broken, _, _ = record("relu", nan_relu, tmp_path / "broken")
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

    _, backend, _ = record("softmax", counted, tmp_path / "run")
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
    clean, _, _ = record("softmax", correct_softmax, tmp_path / "run-clean")
    broken, _, _ = record("softmax", unnormalized_softmax, tmp_path / "run-broken")

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
    table, _, _ = record("softmax", correct_softmax, tmp_path / "run")
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
    table, _, _ = record("softmax", correct_softmax, tmp_path / "run")
    recorded = {tuple(row.case.shape) for row in table.read()}
    assert set(task.domain.shapes) <= recorded
    assert all(len(shape) == 2 for shape in task.domain.shapes)
