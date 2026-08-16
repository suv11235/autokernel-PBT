"""ScoreTable round-trip tests.

The score table is the only path by which a verdict outlives the process that
produced it. Every test here is about one of three things: that a result comes
back as the *same* object it went in as (not a lookalike string), that it still
says which arm produced it and what that arm cost, and that it still names the
execution row and the case *group* it judged.

Two things every persisted row carries, and both are load-bearing:

* ``group_id``, always — the case group is the unit at which arms are comparable
  (see ``ScoreTable._record``), so recording it makes the correct rollup a
  ``GROUP BY`` needing no join. ``case_id`` refines it and is ``""`` on a
  group-scoped verdict.
* ``corpus_fingerprint`` — which execution table these verdicts are about. It is
  passed in rather than defaulted, so the tests here name one explicitly.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autokernel_pbt.props.backends.base import ExecutionResult, Status
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.scores import (
    SCHEMA,
    SCORES_FILE,
    ArmScores,
    ScoreTable,
)
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.verdict import TIER_BACKEND, TIER_PORTABLE, PropertyResult, Verdict

#: A stand-in corpus identity. Opaque by design: nothing here may depend on its
#: shape, only on whether two files carry the same one.
CORPUS = "corpus-fingerprint-0000"


def _write(run, arms) -> None:
    """Every write in this module goes through one place.

    ``corpus_fingerprint`` is a required keyword on ``ScoreTable.write`` — a default
    would let a caller that forgot it produce a scores file that pairs with any
    execution table, which is the hole the column exists to close — and repeating it
    at 36 call sites would say nothing 36 times.
    """
    ScoreTable(run).write(arms, corpus_fingerprint=CORPUS)


def _case_result(
    case_id: str = "c0",
    *,
    group_id: str = "g0",
    property_name: str = "finite_outputs",
    verdict: Verdict = Verdict.PASS,
    tier: int = TIER_PORTABLE,
    tolerance_free: bool = True,
    detail: str = "",
) -> PropertyResult:
    """A case-scoped verdict, carrying *both* keys.

    This is what the driver persists: the oracles emit exactly one of the two (the
    hybrid arm's split point depends on it), and the driver stamps the group before
    writing. ``group_id`` defaults here because a persisted row without one is
    refused — the unit of analysis is not optional.
    """
    return PropertyResult(
        property_name=property_name,
        tier=tier,
        tolerance_free=tolerance_free,
        verdict=verdict,
        detail=detail,
        case_id=case_id,
        group_id=group_id,
    )


def _group_result(group_id: str = "g0", **kwargs) -> PropertyResult:
    fields = {
        "property_name": "shift_invariance",
        "tier": TIER_PORTABLE,
        "tolerance_free": True,
        "verdict": Verdict.FAIL,
        "detail": "residual 1e-2",
    }
    fields.update(kwargs)
    return PropertyResult(group_id=group_id, **fields)


def _patch_column(run, column: str, values, type_) -> None:
    """Rewrite one column of a written score file.

    Every use of this forges a file `write` refuses to emit — that is the point:
    the read-side guards exist for foreign, hand-edited, or older-build files,
    which is the only way those shapes can reach a reader.
    """
    table = pq.read_table(run / SCORES_FILE)
    patched = table.set_column(
        table.schema.get_field_index(column), column, pa.array(values, type=type_)
    )
    pq.write_table(patched, run / SCORES_FILE)


def _execution(case_id: str, group_id: str = "g0", relation: str = "base") -> ExecutionResult:
    case = Case(
        case_id=case_id,
        group_id=group_id,
        relation=relation,
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.full((2, 3), 0.5, dtype=np.float32)},
    )
    return ExecutionResult(
        case=case,
        outputs={"y": np.full((2, 3), 0.25, dtype=np.float32)},
        telemetry={"backend": "numpy"},
        status=Status.OK,
        kernel_id="softmax_missing_max_subtraction",
        kernel_is_broken=True,
    )


# --- Criterion SCORES_PERSIST -------------------------------------------------


def test_scores_round_trip_and_rejoin_their_rows(tmp_path):
    """Scores must survive persistence *and* still join to the executions they judged.

    The join is the whole point of carrying `case_id`/`group_id`: a verdict that
    cannot be traced back to a recorded row cannot contribute to a detection rate,
    because nothing connects it to the ground truth on that row. So this reads
    *both* tables back from disk and joins them, rather than comparing scores to
    the in-memory objects that produced them.
    """
    run = tmp_path / "run"
    executions = [
        _execution("softmax-g00000-base"),
        _execution("softmax-g00000-base::shift_rows", relation="shift_rows"),
        _execution("softmax-g00001-base", group_id="g1"),
    ]
    ExecutionTable(run).write(executions)

    scored = ArmScores(
        arm="reference",
        elapsed_s=0.25,
        results=[
            _case_result("softmax-g00000-base"),
            _case_result("softmax-g00000-base::shift_rows", verdict=Verdict.FAIL),
            _case_result("softmax-g00001-base"),
            _group_result("g0"),
            _group_result("g1", verdict=Verdict.PASS),
        ],
    )
    _write(run, [scored])

    rows = ExecutionTable(run).read()
    arms = ScoreTable(run).read()

    recorded_cases = {row.case.case_id for row in rows}
    recorded_groups = {row.case.group_id for row in rows}
    assert len(arms) == 1
    assert len(arms[0].results) == 5

    for result in arms[0].results:
        if result.case_id:
            assert result.case_id in recorded_cases
        else:
            assert result.group_id in recorded_groups

    # The join is not vacuous: both keys are actually exercised, and the joined
    # rows carry the ground truth a detection rate needs.
    joined = {
        result.case_id: next(r for r in rows if r.case.case_id == result.case_id)
        for result in arms[0].results
        if result.case_id
    }
    assert len(joined) == 3
    assert all(row.kernel_is_broken is True for row in joined.values())
    assert {r.group_id for r in arms[0].results if r.group_id} == {"g0", "g1"}


# --- Criterion ARM_ATTRIBUTION ------------------------------------------------


def test_each_result_records_its_arm(tmp_path):
    """Every persisted row says which arm produced it, even when arms interleave.

    Arm identity is what makes an arm-vs-arm comparison possible at all; a verdict
    with no arm is a number with no owner.
    """
    run = tmp_path / "run"
    _write(run,
        [
            ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")]),
            ArmScores(
                arm="declarative",
                elapsed_s=2.0,
                results=[_case_result("c0"), _group_result("g0")],
            ),
        ]
    )

    arms = ScoreTable(run).read()
    assert [arm.arm for arm in arms] == ["reference", "declarative"]
    assert [len(arm.results) for arm in arms] == [1, 2]

    # And the column is genuinely per-row, not an artifact of the grouping.
    table = pq.read_table(run / SCORES_FILE)
    assert table.column("arm").to_pylist() == ["reference", "declarative", "declarative"]


# --- Criterion ORACLE_COST_RECORDED -------------------------------------------


def test_arm_elapsed_is_recorded(tmp_path):
    """Cost-per-bug is bugs-caught over seconds-spent; without the seconds it is not
    derivable from the artifacts at all."""
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="hybrid", elapsed_s=12.5, results=[_case_result("c0")])]
    )

    arm = ScoreTable(run).read()[0]
    assert arm.elapsed_s == 12.5
    assert isinstance(arm.elapsed_s, float)


# --- Requirement B: round-trip fidelity of each field type --------------------


def test_verdict_round_trips_as_the_enum_member(tmp_path):
    """`Verdict` subclasses `str`, so a bare string compares equal to the member.

    A naive `== Verdict.FAIL` therefore passes on a plain `str`, while every
    identity-based dispatch downstream (`is`, `match`) silently never matches.
    """
    run = tmp_path / "run"
    results = [
        _case_result("c0", verdict=Verdict.PASS),
        _case_result("c1", verdict=Verdict.FAIL),
        _case_result("c2", verdict=Verdict.INCONCLUSIVE),
    ]
    _write(run, [ArmScores(arm="reference", elapsed_s=0.5, results=results)])

    read_back = ScoreTable(run).read()[0].results
    assert [r.verdict for r in read_back] == [
        Verdict.PASS,
        Verdict.FAIL,
        Verdict.INCONCLUSIVE,
    ]
    for result in read_back:
        assert isinstance(result.verdict, Verdict)
    assert read_back[0].verdict is Verdict.PASS
    assert read_back[1].verdict is Verdict.FAIL
    assert read_back[2].verdict is Verdict.INCONCLUSIVE


def test_unknown_verdict_names_the_score_file(tmp_path):
    """A table written by a build with a different verdict vocabulary must refuse to
    read, and must say which file is stale."""
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=0.5, results=[_case_result("c0")])]
    )
    _patch_column(run, "verdict", ["skipped"], pa.string())

    with pytest.raises(ValueError, match=SCORES_FILE):
        ScoreTable(run).read()


def test_tier_and_tolerance_free_keep_their_types(tmp_path):
    """`tier` is an int and `tolerance_free` a bool; a tier that came back as a float
    or a flag that came back as a string would split the tolerance-free report."""
    run = tmp_path / "run"
    _write(run,
        [
            ArmScores(
                arm="reference",
                elapsed_s=0.5,
                results=[
                    _case_result("c0", tier=TIER_PORTABLE, tolerance_free=True),
                    _case_result("c1", tier=TIER_BACKEND, tolerance_free=False),
                ],
            )
        ]
    )

    portable, backend = ScoreTable(run).read()[0].results
    assert portable.tier == TIER_PORTABLE
    assert backend.tier == TIER_BACKEND
    assert type(portable.tier) is int
    assert portable.tolerance_free is True
    assert backend.tolerance_free is False


def test_every_field_of_a_result_round_trips(tmp_path):
    """Whole-object equality, so a field added later cannot quietly stop persisting."""
    run = tmp_path / "run"
    original = [
        _case_result(
            "c0",
            property_name="finite_outputs",
            verdict=Verdict.INCONCLUSIVE,
            tier=TIER_BACKEND,
            tolerance_free=False,
            detail="dtype int32 has no tolerance",
        ),
        _group_result("g0"),
    ]
    _write(run, [ArmScores(arm="declarative", elapsed_s=3.5, results=original)])
    assert ScoreTable(run).read()[0].results == original


def test_elapsed_s_survives_as_an_exact_float(tmp_path):
    """float64 in, float64 out: a value that took a decimal detour would perturb
    every cost-per-bug figure derived from it."""
    run = tmp_path / "run"
    elapsed = 0.1 + 0.2  # 0.30000000000000004
    _write(run,
        [ArmScores(arm="reference", elapsed_s=elapsed, results=[_case_result("c0")])]
    )
    assert ScoreTable(run).read()[0].elapsed_s == elapsed


def test_an_int_elapsed_reads_back_as_a_float(tmp_path):
    """A caller passing `elapsed_s=1` must not widen the declared float64 column."""
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=1, results=[_case_result("c0")])]
    )
    assert ScoreTable(run).read()[0].elapsed_s == 1.0
    assert isinstance(ScoreTable(run).read()[0].elapsed_s, float)


# --- Requirement E: several arms in one file ----------------------------------


def test_two_arms_keep_their_own_elapsed_and_results(tmp_path):
    """One file, two arms: the reassembly must not smear timings across arms."""
    run = tmp_path / "run"
    _write(run,
        [
            ArmScores(
                arm="reference",
                elapsed_s=1.25,
                results=[_case_result("c0"), _case_result("c1")],
            ),
            ArmScores(arm="declarative", elapsed_s=7.5, results=[_group_result("g0")]),
        ]
    )

    reference, declarative = ScoreTable(run).read()
    assert reference.arm == "reference"
    assert reference.elapsed_s == 1.25
    assert [r.case_id for r in reference.results] == ["c0", "c1"]
    assert declarative.arm == "declarative"
    assert declarative.elapsed_s == 7.5
    assert [r.group_id for r in declarative.results] == ["g0"]


def test_a_repeated_arm_name_is_rejected(tmp_path):
    """Two `ArmScores` with one name would reassemble as a single arm on read, with
    one of the two `elapsed_s` values silently discarded — a cost-per-bug figure
    computed against half the work that produced it."""
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="duplicate arm names"):
        _write(run,
            [
                ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")]),
                ArmScores(arm="reference", elapsed_s=2.0, results=[_case_result("c1")]),
            ]
        )


def test_a_repeated_arm_name_is_rejected_before_anything_is_written(tmp_path):
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])]
    )
    with pytest.raises(ValueError, match="duplicate arm names"):
        _write(run,
            [
                ArmScores(arm="declarative", elapsed_s=1.0, results=[_case_result("c0")]),
                ArmScores(arm="declarative", elapsed_s=2.0, results=[_case_result("c1")]),
            ]
        )
    assert [arm.arm for arm in ScoreTable(run).read()] == ["reference"]


def test_a_foreign_file_with_conflicting_elapsed_for_one_arm_is_refused(tmp_path):
    """`write` cannot emit this, but a hand-built or foreign file can: two rows of one
    arm disagreeing about how long that arm took. Picking the first would invent a
    cost figure, so the read refuses."""
    run = tmp_path / "run"
    _write(run,
        [
            ArmScores(
                arm="reference",
                elapsed_s=1.0,
                results=[_case_result("c0"), _case_result("c1")],
            )
        ]
    )
    _patch_column(run, "elapsed_s", [1.0, 2.0], pa.float64())

    with pytest.raises(ValueError, match="disagreeing elapsed_s"):
        ScoreTable(run).read()


# --- elapsed_s must be usable as a denominator --------------------------------


def test_a_nan_elapsed_is_rejected(tmp_path):
    """A nan writes cleanly and then makes the file permanently unreadable.

    `nan != nan`, so the read-side agreement check trips on the arm's *second*
    row and blames a hand-edit that never happened. It is also not a cost figure.
    """
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="finite, non-negative"):
        _write(run,
            [
                ArmScores(
                    arm="reference",
                    elapsed_s=float("nan"),
                    results=[_case_result("c0"), _case_result("c1")],
                )
            ]
        )


def test_an_infinite_elapsed_is_rejected(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="finite, non-negative"):
        _write(run,
            [ArmScores(arm="reference", elapsed_s=float("inf"), results=[_case_result("c0")])]
        )


def test_a_negative_elapsed_is_rejected(tmp_path):
    """A subtraction the wrong way round; it would poison cost-per-bug silently."""
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="finite, non-negative"):
        _write(run,
            [ArmScores(arm="reference", elapsed_s=-5.0, results=[_case_result("c0")])]
        )


def test_a_zero_elapsed_is_accepted(tmp_path):
    """Zero is a measurement, not a placeholder: a trivially fast arm on a coarse
    clock reports it honestly, and rejecting it would fail a legitimate run."""
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=0.0, results=[_case_result("c0")])]
    )
    assert ScoreTable(run).read()[0].elapsed_s == 0.0


# --- Rejections at write ------------------------------------------------------


def test_an_arm_with_no_results_is_rejected(tmp_path):
    """An arm that judged nothing would still enter the denominator of every rate
    computed per arm, and would still charge its `elapsed_s` against zero verdicts."""
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="no results"):
        _write(run, [ArmScores(arm="reference", elapsed_s=1.0)])


def test_an_arm_with_no_results_is_rejected_by_name(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="'declarative'"):
        _write(run,
            [
                ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")]),
                ArmScores(arm="declarative", elapsed_s=1.0),
            ]
        )


def test_a_result_with_neither_case_id_nor_group_id_is_rejected(tmp_path):
    """An orphaned verdict cannot be rejoined to anything, so it can never be
    attributed to a kernel — it would inflate a numerator with no denominator."""
    run = tmp_path / "run"
    orphan = PropertyResult(
        property_name="finite_outputs",
        tier=TIER_PORTABLE,
        tolerance_free=True,
        verdict=Verdict.PASS,
    )
    with pytest.raises(ValueError, match="carries no group_id"):
        _write(run,
            [ArmScores(arm="reference", elapsed_s=1.0, results=[orphan])]
        )


def test_a_case_scoped_result_without_its_group_is_rejected(tmp_path):
    """A verdict that names a case but not its group cannot be rolled up.

    It is not orphaned — it joins to one execution — but the case group is the unit
    at which arms are comparable, and recovering the group would require the reader
    to join back to `rows.parquet` with knowledge no column supplies. This is the
    write the driver's ``_keyed_by_group`` exists to prevent.
    """
    run = tmp_path / "run"
    unkeyed = PropertyResult(
        property_name="finite_outputs",
        tier=TIER_PORTABLE,
        tolerance_free=True,
        verdict=Verdict.PASS,
        case_id="c0",
    )
    with pytest.raises(ValueError, match="carries no group_id"):
        _write(run,
            [ArmScores(arm="reference", elapsed_s=1.0, results=[unkeyed])]
        )


def test_both_keys_together_are_the_normal_case_scoped_row(tmp_path):
    """The counterpart of the two rejections above, and the reason they are narrow.

    A row carrying both keys is not ambiguous, it is the *ordinary* one: `group_id`
    is the unit of analysis and `case_id` refines it to the execution judged. If
    this were rejected, no case-scoped verdict could ever be rolled up without a
    join, which is the whole point of the column.
    """
    run = tmp_path / "run"
    both = PropertyResult(
        property_name="rows_sum_to_one",
        tier=TIER_PORTABLE,
        tolerance_free=True,
        verdict=Verdict.FAIL,
        case_id="c0",
        group_id="g0",
    )
    _write(run, [ArmScores(arm="declarative", elapsed_s=1.0, results=[both])])

    (read_back,) = ScoreTable(run).read()[0].results
    assert read_back == both
    assert read_back.case_id == "c0"
    assert read_back.group_id == "g0"


def test_an_attribution_failure_names_the_arm_and_the_property(tmp_path):
    run = tmp_path / "run"
    orphan = PropertyResult(
        property_name="finite_outputs",
        tier=TIER_PORTABLE,
        tolerance_free=True,
        verdict=Verdict.PASS,
    )
    with pytest.raises(ValueError, match=r"'declarative'.*'finite_outputs'"):
        _write(run,
            [ArmScores(arm="declarative", elapsed_s=1.0, results=[orphan])]
        )


def test_a_rejected_write_leaves_the_previous_table_intact(tmp_path):
    """Every write-side guard fires before the file is touched, so a bad call costs
    nothing already recorded."""
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])]
    )
    with pytest.raises(ValueError, match="no results"):
        _write(run, [ArmScores(arm="declarative", elapsed_s=2.0)])

    arms = ScoreTable(run).read()
    assert [(arm.arm, arm.elapsed_s) for arm in arms] == [("reference", 1.0)]


# --- Empty and missing runs ---------------------------------------------------


def test_read_on_missing_run_returns_empty(tmp_path):
    assert ScoreTable(tmp_path / "nope").read() == []


def test_write_of_no_arms_produces_a_readable_empty_table(tmp_path):
    run = tmp_path / "run"
    _write(run, [])
    assert (run / SCORES_FILE).exists()
    assert ScoreTable(run).read() == []


def test_rewrite_replaces_the_previous_scores(tmp_path):
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])]
    )
    _write(run,
        [ArmScores(arm="declarative", elapsed_s=2.0, results=[_case_result("c1")])]
    )
    assert [arm.arm for arm in ScoreTable(run).read()] == ["declarative"]


def test_write_leaves_no_temporary_file_behind(tmp_path):
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])]
    )
    assert {p.name for p in run.iterdir()} == {SCORES_FILE}


def test_scores_live_beside_the_execution_table_in_one_run_dir(tmp_path):
    """The two tables share a run directory; neither write may disturb the other."""
    run = tmp_path / "run"
    ExecutionTable(run).write([_execution("c0")])
    _write(run,
        [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])]
    )
    assert [r.case.case_id for r in ExecutionTable(run).read()] == ["c0"]
    assert [a.arm for a in ScoreTable(run).read()] == ["reference"]


# --- Schema conformance and its wiring ----------------------------------------


def test_read_rejects_a_table_written_by_a_narrower_build(tmp_path):
    """A missing column must be refused by name, not surface as a bare KeyError from
    inside the row loop, which names neither the run nor the cause."""
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])]
    )
    narrowed = pq.read_table(run / SCORES_FILE).drop_columns(["arm", "elapsed_s"])
    pq.write_table(narrowed, run / SCORES_FILE)

    with pytest.raises(ValueError, match=r"different schema.*arm.*elapsed_s"):
        ScoreTable(run).read()


def test_read_rejects_a_row_that_lost_its_group(tmp_path):
    """A row with no `group_id` cannot be rolled up at the unit arms are compared at.

    `write` cannot emit this; a foreign, hand-edited or older-build file can, and it
    would silently drop a group from a rate rather than raise. `read` already
    re-checks the verdict vocabulary and the column set; this invariant is equally
    visible in the values.
    """
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])]
    )
    _patch_column(run, "group_id", [None], pa.string())

    with pytest.raises(ValueError, match="must name the group it judged"):
        ScoreTable(run).read()


def test_read_rejects_a_row_with_a_null_case_id(tmp_path):
    """A null `case_id` rebuilds as `PropertyResult(case_id=None)`.

    `""` is the legitimate value on a group-scoped verdict, so the check is against
    null specifically: a None in a column declared `str` makes a downstream
    `.startswith` an AttributeError and sorts apart from `""`.
    """
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])]
    )
    _patch_column(run, "case_id", [None], pa.string())

    with pytest.raises(ValueError, match="null case_id"):
        ScoreTable(run).read()


def test_a_join_key_failure_on_read_names_the_score_file(tmp_path):
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])]
    )
    _patch_column(run, "case_id", [None], pa.string())

    with pytest.raises(ValueError, match=SCORES_FILE):
        ScoreTable(run).read()


def test_an_unknown_tier_names_the_score_file(tmp_path):
    """`PropertyResult.__post_init__` rejects the tier but names no file, which is
    inconsistent with every other refusal on this read path."""
    run = tmp_path / "run"
    _write(run,
        [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])]
    )
    _patch_column(run, "tier", [7], pa.int64())

    with pytest.raises(ValueError, match=rf"tier must be in.*{SCORES_FILE}"):
        ScoreTable(run).read()


def test_the_builder_actually_applies_the_schema_check(tmp_path, monkeypatch):
    """`_record` must route through `_conform`, not merely coexist with it.

    Deleting the call keeps every other test green, because every record the
    current builder produces is well-formed — the failure only appears once the
    schema gains a column the builder was never taught. That is the mutation
    applied here, and it is exactly the mistake `_conform` exists to catch.
    """
    widened = pa.schema([*SCHEMA, pa.field("run_id", pa.string())])
    monkeypatch.setattr("autokernel_pbt.props.scores.SCHEMA", widened)

    with pytest.raises(ValueError, match=r"missing: \['run_id'\]"):
        _write(tmp_path / "run",
            [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])]
        )


def test_the_corpus_fingerprint_is_stamped_on_every_row(tmp_path):
    """Which execution table these verdicts are about, on every row.

    Alone it says nothing; paired with `rows.parquet` it is the only thing that can
    tell a scores file about *this* run from an identically-shaped one about another.
    """
    run = tmp_path / "run"
    _write(run, [ArmScores(arm="reference", elapsed_s=1.0, results=[_case_result("c0")])])

    stamped = pq.read_table(run / SCORES_FILE).column("corpus_fingerprint").to_pylist()
    assert stamped == [CORPUS]
    assert ScoreTable(run).corpus_fingerprint() == CORPUS


def test_a_scores_file_holding_two_corpora_is_refused(tmp_path):
    """Two fingerprints in one file means scores from two runs concatenated.

    `write` broadcasts one value, so this is unreachable from here — but a file
    assembled by hand, or by a future multi-run tool, would otherwise pair happily
    with whichever table it was asked about.
    """
    run = tmp_path / "run"
    _write(
        run,
        [
            ArmScores(
                arm="reference",
                elapsed_s=1.0,
                results=[_case_result("c0"), _case_result("c1")],
            )
        ],
    )
    _patch_column(run, "corpus_fingerprint", ["aaa", "bbb"], pa.string())

    with pytest.raises(ValueError, match="different corpus"):
        ScoreTable(run).corpus_fingerprint()


def test_a_run_with_no_scores_has_no_corpus_fingerprint(tmp_path):
    assert ScoreTable(tmp_path / "nope").corpus_fingerprint() == ""
    empty = tmp_path / "empty"
    _write(empty, [])
    assert ScoreTable(empty).corpus_fingerprint() == ""


def test_the_schema_check_is_the_shared_one_from_the_execution_table():
    """One `_conform`, not two: a second copy would drift from the first."""
    from autokernel_pbt.props import scores, table

    assert scores._conform is table._conform
