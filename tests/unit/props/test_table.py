"""ExecutionTable round-trip tests."""

import itertools
import re
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from autokernel_pbt.props.backends.base import ExecutionResult, Status
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.table import SCHEMA, ExecutionTable, _conform, payload_filename


def _result(case_id: str, group_id: str = "g0", relation: str = "base") -> ExecutionResult:
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
        telemetry={"backend": "numpy", "wall_ms": 1.5},
        status=Status.OK,
    )


def _tagged(case_id: str, tag: float) -> ExecutionResult:
    """A row whose tensor value and telemetry carry the same tag.

    Any row read back with `outputs["y"] != telemetry["wall_ms"]` is a payload
    paired with metadata from a different write — the torn-table failure.
    """
    result = _result(case_id)
    result.outputs = {"y": np.full((2, 3), tag, dtype=np.float32)}
    result.telemetry = {"backend": "numpy", "wall_ms": tag}
    return result


def _observed(run_dir) -> list[tuple[str, float, float]]:
    return [
        (r.case.case_id, float(r.outputs["y"][0, 0]), r.telemetry["wall_ms"])
        for r in ExecutionTable(run_dir).read()
    ]


def _assert_identical(actual: np.ndarray, expected: np.ndarray) -> None:
    """Bitwise identity, not `np.array_equal`.

    `np.array_equal` is True for a float32 array and a float64 array holding the
    same values, so it would not catch a writer that silently upcasts. Comparing
    raw bytes plus dtype plus shape is what "survives persistence bitwise"
    actually means, and is what criterion TABLE_ROUND_TRIP is asserting.
    """
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_round_trip_preserves_tensors_bitwise(tmp_path):
    table = ExecutionTable(tmp_path / "run1")
    table.write([_result("c0")])
    rows = ExecutionTable(tmp_path / "run1").read()
    _assert_identical(rows[0].outputs["y"], np.full((2, 3), 0.25, dtype=np.float32))
    _assert_identical(rows[0].case.tensors["x"], np.full((2, 3), 0.5, dtype=np.float32))


def test_round_trip_preserves_irrational_bit_patterns(tmp_path):
    """Values whose bits do not survive a decimal detour, plus nan/inf/-0.0."""
    case = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="softmax",
        dtype="float32",
        shape=(6,),
        tensors={
            "x": np.array(
                [np.pi, np.e, np.nan, np.inf, -np.inf, -0.0], dtype=np.float32
            )
        },
    )
    result = ExecutionResult(case=case, outputs={"y": case.tensors["x"].astype(np.float64)})
    ExecutionTable(tmp_path / "run1").write([result])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    _assert_identical(row.case.tensors["x"], case.tensors["x"])
    _assert_identical(row.outputs["y"], case.tensors["x"].astype(np.float64))


def test_round_trip_preserves_every_persistable_dtype_kind(tmp_path):
    """`PERSISTABLE_KINDS` is "biuf"; the domain also allows float16."""
    tensors = {
        "f16": np.array([1.5, np.nan, -0.0], dtype=np.float16),
        "f64": np.array([np.pi], dtype=np.float64),
        "i64": np.array([-(2**62)], dtype=np.int64),
        "u8": np.array([255], dtype=np.uint8),
        "b": np.array([True, False], dtype=bool),
    }
    case = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="t",
        dtype="float16",
        shape=(3,),
        tensors=tensors,
    )
    ExecutionTable(tmp_path / "run1").write([ExecutionResult(case=case)])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    for name, array in tensors.items():
        _assert_identical(row.case.tensors[name], array)


def test_round_trip_preserves_metadata(tmp_path):
    table = ExecutionTable(tmp_path / "run1")
    table.write([_result("c0")])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.case.case_id == "c0"
    assert row.case.group_id == "g0"
    assert row.case.shape == (2, 3)
    assert row.telemetry["backend"] == "numpy"
    assert row.status == "ok"


def test_grouping_reassembles_case_groups(tmp_path):
    table = ExecutionTable(tmp_path / "run1")
    table.write([_result("c0"), _result("c1", relation="shift_rows")])
    groups = ExecutionTable(tmp_path / "run1").read_groups()
    assert list(groups) == ["g0"]
    assert {r.case.relation for r in groups["g0"]} == {"base", "shift_rows"}


def test_grouping_preserves_write_order(tmp_path):
    results = [
        _result("c0", group_id="g1"),
        _result("c1", group_id="g0"),
        _result("c2", group_id="g1", relation="shift_rows"),
    ]
    ExecutionTable(tmp_path / "run1").write(results)
    groups = ExecutionTable(tmp_path / "run1").read_groups()
    assert list(groups) == ["g1", "g0"]
    assert [r.case.case_id for r in groups["g1"]] == ["c0", "c2"]


def test_failed_execution_round_trips(tmp_path):
    failed = _result("c0")
    failed.status = "launch_error"
    failed.error = "boom"
    failed.outputs = {}
    ExecutionTable(tmp_path / "run1").write([failed])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.status == "launch_error"
    assert row.error == "boom"
    assert row.outputs == {}


def test_output_error_row_round_trips_with_empty_outputs(tmp_path):
    failed = _result("c0")
    failed.status = Status.OUTPUT_ERROR
    failed.outputs = {}
    ExecutionTable(tmp_path / "run1").write([failed])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.status is Status.OUTPUT_ERROR
    assert row.outputs == {}
    # A failed row still persists its inputs: that is what makes it replayable.
    _assert_identical(row.case.tensors["x"], np.full((2, 3), 0.5, dtype=np.float32))


def test_none_error_round_trips_as_empty_string(tmp_path):
    """`ExecutionResult.error` is declared `str = ""`; the table must not widen it."""
    result = _result("c0")
    result.error = None
    ExecutionTable(tmp_path / "run1").write([result])
    assert ExecutionTable(tmp_path / "run1").read()[0].error == ""


def test_unknown_status_names_the_run_directory(tmp_path):
    """A stale table must say *which* run is stale, not just that a value is bad."""
    run = tmp_path / "run1"
    result = _result("c0")
    result.status = "oom_error"  # a member some future phase adds
    ExecutionTable(run).write([result])
    with pytest.raises(ValueError, match="rows.parquet"):
        ExecutionTable(run).read()


def test_read_on_missing_run_returns_empty(tmp_path):
    assert ExecutionTable(tmp_path / "nope").read() == []


# --- Requirement A: status must come back as a Status member, not a bare str ---


def test_status_round_trips_as_enum_member(tmp_path):
    """`==` passes for a bare string because Status subclasses str; identity does not.

    Downstream code doing `row.status is Status.OK`, or matching on the enum,
    would silently never match if `read()` handed back a plain `str`.
    """
    ExecutionTable(tmp_path / "run1").write([_result("c0")])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.status is Status.OK
    assert isinstance(row.status, Status)


def test_status_written_as_bare_string_still_reads_as_enum(tmp_path):
    """A caller may build an ExecutionResult with the wire value directly."""
    result = _result("c0")
    result.status = "launch_error"
    ExecutionTable(tmp_path / "run1").write([result])
    assert ExecutionTable(tmp_path / "run1").read()[0].status is Status.LAUNCH_ERROR


# --- Requirement C: 0-d input tensors ---


def test_zero_dim_input_tensor_keeps_its_shape(tmp_path):
    """`np.ascontiguousarray` is documented `ndmin=1` and promotes 0-d to (1,).

    Outputs are normalized by `single_output`'s `np.atleast_1d`, but inputs never
    pass through it, and `InputDomain` accepts `shapes=((),)` — so a 0-d input is
    reachable. Persistence must not be the thing that changes a shape.
    """
    case = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="reduce",
        dtype="float32",
        shape=(),
        tensors={"x": np.array(0.5, dtype=np.float32)},
    )
    ExecutionTable(tmp_path / "run1").write([ExecutionResult(case=case)])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.case.shape == ()
    assert row.case.tensors["x"].shape == ()
    _assert_identical(row.case.tensors["x"], np.array(0.5, dtype=np.float32))


def test_empty_shaped_tensor_round_trips(tmp_path):
    """A `(0,)` tensor has a shape but no bytes; both must survive."""
    case = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="t",
        dtype="float32",
        shape=(0,),
        tensors={"x": np.zeros((0,), dtype=np.float32)},
    )
    outputs = {"y": np.zeros((0, 3), dtype=np.float32)}
    ExecutionTable(tmp_path / "run1").write([ExecutionResult(case=case, outputs=outputs)])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.case.shape == (0,)
    _assert_identical(row.case.tensors["x"], np.zeros((0,), dtype=np.float32))
    _assert_identical(row.outputs["y"], np.zeros((0, 3), dtype=np.float32))


def test_case_with_no_tensors_round_trips(tmp_path):
    """An empty safetensors payload keeps `read()` free of a per-row existence check."""
    case = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="t",
        dtype="float32",
        shape=(2, 3),
        tensors={},
    )
    ExecutionTable(tmp_path / "run1").write([ExecutionResult(case=case)])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.case.tensors == {}
    assert row.outputs == {}


def test_non_contiguous_input_tensor_round_trips(tmp_path):
    """safetensors cannot write a transpose view at all, so the writer must copy."""
    view = np.arange(6, dtype=np.float32).reshape(2, 3).T
    assert not view.flags.c_contiguous
    case = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="t",
        dtype="float32",
        shape=(3, 2),
        tensors={"x": view},
    )
    ExecutionTable(tmp_path / "run1").write([ExecutionResult(case=case, outputs={"y": view})])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    _assert_identical(row.case.tensors["x"], np.ascontiguousarray(view))
    _assert_identical(row.outputs["y"], np.ascontiguousarray(view))


# --- Requirement D: helper tensors and prefix collisions ---


def test_helper_tensor_keeps_its_own_dtype_and_shape(tmp_path):
    """`PermuteLastAxis` stores an int64 `__perm__` beside a float32 `x`."""
    perm = np.array([2, 0, 1], dtype=np.int64)
    x = np.full((2, 3), 0.5, dtype=np.float32)
    case = Case(
        case_id="c0",
        group_id="g0",
        relation="permute_last_axis",
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": x, "__perm__": perm},
    )
    ExecutionTable(tmp_path / "run1").write([ExecutionResult(case=case)])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert set(row.case.tensors) == {"x", "__perm__"}
    _assert_identical(row.case.tensors["__perm__"], perm)
    _assert_identical(row.case.tensors["x"], x)
    # The case-level dtype describes `x` only; the helper carries its own.
    assert row.case.dtype == "float32"


def test_tensor_names_do_not_collide_with_the_key_prefixes(tmp_path):
    """`in_bias`, and the pathological `in.y` / `out.x`, must stay distinct."""
    tensors = {
        "in_bias": np.array([1.0], dtype=np.float32),
        "in.y": np.array([2.0], dtype=np.float32),
        "out.x": np.array([3.0], dtype=np.float32),
        "x": np.array([4.0], dtype=np.float32),
    }
    case = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="t",
        dtype="float32",
        shape=(1,),
        tensors=tensors,
    )
    outputs = {"y": np.array([5.0], dtype=np.float32), "in.y": np.array([6.0], dtype=np.float32)}
    ExecutionTable(tmp_path / "run1").write([ExecutionResult(case=case, outputs=outputs)])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert set(row.case.tensors) == set(tensors)
    assert set(row.outputs) == set(outputs)
    for name, array in tensors.items():
        _assert_identical(row.case.tensors[name], array)
    for name, array in outputs.items():
        _assert_identical(row.outputs[name], array)


# --- Requirement E: read-only arrays ---


def test_readonly_arrays_persist_and_load_writeable(tmp_path):
    """An identity-like kernel returns an output that inherited the read-only flag.

    safetensors writes it fine. What matters downstream is that the *loaded*
    array is writeable, so an oracle doing in-place work on a replayed row is not
    blocked by a flag that was an artifact of the execution boundary.
    """
    x = np.full((2, 3), 0.5, dtype=np.float32)
    x.flags.writeable = False
    case = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="t",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": x},
    )
    ExecutionTable(tmp_path / "run1").write([ExecutionResult(case=case, outputs={"y": x})])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.case.tensors["x"].flags.writeable
    assert row.outputs["y"].flags.writeable
    _assert_identical(row.outputs["y"], np.full((2, 3), 0.5, dtype=np.float32))
    row.outputs["y"][0, 0] = 9.0  # must not raise


# --- Requirement F: empty write ---


def test_empty_write_produces_a_readable_empty_table(tmp_path):
    ExecutionTable(tmp_path / "run1").write([])
    assert ExecutionTable(tmp_path / "run1").read() == []
    assert ExecutionTable(tmp_path / "run1").read_groups() == {}


def test_empty_write_then_rewrite_replaces_the_table(tmp_path):
    table = ExecutionTable(tmp_path / "run1")
    table.write([])
    table.write([_result("c0")])
    assert [r.case.case_id for r in ExecutionTable(tmp_path / "run1").read()] == ["c0"]


def test_rewrite_does_not_resurrect_rows_from_the_previous_write(tmp_path):
    """Parquet is the index: an orphaned payload file must not become a row."""
    table = ExecutionTable(tmp_path / "run1")
    table.write([_result("c0"), _result("c1")])
    table.write([_result("c1")])
    assert [r.case.case_id for r in ExecutionTable(tmp_path / "run1").read()] == ["c1"]


# --- Atomicity: the index and the payload set are never observed out of step ---


def test_torn_rewrite_never_pairs_old_metadata_with_new_tensors(tmp_path):
    """A rewrite that fails midway must not leave a readable mixture.

    Reusing case_ids across writes of the same run is the normal case — a
    re-run, a resume — so a payload overwritten before the index is republished
    would pair v2 tensor bytes with v1 telemetry and `read()` would report it
    without raising. That is worse than a loud failure: every oracle scored
    against such a table is scored against an execution that never happened.
    """
    run = tmp_path / "run1"
    ExecutionTable(run).write([_tagged("c0", 1.0), _tagged("c1", 1.0)])
    assert _observed(run) == [("c0", 1.0, 1.0), ("c1", 1.0, 1.0)]

    doomed = _tagged("c1", 2.0)
    # An object-dtype array is genuinely unpersistable: safetensors rejects it,
    # so the second iteration of the write loop raises after the first has
    # already written its payload under a reused case_id.
    doomed.outputs = {"y": np.array([object()], dtype=object)}
    with pytest.raises(Exception):  # noqa: B017 - SafetensorError is not public
        ExecutionTable(run).write([_tagged("c0", 2.0), doomed])

    assert _observed(run) in ([], [("c0", 1.0, 1.0), ("c1", 1.0, 1.0)])


def test_failed_write_leaves_no_orphan_payloads_behind(tmp_path):
    run = tmp_path / "run1"
    ExecutionTable(run).write([_result("c0"), _result("c1")])
    ExecutionTable(run).write([_result("c0")])
    payloads = {p.name for p in (run / "tensors").iterdir()}
    assert payloads == {"c0.safetensors"}


def test_duplicate_case_ids_in_one_write_are_rejected(tmp_path):
    """Two rows sharing a payload file is the torn table again, with no crash."""
    run = tmp_path / "run1"
    ExecutionTable(run).write([_tagged("c0", 1.0)])
    with pytest.raises(ValueError, match="duplicate case_ids"):
        ExecutionTable(run).write([_tagged("c1", 2.0), _tagged("c1", 3.0)])
    # Rejected before any payload was touched, so the old table is intact.
    assert _observed(run) == [("c0", 1.0, 1.0)]


def test_write_leaves_no_temporary_files_in_the_run_dir(tmp_path):
    run = tmp_path / "run1"
    ExecutionTable(run).write([_result("c0")])
    assert {p.name for p in run.iterdir()} == {"rows.parquet", "tensors"}


# --- Telemetry serialization gate ---


def test_numpy_scalar_telemetry_round_trips(tmp_path):
    """A device backend reporting a counter as a numpy scalar must not abort the run.

    `np.float64` subclasses `float` and would have slipped through, but
    `np.float32`, `np.int64` and `np.bool_` do not — and the failure would fire
    after the tensor loop had already overwritten payloads.
    """
    result = _result("c0")
    result.telemetry = {
        "backend": "cuda",
        "wall_ms": np.float32(1.5),
        "sm_occupancy": np.int64(7),
        "spilled": np.bool_(True),
        "per_warp": np.arange(3, dtype=np.int64),
    }
    ExecutionTable(tmp_path / "run1").write([result])
    telemetry = ExecutionTable(tmp_path / "run1").read()[0].telemetry
    assert telemetry == {
        "backend": "cuda",
        "wall_ms": 1.5,
        "sm_occupancy": 7,
        "spilled": True,
        "per_warp": [0, 1, 2],
    }


def test_unserializable_telemetry_fails_loudly_and_preserves_the_old_table(tmp_path):
    """The gate runs before any payload is touched, so the old table survives."""
    run = tmp_path / "run1"
    ExecutionTable(run).write([_tagged("c0", 1.0)])

    doomed = _tagged("c0", 2.0)
    doomed.telemetry = {"backend": "numpy", "wall_ms": {1, 2}}
    with pytest.raises(TypeError, match="not JSON-serializable"):
        ExecutionTable(run).write([doomed])

    assert _observed(run) == [("c0", 1.0, 1.0)]


# --- Requirement G: case_id becomes a filename ---


def test_case_id_with_relation_separator_round_trips(tmp_path):
    """Relation-derived ids contain `::` (see `relations._derived`)."""
    result = _result("softmax-g00000-base::shift_rows", relation="shift_rows")
    ExecutionTable(tmp_path / "run1").write([result])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.case.case_id == "softmax-g00000-base::shift_rows"
    _assert_identical(row.outputs["y"], np.full((2, 3), 0.25, dtype=np.float32))


# --- Criterion KERNEL_IDENTITY: ground truth on the row ---


def test_kernel_identity_round_trips(tmp_path):
    """A row must say which kernel produced it and whether that kernel was broken.

    Without both, a detection rate cannot be computed from the table: nothing joins
    a verdict to the ground truth about the kernel under test.
    """
    result = _result("c0")
    result.kernel_id = "softmax_missing_max_subtraction"
    result.kernel_is_broken = True
    ExecutionTable(tmp_path / "run").write([result])

    row = ExecutionTable(tmp_path / "run").read()[0]
    assert row.kernel_id == "softmax_missing_max_subtraction"
    assert row.kernel_is_broken is True


def test_kernel_identity_defaults_are_recorded_not_guessed(tmp_path):
    """An unlabelled kernel round-trips as unlabelled, never as a silent False.

    `kernel_is_broken=None` means "ground truth not stated"; False means "stated
    correct". Collapsing the two would silently enlarge the correct-kernel
    denominator of the false-positive rate.
    """
    ExecutionTable(tmp_path / "run").write([_result("c0")])
    row = ExecutionTable(tmp_path / "run").read()[0]
    assert row.kernel_id == ""
    assert row.kernel_is_broken is None


def test_kernel_is_broken_keeps_all_three_states_in_one_batch(tmp_path):
    """True, False and None must stay distinct through Parquet, in one column.

    `pa.bool_()` is nullable, but a mixed batch is where a null would most
    plausibly collapse into False, or a False into a null. Either direction moves
    a row between the numerator and the denominator of a rate with no error, so
    all three states are asserted by identity rather than truthiness.
    """
    broken = _result("c0")
    broken.kernel_id = "softmax_missing_max_subtraction"
    broken.kernel_is_broken = True
    correct = _result("c1")
    correct.kernel_id = "softmax_reference"
    correct.kernel_is_broken = False
    unlabelled = _result("c2")
    unlabelled.kernel_id = "softmax_unknown"

    ExecutionTable(tmp_path / "run").write([broken, correct, unlabelled])

    rows = ExecutionTable(tmp_path / "run").read()
    assert [row.kernel_id for row in rows] == [
        "softmax_missing_max_subtraction",
        "softmax_reference",
        "softmax_unknown",
    ]
    assert rows[0].kernel_is_broken is True
    assert rows[1].kernel_is_broken is False
    assert rows[2].kernel_is_broken is None


def test_read_rejects_a_table_written_before_the_kernel_columns_existed(tmp_path):
    """A narrower table must be refused by name, not backfilled and not a KeyError.

    Backfilling `kernel_is_broken` would forge ground truth: the default is
    indistinguishable from a recorded value. A bare `KeyError: 'kernel_id'` from
    inside the row loop names neither the run directory nor the cause.
    """
    run = tmp_path / "run"
    ExecutionTable(run).write([_result("c0")])
    narrowed = pq.read_table(run / "rows.parquet").drop_columns(
        ["kernel_id", "kernel_is_broken"]
    )
    pq.write_table(narrowed, run / "rows.parquet")

    with pytest.raises(ValueError, match=r"different schema.*kernel_id.*kernel_is_broken"):
        ExecutionTable(run).read()


def test_a_record_missing_a_schema_key_is_rejected_before_anything_is_written():
    """A builder that drops a key must fail, not write a silently-null column.

    `pa.Table.from_pylist` presence-checks nothing, so the omission would produce
    a *complete* file whose column reads back all-None — indistinguishable from an
    honest unlabelled run, and invisible to the read-side column check. This is
    the only place that failure is catchable.
    """
    with pytest.raises(ValueError, match=r"missing: \['kernel_is_broken'\]"):
        _conform({name: "" for name in SCHEMA.names if name != "kernel_is_broken"}, SCHEMA)


def test_a_record_with_a_mistyped_key_names_both_halves():
    """A typo is one missing key and one unexpected key; naming only the missing
    half sends the reader looking for a deletion that never happened."""
    record = {name: "" for name in SCHEMA.names}
    record["kernel_is_borken"] = record.pop("kernel_is_broken")

    with pytest.raises(
        ValueError,
        match=r"missing: \['kernel_is_broken'\].*unexpected: \['kernel_is_borken'\]",
    ):
        _conform(record, SCHEMA)


def test_the_builder_actually_applies_the_schema_check(tmp_path, monkeypatch):
    """`_record` must route through `_conform`, not merely coexist with it.

    Testing `_conform` directly leaves the wiring uncovered: deleting the call
    from `_record` keeps every other test green, because every record the current
    builder produces is well-formed. Widening `SCHEMA` without teaching `_record`
    the new column is exactly the Task 2 mistake this guard exists to catch, so
    that is the mutation applied here.
    """
    widened = pa.schema([*SCHEMA, pa.field("arm", pa.string())])
    monkeypatch.setattr("autokernel_pbt.props.table.SCHEMA", widened)

    with pytest.raises(ValueError, match=r"missing: \['arm'\]"):
        ExecutionTable(tmp_path / "run").write([_result("c0")])


def test_a_none_kernel_id_does_not_widen_the_string_column(tmp_path):
    """`kernel_id` is declared `str = ""`; a caller that skipped the default must
    not put a None into a str-typed column, where a downstream `.startswith(...)`
    becomes an AttributeError and None groups apart from "".
    """
    result = _result("c0")
    result.kernel_id = None
    ExecutionTable(tmp_path / "run").write([result])
    assert ExecutionTable(tmp_path / "run").read()[0].kernel_id == ""


# --- Corpus identity ----------------------------------------------------------


def test_the_corpus_fingerprint_is_stamped_on_every_row(tmp_path):
    """One identity per write, broadcast to the whole table.

    Per-row rather than in a side table because Parquet dictionary-encodes a
    single-valued string column to almost nothing, and a side table would cost a
    join to save that nothing.
    """
    run = tmp_path / "run"
    ExecutionTable(run).write([_result("c0"), _result("c1")])

    stamped = pq.read_table(run / "rows.parquet").column("corpus_fingerprint").to_pylist()
    assert len(set(stamped)) == 1
    assert stamped[0] == ExecutionTable(run).corpus_fingerprint()
    assert stamped[0], "an empty fingerprint would pair with anything"


def test_re_recording_the_same_cases_mints_a_new_corpus_identity(tmp_path):
    """The defence against the worst attack, and why this is not a pure hash.

    Case ids are a pure function of `(seed, index)`, so a re-record — the normal
    workflow after fixing a kernel — produces byte-for-byte the same id set while
    recording entirely different executions. A content-derived fingerprint would be
    identical across the two, and the *previous* run's scores would still appear to
    belong to the new table. The per-write uuid is what separates them.
    """
    run = tmp_path / "run"
    ExecutionTable(run).write([_result("c0")])
    first = ExecutionTable(run).corpus_fingerprint()
    ExecutionTable(run).write([_result("c0")])
    second = ExecutionTable(run).corpus_fingerprint()

    assert first and second
    assert first != second, "a re-record kept the old corpus identity"


def test_a_different_case_set_gets_a_different_fingerprint(tmp_path):
    """The other half: the identity is also a statement about what is in the table."""
    one = tmp_path / "one"
    two = tmp_path / "two"
    ExecutionTable(one).write([_result("c0")])
    ExecutionTable(two).write([_result("c0"), _result("c1")])
    assert ExecutionTable(one).corpus_fingerprint() != ExecutionTable(two).corpus_fingerprint()


def test_a_table_carrying_two_corpus_fingerprints_is_refused(tmp_path):
    """`write` broadcasts one value, so two means a file assembled from two runs.

    Picking either would certify half a table as the whole of it.
    """
    run = tmp_path / "run"
    ExecutionTable(run).write([_result("c0"), _result("c1")])
    table = pq.read_table(run / "rows.parquet")
    patched = table.set_column(
        table.schema.get_field_index("corpus_fingerprint"),
        "corpus_fingerprint",
        pa.array(["aaa", "bbb"], type=pa.string()),
    )
    pq.write_table(patched, run / "rows.parquet")

    with pytest.raises(ValueError, match="different corpus"):
        ExecutionTable(run).corpus_fingerprint()


def test_a_run_with_no_rows_has_no_corpus_fingerprint(tmp_path):
    """`""` means "unstated"; every pairing check must refuse it rather than match it."""
    assert ExecutionTable(tmp_path / "nope").corpus_fingerprint() == ""
    empty = tmp_path / "empty"
    ExecutionTable(empty).write([])
    assert ExecutionTable(empty).corpus_fingerprint() == ""


def test_case_spec_round_trips(tmp_path):
    """The persistence half of CaseSpec's reason for existing.

    In-memory regeneration is covered by test_spec.py. This is the other half: a run
    that has been written and re-read must still carry the recipe, or an offline
    shrinker reading it months later has nothing to reduce and must guess
    (seed, task_id, group_index, shape, transforms) back out of the domain -- exactly
    the retrofit spec.py's docstring argues must not be deferred.
    """
    from autokernel_pbt.props.spec import CaseSpec

    spec = CaseSpec(
        seed=11, task_id="softmax", group_index=3, shape=(2, 3), transforms=("shift_rows",)
    )
    result = _result("c0")
    result.case_spec = spec
    ExecutionTable(tmp_path / "run1").write([result])
    assert ExecutionTable(tmp_path / "run1").read()[0].case_spec == spec


def test_a_row_with_no_spec_round_trips_as_none(tmp_path):
    # A hand-built group has no recipe, and inventing one would regenerate something
    # else entirely. "" on disk must come back as None, not as an empty CaseSpec.
    ExecutionTable(tmp_path / "run1").write([_result("c0")])
    assert ExecutionTable(tmp_path / "run1").read()[0].case_spec is None


def test_one_read_returns_the_rows_and_the_fingerprint_they_carry(tmp_path):
    """The pair a consumer stamps or compares must come from the read it scored.

    A fingerprint fetched by a *second* read describes whatever is on disk by then,
    which is the corpus a concurrent re-record left -- not the one in hand.
    """
    run = tmp_path / "run"
    ExecutionTable(run).write([_result("c0"), _result("c1", relation="shift_rows")])

    rows, fingerprint = ExecutionTable(run).read_with_fingerprint()
    groups, grouped_fingerprint = ExecutionTable(run).read_groups_with_fingerprint()

    assert [r.case.case_id for r in rows] == ["c0", "c1"]
    assert {gid: [r.case.case_id for r in rs] for gid, rs in groups.items()} == {"g0": ["c0", "c1"]}
    assert fingerprint == grouped_fingerprint == ExecutionTable(run).corpus_fingerprint()
    assert ExecutionTable(tmp_path / "nope").read_with_fingerprint() == ([], "")


def test_the_paired_read_opens_the_index_once(tmp_path, monkeypatch):
    """"From one read" is the whole contract, so it is asserted directly.

    A version that returned `self.read()` beside `self.corpus_fingerprint()` would have
    the right signature and the original defect: two reads of `rows.parquet`, between
    which a re-record lands unseen.
    """
    run = tmp_path / "run"
    ExecutionTable(run).write([_result("c0")])
    original = pq.read_table
    opened: list[str] = []

    def read_table(source, *args, **kwargs):
        opened.append(str(source))
        return original(source, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", read_table)
    ExecutionTable(run).read_with_fingerprint()
    assert opened == [str(run / "rows.parquet")]


# --- Payload identity: a payload names the corpus it belongs to -----------------

#: What the build before payload stamping wrote: a bare sha256 hexdigest over a
#: uuid that was never stored, so its case-id half cannot be recomputed by anyone.
LEGACY_FINGERPRINT = "ab" * 32


def _payload_metadata(path) -> dict | None:
    with safe_open(str(path), framework="np") as handle:
        return handle.metadata()


def _patch_rows(run, transform) -> None:
    table = pq.read_table(run / "rows.parquet")
    pq.write_table(transform(table), run / "rows.parquet")


def _with_fingerprint(value: str):
    def transform(table: pa.Table) -> pa.Table:
        return table.set_column(
            table.schema.get_field_index("corpus_fingerprint"),
            "corpus_fingerprint",
            pa.array([value] * table.num_rows, type=pa.string()),
        )

    return transform


def _as_legacy(run) -> None:
    """Rewrite a freshly written run the way the pre-stamping build laid it out.

    Verbatim case ids as filenames, no safetensors metadata, and a bare 64-hex
    fingerprint. The five recorded GPU runs have exactly this shape, and they are
    the corpus the project cannot re-record, so the reader must keep accepting it.
    """
    _patch_rows(run, _with_fingerprint(LEGACY_FINGERPRINT))
    for case_id in pq.read_table(run / "rows.parquet").column("case_id").to_pylist():
        encoded = run / "tensors" / payload_filename(case_id)
        tensors = load_file(str(encoded))
        encoded.unlink()
        save_file(tensors, str(run / "tensors" / f"{case_id}.safetensors"))


def _strip_payload_metadata(run) -> None:
    for path in (run / "tensors").iterdir():
        save_file(load_file(str(path)), str(path))


def test_every_payload_is_stamped_with_its_corpus_fingerprint(tmp_path):
    run = tmp_path / "run"
    ExecutionTable(run).write([_result("c0"), _result("c1")])
    fingerprint = ExecutionTable(run).corpus_fingerprint()

    stamps = [_payload_metadata(path) for path in sorted((run / "tensors").iterdir())]
    assert stamps == [{"corpus_fingerprint": fingerprint}] * 2


def test_payloads_from_another_write_are_refused(tmp_path):
    """A v1 index beside v2 payloads -- a partial scp of a run dir, or a write landing
    between the index read and the payload reads -- read clean before payloads were
    stamped: v1 telemetry paired with v2 tensor bytes, and nothing said so.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    ExecutionTable(a).write([_tagged("c0", 1.0), _tagged("c1", 1.0)])
    ExecutionTable(b).write([_tagged("c0", 2.0), _tagged("c1", 2.0)])
    indexed, foreign = ExecutionTable(a).corpus_fingerprint(), ExecutionTable(b).corpus_fingerprint()
    shutil.rmtree(a / "tensors")
    shutil.copytree(b / "tensors", a / "tensors")

    with pytest.raises(
        ValueError,
        match=rf"^payload \S+/c0\.safetensors belongs to corpus {foreign}, but "
        rf"\S+/rows\.parquet indexes corpus {indexed}",
    ):
        ExecutionTable(a).read()


def test_a_legacy_index_beside_stamped_payloads_is_refused(tmp_path):
    """The scp case the stamp exists for: an old run's `rows.parquet` left beside a
    newer recording's `tensors/`. The index cannot vouch for itself, but every
    stamped payload can still say it belongs to a different corpus."""
    a, b = tmp_path / "a", tmp_path / "b"
    ExecutionTable(a).write([_tagged("c0", 1.0)])
    _as_legacy(a)
    ExecutionTable(b).write([_tagged("c0", 2.0)])
    foreign = ExecutionTable(b).corpus_fingerprint()
    shutil.rmtree(a / "tensors")
    shutil.copytree(b / "tensors", a / "tensors")

    with pytest.raises(
        ValueError, match=rf"belongs to corpus {foreign}, but \S+ indexes corpus {LEGACY_FINGERPRINT}"
    ):
        ExecutionTable(a).read()


def test_a_stamped_index_beside_unstamped_payloads_is_refused(tmp_path):
    """Only a legacy index excuses a payload with no stamp.

    Every payload a stamping build writes carries one, so an unstamped payload under a
    stamped index is not legacy data -- it is a payload from some other write.
    """
    run = tmp_path / "run"
    ExecutionTable(run).write([_tagged("c0", 1.0)])
    _strip_payload_metadata(run)

    with pytest.raises(
        ValueError, match=r"^payload \S+/c0\.safetensors carries no corpus fingerprint, but"
    ):
        ExecutionTable(run).read()


def test_a_legacy_run_still_reads(tmp_path):
    """Unstamped payloads under verbatim names, beneath a legacy index, read as before.

    `::` ids are the case that needs the verbatim-name fallback: their encoded name
    differs from the name the old build wrote. Base ids encode to themselves.
    """
    run = tmp_path / "run"
    ids = ["softmax-g00000-base", "softmax-g00000-base::shift_rows"]
    ExecutionTable(run).write([_tagged(ids[0], 1.0), _tagged(ids[1], 2.0)])
    _as_legacy(run)
    assert sorted(p.name for p in (run / "tensors").iterdir()) == [f"{i}.safetensors" for i in ids]

    assert _observed(run) == [(ids[0], 1.0, 1.0), (ids[1], 2.0, 2.0)]
    assert ExecutionTable(run).corpus_fingerprint() == LEGACY_FINGERPRINT


def test_the_verbatim_name_fallback_is_for_legacy_tables_only(tmp_path):
    """A stamping build never writes a verbatim name, so under its index one is a stray.

    And a stray can be worse than foreign: on a case-insensitive filesystem the
    verbatim `Case-0.safetensors` *is* `case-0.safetensors`, another row of the same
    corpus, whose stamp matches. Copying the encoded file to the verbatim name is the
    filesystem-independent form of that: a payload the stamp cannot tell apart.
    """
    run = tmp_path / "run"
    ExecutionTable(run).write([_tagged("a::b", 1.0)])
    encoded = run / "tensors" / payload_filename("a::b")
    shutil.copy(encoded, run / "tensors" / "a::b.safetensors")
    encoded.unlink()

    with pytest.raises(FileNotFoundError):
        ExecutionTable(run).read()


def test_the_legacy_fallback_never_leaves_the_tensor_dir(tmp_path):
    """The pre-encoding build wrote `../outside` *outside* `tensors/`; the reader must
    not follow an index there, or a crafted `rows.parquet` names any file on disk."""
    run = tmp_path / "run"
    ExecutionTable(run).write([_tagged("../outside", 1.0)])
    _as_legacy(run)
    assert (run / "outside.safetensors").exists()

    with pytest.raises(FileNotFoundError, match=re.escape(payload_filename("../outside"))):
        ExecutionTable(run).read()


# --- Payload filenames: injective, portable, case-fold-safe ---------------------

#: Characters at least one target filesystem refuses, or that sync clients rewrite.
#: OneDrive renamed every `...::shift_rows.safetensors` to `..._shift_rows`, making
#: those runs unreadable.
_UNPORTABLE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))}


def test_case_ids_differing_only_in_case_get_distinct_payloads(tmp_path):
    """macOS's default filesystem is case-insensitive: `Case-0` and `case-0` passed the
    duplicate guard as strings and then shared one file, both rows reading the
    survivor's bytes under their own metadata."""
    run = tmp_path / "run"
    ExecutionTable(run).write([_tagged("Case-0", 1.0), _tagged("case-0", 2.0)])

    names = [p.name for p in (run / "tensors").iterdir()]
    assert len({name.casefold() for name in names}) == 2, names
    assert _observed(run) == [("Case-0", 1.0, 1.0), ("case-0", 2.0, 2.0)]


def test_case_ids_with_path_separators_stay_inside_the_tensor_dir(tmp_path):
    run = tmp_path / "run"
    ids = ["../escaped", "a/b", "a\\b", "/abs"]
    ExecutionTable(run).write([_tagged(case_id, float(i)) for i, case_id in enumerate(ids)])

    assert {p.name for p in run.iterdir()} == {"rows.parquet", "tensors"}
    assert all(p.is_file() for p in (run / "tensors").iterdir())
    assert _observed(run) == [(case_id, float(i), float(i)) for i, case_id in enumerate(ids)]


def test_payload_names_are_portable(tmp_path):
    """No character a target filesystem refuses, no hidden or dot-only name, and no
    Windows device name -- `con.safetensors` is unopenable there, extension or not."""
    run = tmp_path / "run"
    ids = ["softmax-g00000-base::shift_rows", "con", "AUX.x", ".", "..", ".hidden", 'q?"<>|*', "b\\c"]
    ExecutionTable(run).write([_tagged(case_id, float(i)) for i, case_id in enumerate(ids)])

    for path in (run / "tensors").iterdir():
        assert not _UNPORTABLE.search(path.name), path.name
        assert not path.name.startswith("."), path.name
        assert path.name.split(".")[0].casefold() not in _WINDOWS_RESERVED, path.name
    assert _observed(run) == [(case_id, float(i), float(i)) for i, case_id in enumerate(ids)]


def test_payload_filename_is_injective_and_case_fold_safe():
    """Exhaustive over every id of length <= 3 drawn from the characters that break a
    naive escape: the escape character itself and a hex digit (so `%3a` can meet the
    escape of `:`), case pairs, dots, separators, `:`, and a non-ASCII letter with an
    uppercase twin. Two ids sharing a name -- exactly, or only under case folding --
    would share a payload file.

    Which characters a name may *contain* is the two tests above; this one is only
    about distinctness, so that an unescaped `%` or a non-ASCII pass-through dies here
    and nowhere else.
    """
    alphabet = "aA3.%:/_-0 \u00e9\u00c9"
    ids = ["".join(chars) for n in range(4) for chars in itertools.product(alphabet, repeat=n)]
    names = [payload_filename(case_id) for case_id in ids]

    assert len(set(names)) == len(ids)
    assert len({name.casefold() for name in names}) == len(ids)


def test_the_duplicate_guard_checks_the_filename_not_the_string(tmp_path, monkeypatch):
    """The key that must be unique is the payload *file*, so that is what is checked.

    With today's injective encoding the two coincide, which is exactly why this needs
    its own test: a guard on the string passes every other test here, and would pass
    `Case-0`/`case-0` straight into one file again the day the encoding stops being
    injective -- as the verbatim filename already was not, on a case-insensitive disk.
    """
    monkeypatch.setattr(
        "autokernel_pbt.props.table.payload_filename", lambda case_id: f"{case_id.lower()}.st"
    )
    run = tmp_path / "run"
    with pytest.raises(
        ValueError,
        match=r"^duplicate case_ids in one write, which would share a payload: "
        r"\['Case-0', 'case-0'\]$",
    ):
        ExecutionTable(run).write([_tagged("Case-0", 1.0), _tagged("case-0", 2.0)])
    assert not run.exists(), "rejected only after touching the filesystem"


def test_base_case_ids_keep_their_legacy_payload_names():
    # The recorded runs' base payloads are found under the encoded name directly; only
    # relation-derived ids need the verbatim fallback.
    assert payload_filename("softmax-g00000-base") == "softmax-g00000-base.safetensors"


# --- Corpus identity is verifiable on read --------------------------------------


def _drop_row(run) -> None:
    _patch_rows(run, lambda table: table.filter(pc.not_equal(table["case_id"], "c1")))


def _relabel_row(run) -> None:
    """Same row count, a different id -- with a payload under the new name, so the
    only thing left to notice is the fingerprint."""
    def relabel(table: pa.Table) -> pa.Table:
        ids = ["c9" if cid == "c1" else cid for cid in table.column("case_id").to_pylist()]
        return table.set_column(table.schema.get_field_index("case_id"), "case_id", pa.array(ids))

    _patch_rows(run, relabel)
    shutil.copy(run / "tensors" / "c1.safetensors", run / "tensors" / "c9.safetensors")


def _add_row(run) -> None:
    def add(table: pa.Table) -> pa.Table:
        extra = table.slice(0, 1)
        extra = extra.set_column(extra.schema.get_field_index("case_id"), "case_id", pa.array(["c9"]))
        return pa.concat_tables([table, extra])

    _patch_rows(run, add)


@pytest.mark.parametrize("edit", [_drop_row, _relabel_row, _add_row], ids=lambda f: f.__name__[1:])
def test_rows_edited_under_an_intact_fingerprint_are_refused(tmp_path, edit):
    """The case-id half of the fingerprint, checked rather than merely hashed.

    The uuid half was mixed into one digest and never stored, so no reader could
    recompute it: a table that lost a group kept its stamp and paired with the scores
    that judged the group, every rate computed over a corpus nobody scored.
    """
    run = tmp_path / "run"
    ExecutionTable(run).write([_result("c0"), _result("c1"), _result("c2")])
    stamped = ExecutionTable(run).corpus_fingerprint()
    edit(run)

    expected = (
        rf"^\S+/rows\.parquet carries corpus fingerprint {stamped}, whose case-id half "
        rf"does not match the \d+ case id\(s\) the table holds"
    )
    with pytest.raises(ValueError, match=expected):
        ExecutionTable(run).read()
    with pytest.raises(ValueError, match=expected):
        ExecutionTable(run).corpus_fingerprint()
