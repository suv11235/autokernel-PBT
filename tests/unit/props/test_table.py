"""ExecutionTable round-trip tests."""

import numpy as np

from autokernel_pbt.props.backends.base import ExecutionResult, Status
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.table import ExecutionTable


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
        status="ok",
    )


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


# --- Requirement G: case_id becomes a filename ---


def test_case_id_with_relation_separator_round_trips(tmp_path):
    """Relation-derived ids contain `::` (see `relations._derived`)."""
    result = _result("softmax-g00000-base::shift_rows", relation="shift_rows")
    ExecutionTable(tmp_path / "run1").write([result])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.case.case_id == "softmax-g00000-base::shift_rows"
    _assert_identical(row.outputs["y"], np.full((2, 3), 0.25, dtype=np.float32))
