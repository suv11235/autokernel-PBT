"""NumPy backend tests."""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import (
    Status,
    kernel_inputs,
    readonly_inputs,
    single_output,
)
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.case import Case


def _case() -> Case:
    return Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="relu",
        dtype="float32",
        shape=(4,),
        tensors={"x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)},
    )


def _case_2d() -> Case:
    rng = np.random.default_rng(0)
    return Case(
        case_id="c1",
        group_id="g1",
        relation="base",
        task_id="softmax",
        dtype="float32",
        shape=(3, 4),
        tensors={"x": rng.normal(size=(3, 4)).astype(np.float32)},
    )


def test_successful_run_reports_ok():
    result = NumpyBackend().run(lambda x: np.maximum(x, 0.0), _case())
    assert result.status == Status.OK
    assert np.array_equal(result.outputs["y"], np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32))


def test_exception_is_captured_not_raised():
    def boom(x):
        raise RuntimeError("kernel exploded")

    result = NumpyBackend().run(boom, _case())
    assert result.status == Status.LAUNCH_ERROR
    assert "kernel exploded" in result.error


def test_telemetry_records_backend_name():
    result = NumpyBackend().run(lambda x: x, _case())
    assert result.telemetry["backend"] == "numpy"


def test_telemetry_records_wall_time():
    result = NumpyBackend().run(lambda x: x, _case())
    assert result.telemetry["wall_ms"] >= 0.0


def test_perm_helper_tensor_is_not_passed_to_kernel():
    """Requirement A: assert on the kwargs the kernel actually saw.

    Asserting only ``status == "ok"`` would still fail if the filter were
    removed (Python raises TypeError for an unexpected kwarg), but it would
    fail for the wrong reason and would stop failing the moment a kernel
    accepted ``**kwargs``. Record the kwargs and assert on them directly.
    """
    seen: dict[str, np.ndarray] = {}

    def recording_kernel(**kwargs):
        seen.update(kwargs)
        return kwargs["x"]

    case = _case()
    case.tensors["__perm__"] = np.array([0, 1, 2, 3], dtype=np.int64)
    result = NumpyBackend().run(recording_kernel, case)

    assert result.status == Status.OK
    assert set(seen) == {"x"}
    assert "__perm__" not in seen


def test_kernel_inputs_filters_only_helper_prefixed_names():
    case = _case()
    case.tensors["__perm__"] = np.arange(4, dtype=np.int64)
    case.tensors["bias_"] = np.zeros(4, dtype=np.float32)
    assert set(kernel_inputs(case)) == {"x", "bias_"}


# --- Multi-output kernels are rejected, not silently mangled ---


def test_tuple_output_is_rejected_with_a_clear_error():
    def two_outputs(x):
        return x, x + 1.0

    result = NumpyBackend().run(two_outputs, _case())
    assert result.status == Status.OUTPUT_ERROR
    assert "single" in result.error
    assert result.outputs == {}


def test_dict_output_is_rejected_with_a_clear_error():
    result = NumpyBackend().run(lambda x: {"y": x}, _case())
    assert result.status == Status.OUTPUT_ERROR
    assert result.outputs == {}


def test_none_output_is_rejected():
    result = NumpyBackend().run(lambda x: None, _case())
    assert result.status == Status.OUTPUT_ERROR


def test_list_of_arrays_is_rejected_rather_than_stacked():
    """np.asarray would silently stack these into a (2, 4) array."""
    result = NumpyBackend().run(lambda x: [x, x], _case())
    assert result.status == Status.OUTPUT_ERROR


# --- Unpersistable dtypes are rejected here, not at the batch write ---
#
# Task 7 writes every result in one loop over safetensors.save_file. Any dtype
# outside bool/int/uint/float raises there and takes the whole run's
# persistence down with it, so one broken corpus kernel would abort a paid
# batch. These must come back as data.


@pytest.mark.parametrize(
    ("name", "kernel"),
    [
        ("str", lambda x: "error"),
        ("bytes", lambda x: b"err"),
        ("complex", lambda x: x.astype(np.complex128)),
        ("complex_scalar", lambda x: 1 + 2j),
        ("structured", lambda x: np.zeros(2, dtype=[("a", np.int16), ("b", np.int16)])),
        ("str_array", lambda x: np.array(["a", "b"])),
    ],
)
def test_unpersistable_output_dtype_is_rejected(name, kernel):
    result = NumpyBackend().run(kernel, _case())
    assert result.status == Status.OUTPUT_ERROR, name
    assert result.outputs == {}


@pytest.mark.parametrize(
    "kernel",
    [
        lambda x: x.astype(np.float64),
        lambda x: x.astype(np.float16),
        lambda x: x.astype(np.int32),
        lambda x: x.astype(np.uint8),
        lambda x: x > 0.0,
    ],
)
def test_persistable_output_dtypes_are_accepted(kernel):
    result = NumpyBackend().run(kernel, _case())
    assert result.status == Status.OK
    assert result.outputs["y"].dtype.kind in "biuf"


def test_scalar_output_is_normalized_to_one_element():
    """A 0-d reduction output is normalized to (1,) because persistence forces
    it: Task 7's writer calls `np.ascontiguousarray`, which is `ndmin=1`. Doing
    it up front keeps the in-memory row identical to the replayed one instead
    of letting the shape change under us.
    """
    result = NumpyBackend().run(lambda x: np.sum(x), _case())
    assert result.status == Status.OK
    assert result.outputs["y"].shape == (1,)
    assert result.outputs["y"][0] == pytest.approx(2.0)


def test_zero_dim_output_round_trips_at_a_stable_shape(tmp_path):
    """Replay fairness for reductions: (1,) at every step of the pipeline.

    The 0-d array enters `single_output`, is persisted the way Task 7 persists
    it, and is read back — the shape must never change along the way.
    """
    from safetensors.numpy import load_file, save_file

    raw = np.asarray(np.sum(np.arange(4, dtype=np.float32)))
    assert raw.shape == ()

    normalized = single_output(raw)
    assert normalized.shape == (1,)

    path = tmp_path / "t.safetensors"
    save_file({"y": np.ascontiguousarray(normalized)}, str(path))
    assert load_file(str(path))["y"].shape == (1,)

    # And the backend's own output agrees with that end-to-end shape.
    assert NumpyBackend().run(lambda x: np.sum(x), _case()).outputs["y"].shape == (1,)


def test_ascontiguousarray_not_safetensors_is_what_promotes_zero_dim(tmp_path):
    """Pins the actual cause, so a future reader does not blame safetensors.

    If numpy ever stops promoting 0-d in `ascontiguousarray`, the normalization
    above becomes unnecessary and this test says so.
    """
    from safetensors.numpy import load_file, save_file

    raw = np.asarray(np.float32(6.0))
    path = tmp_path / "t.safetensors"

    save_file({"y": raw}, str(path))
    assert load_file(str(path))["y"].shape == ()  # safetensors preserves 0-d

    save_file({"y": np.ascontiguousarray(raw)}, str(path))
    assert load_file(str(path))["y"].shape == (1,)  # ascontiguousarray promotes


# --- Inputs are aliased but read-only for the duration of the call ---


def test_kernel_inputs_aliases_case_tensors_rather_than_copying():
    """No defensive copy: copying every input would double peak memory on the
    hardware backends this boundary exists for."""
    case = _case()
    assert kernel_inputs(case)["x"] is case.tensors["x"]


def test_in_place_kernel_mutation_is_rejected_not_silently_applied():
    def in_place(x):
        x *= 0.0
        return x

    case = _case()
    before = case.tensors["x"].copy()
    result = NumpyBackend().run(in_place, case)

    assert result.status == Status.LAUNCH_ERROR
    assert "read-only" in result.error
    assert np.array_equal(case.tensors["x"], before)


def test_wellbehaved_kernel_does_not_mutate_case_inputs():
    case = _case()
    before = case.tensors["x"].copy()
    NumpyBackend().run(lambda x: np.maximum(x, 0.0), case)
    assert np.array_equal(case.tensors["x"], before)


def test_writeable_flag_is_restored_after_the_call():
    """Scoped to the call: downstream consumers, and Phase 3's torch.from_numpy,
    see an unchanged Case."""
    case = _case()
    assert case.tensors["x"].flags.writeable
    NumpyBackend().run(lambda x: x + 1.0, case)
    assert case.tensors["x"].flags.writeable


def test_writeable_flag_is_restored_even_when_the_kernel_raises():
    def boom(x):
        raise RuntimeError("nope")

    case = _case()
    NumpyBackend().run(boom, case)
    assert case.tensors["x"].flags.writeable


def test_readonly_inputs_restores_original_flags():
    arrays = {"a": np.zeros(3), "b": np.zeros(3)}
    arrays["b"].flags.writeable = False
    with readonly_inputs(arrays) as inner:
        assert not inner["a"].flags.writeable
        assert not inner["b"].flags.writeable
    assert arrays["a"].flags.writeable
    assert not arrays["b"].flags.writeable


@pytest.mark.parametrize(
    ("name", "kernel"),
    [
        ("relu", lambda x: np.maximum(x, 0.0)),
        ("relu_out", lambda x: np.maximum(x, 0.0, out=np.empty_like(x))),
        ("row_softmax", lambda x: np.exp(x - x.max(-1, keepdims=True))
         / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)),
        ("layernorm", lambda x: (x - x.mean(-1, keepdims=True))
         / np.sqrt(x.var(-1, keepdims=True) + 1e-5)),
        ("gelu", lambda x: 0.5 * x * (1.0 + np.tanh(0.797885 * (x + 0.044715 * x**3)))),
        ("sum_reduction", lambda x: np.sum(x, axis=-1)),
        ("identity", lambda x: x),
        ("transpose_view", lambda x: x.T),
        ("reshape_view", lambda x: x.reshape(-1)),
        ("prealloc_out", lambda x: _prealloc(x)),
    ],
)
def test_legitimate_kernels_are_unaffected_by_readonly_inputs(name, kernel):
    result = NumpyBackend().run(kernel, _case_2d())
    assert result.status == Status.OK, f"{name}: {result.error}"


def _prealloc(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    out[...] = x * 2.0
    return out


# --- Error reporting and telemetry ---


def test_wall_ms_is_measured_on_the_error_path():
    def slow_boom(x):
        for _ in range(200_000):
            pass
        raise RuntimeError("late failure")

    result = NumpyBackend().run(slow_boom, _case())
    assert result.status == Status.LAUNCH_ERROR
    assert result.telemetry["wall_ms"] > 0.0


def test_error_carries_a_traceback_without_duplicating_the_message():
    def boom(x):
        raise ValueError("bad")

    result = NumpyBackend().run(boom, _case())
    # A standard traceback, not a message prepended to one: it starts with the
    # "Traceback" header and ends with the exception line.
    assert result.error.startswith("Traceback (most recent call last):")
    assert result.error.rstrip().endswith("ValueError: bad")


def test_backend_satisfies_the_protocol():
    from autokernel_pbt.props.backends.base import Backend

    backend: Backend = NumpyBackend()
    assert backend.name == "numpy"


def test_status_wire_values_are_stable():
    assert Status.OK == "ok"
    assert Status.LAUNCH_ERROR == "launch_error"
    assert Status.OUTPUT_ERROR == "output_error"
    assert Status.COMPILE_ERROR == "compile_error"
    assert Status.TIMEOUT == "timeout"


def test_status_survives_json_and_reconstruction():
    import json

    assert json.dumps({"status": Status.OK}) == '{"status": "ok"}'
    assert Status("launch_error") is Status.LAUNCH_ERROR


def test_status_renders_as_its_wire_value_not_its_name():
    """A bare str-mixin Enum renders as the value on 3.10/3.11 and as
    "Status.OK" on 3.12+, so the same detail string would differ across the
    range `requires-python = ">=3.10"` declares. Pinned by the explicit
    `__str__`/`__format__` on `Status`.
    """
    assert str(Status.OK) == "ok"
    assert f"{Status.OK}" == "ok"
    assert "{}".format(Status.OK) == "ok"  # noqa: UP032 - format() is under test
    assert f"status={Status.LAUNCH_ERROR}" == "status=launch_error"


def test_status_column_round_trips_through_parquet(tmp_path):
    """The wire value is what Task 7 writes and Task 9 reads back."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist([{"status": Status.OK}]), path)
    value = pq.read_table(path).to_pylist()[0]["status"]
    assert value == "ok"
    assert Status(value) is Status.OK
