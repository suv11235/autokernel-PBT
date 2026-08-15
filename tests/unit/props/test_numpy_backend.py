"""NumPy backend tests."""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import (
    STATUS_LAUNCH_ERROR,
    STATUS_OK,
    STATUS_OUTPUT_ERROR,
    kernel_inputs,
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


def test_successful_run_reports_ok():
    result = NumpyBackend().run(lambda x: np.maximum(x, 0.0), _case())
    assert result.status == STATUS_OK
    assert np.array_equal(result.outputs["y"], np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32))


def test_exception_is_captured_not_raised():
    def boom(x):
        raise RuntimeError("kernel exploded")

    result = NumpyBackend().run(boom, _case())
    assert result.status == STATUS_LAUNCH_ERROR
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

    assert result.status == STATUS_OK
    assert set(seen) == {"x"}
    assert "__perm__" not in seen


def test_kernel_inputs_filters_only_helper_prefixed_names():
    case = _case()
    case.tensors["__perm__"] = np.arange(4, dtype=np.int64)
    case.tensors["bias_"] = np.zeros(4, dtype=np.float32)
    assert set(kernel_inputs(case)) == {"x", "bias_"}


# --- Requirement C: multi-output kernels are rejected, not silently mangled ---


def test_tuple_output_is_rejected_with_a_clear_error():
    def two_outputs(x):
        return x, x + 1.0

    result = NumpyBackend().run(two_outputs, _case())
    assert result.status == STATUS_OUTPUT_ERROR
    assert "single" in result.error
    assert result.outputs == {}


def test_dict_output_is_rejected_with_a_clear_error():
    result = NumpyBackend().run(lambda x: {"y": x}, _case())
    assert result.status == STATUS_OUTPUT_ERROR
    assert result.outputs == {}


def test_none_output_is_rejected():
    result = NumpyBackend().run(lambda x: None, _case())
    assert result.status == STATUS_OUTPUT_ERROR


def test_scalar_output_is_accepted():
    """A reduction returning a Python/NumPy scalar is a legitimate single output."""
    result = NumpyBackend().run(lambda x: np.sum(x), _case())
    assert result.status == STATUS_OK
    assert result.outputs["y"].shape == ()


def test_list_of_arrays_is_rejected_rather_than_stacked():
    """np.asarray would silently stack these into a (2, 4) array."""
    result = NumpyBackend().run(lambda x: [x, x], _case())
    assert result.status == STATUS_OUTPUT_ERROR


# --- Requirement D: input aliasing is deliberate; pin it ---


def test_wellbehaved_kernel_does_not_mutate_case_inputs():
    case = _case()
    before = case.tensors["x"].copy()
    NumpyBackend().run(lambda x: np.maximum(x, 0.0), case)
    assert np.array_equal(case.tensors["x"], before)


def test_kernel_inputs_aliases_case_tensors_rather_than_copying():
    """Pinned behavior: no defensive copy.

    A kernel that writes to its input in place therefore corrupts the case,
    and Phase C would compare against the corrupted values. That is accepted
    for Phase 1 (copying every input doubles peak memory on real hardware);
    the mitigation is to persist inputs before execution, not to copy here.
    """
    case = _case()
    assert kernel_inputs(case)["x"] is case.tensors["x"]


def test_in_place_kernel_mutation_is_visible_on_the_case():
    def in_place(x):
        x *= 0.0
        return x

    case = _case()
    NumpyBackend().run(in_place, case)
    assert np.array_equal(case.tensors["x"], np.zeros(4, dtype=np.float32))


# --- Requirement E: the error path reports the time actually spent ---


def test_wall_ms_is_measured_on_the_error_path():
    def slow_boom(x):
        for _ in range(200_000):
            pass
        raise RuntimeError("late failure")

    result = NumpyBackend().run(slow_boom, _case())
    assert result.status == STATUS_LAUNCH_ERROR
    assert result.telemetry["wall_ms"] > 0.0


def test_error_carries_a_traceback():
    def boom(x):
        raise ValueError("bad")

    result = NumpyBackend().run(boom, _case())
    assert "Traceback" in result.error


def test_backend_satisfies_the_protocol():
    from autokernel_pbt.props.backends.base import Backend

    backend: Backend = NumpyBackend()
    assert backend.name == "numpy"


@pytest.mark.parametrize(
    "status",
    [STATUS_OK, STATUS_LAUNCH_ERROR, STATUS_OUTPUT_ERROR],
)
def test_status_constants_have_stable_wire_values(status):
    assert status in {"ok", "launch_error", "output_error"}
