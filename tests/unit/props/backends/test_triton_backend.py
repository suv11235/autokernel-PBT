"""TritonBackend tests, CPU-only.

Status mapping, telemetry assembly and error classification are all structural and are
tested here with an injected fake device probe. The device path itself is covered by
gpu-marked tests in tests/gpu/, run by hand on the instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, Status
from autokernel_pbt.props.backends.telemetry import MISSING, TELEMETRY_SCHEMA_VERSION
from autokernel_pbt.props.backends.triton_backend import TritonBackend, TritonCompilationError
from autokernel_pbt.props.backends.triton_kernel import InputMutatedError, TritonKernel
from autokernel_pbt.props.case import Case


def _case() -> Case:
    return Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="relu",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.ones((2, 3), dtype=np.float32)},
    )


def _kernel(launcher=None, **overrides) -> TritonKernel:
    defaults = {
        "kernel_id": "k",
        "jit_fn": lambda: None,
        "grid": lambda shape, constexprs: (1,),
        "constexprs": {"BLOCK_SIZE": 64},
        "launcher": launcher or (lambda **kw: np.zeros((2, 3), dtype=np.float32)),
    }
    defaults.update(overrides)
    return TritonKernel(**defaults)


def _backend(**overrides) -> TritonBackend:
    # device_probe is injected so the whole backend is exercisable with no CUDA.
    defaults = {"device_probe": lambda: {"device_name": "fake", "compute_capability": "8.6"}}
    defaults.update(overrides)
    return TritonBackend(**defaults)


def test_a_successful_run_reports_ok_and_the_output():
    result = _backend().run(_kernel(), _case())
    assert result.status is Status.OK
    assert result.outputs[OUTPUT_NAME].shape == (2, 3)


def test_telemetry_carries_the_schema_version_and_device_group():
    result = _backend().run(_kernel(), _case())
    assert result.telemetry[TELEMETRY_SCHEMA_VERSION] == 1
    assert result.telemetry["device_name"] == "fake"


def test_telemetry_carries_the_launch_group():
    result = _backend().run(_kernel(), _case())
    assert result.telemetry["constexprs"] == {"BLOCK_SIZE": 64}
    assert result.telemetry["grid"] == [1]


def test_telemetry_carries_the_kernel_source_hash():
    # kernel_id labels the run; the hash identifies the code that produced it.
    result = _backend().run(_kernel(), _case())
    assert len(result.telemetry["kernel_source_hash"]) == 16


def test_compiled_fields_are_missing_when_nothing_compiled():
    # No artifact on a fake launcher. MISSING, not absent and not zero.
    result = _backend().run(_kernel(), _case())
    assert result.telemetry["n_regs"] is MISSING


def test_compile_failure_maps_to_compile_error():
    """The criterion COMPILE_ERROR_IS_DISTINGUISHED_FROM_LAUNCH_ERROR.

    Triton compiles on first call, so a compile error arrives during execution and
    would otherwise be indistinguishable from a launch failure. They mean different
    things: a kernel that never compiled says nothing about numerics, while one that
    launched and crashed may. Status.COMPILE_ERROR has existed unused since phase 1.
    """

    def boom(**kw):
        msg = "at 3:0: unexpected type"
        raise TritonCompilationError(msg)

    result = _backend().run(_kernel(launcher=boom), _case())
    assert result.status is Status.COMPILE_ERROR
    assert "unexpected type" in result.error


def test_launch_failure_maps_to_launch_error():
    def boom(**kw):
        msg = "an illegal memory access was encountered"
        raise RuntimeError(msg)

    assert _backend().run(_kernel(launcher=boom), _case()).status is Status.LAUNCH_ERROR


def test_a_bad_output_maps_to_output_error():
    result = _backend().run(_kernel(launcher=lambda **kw: None), _case())
    assert result.status is Status.OUTPUT_ERROR


def test_status_mapping_is_total():
    """The criterion STATUS_MAPPING_IS_TOTAL.

    Every exception a kernel can raise reaches exactly one Status, and none escapes.
    An escaping exception aborts a run whose executions have already been paid for.
    """

    class Weird(Exception):
        pass

    def boom(**kw):
        raise Weird

    result = _backend().run(_kernel(launcher=boom), _case())
    assert result.status in set(Status)
    assert result.status is Status.LAUNCH_ERROR


def test_an_input_mutation_reported_by_the_launcher_is_a_launch_error():
    """The device replacement for the host-side read-only guarantee.

    The check itself lives in the launcher, the only layer holding the device buffers
    -- see InputMutatedError. What the backend owes is classifying it as LAUNCH_ERROR
    rather than letting it escape and abort a paid run.
    """

    def mutating(**kw):
        msg = "kernel modified its input tensor(s) ['x'] on device"
        raise InputMutatedError(msg)

    result = _backend().run(_kernel(launcher=mutating), _case())
    assert result.status is Status.LAUNCH_ERROR
    assert "modified its input" in result.error
    # The flag is what makes this distinguishable from a plain crash in the
    # artifacts. InputMutatedError subclasses RuntimeError, so without it the
    # generic handler would swallow the distinction and nothing downstream could
    # separate "wrote to its input" from "crashed" -- different fault classes.
    assert result.telemetry["input_mutated"] is True


def test_an_ordinary_crash_does_not_set_the_mutation_flag():
    def boom(**kw):
        raise RuntimeError

    assert _backend().run(_kernel(launcher=boom), _case()).telemetry["input_mutated"] is False


def test_a_successful_run_does_not_set_the_mutation_flag():
    assert _backend().run(_kernel(), _case()).telemetry["input_mutated"] is False


def test_a_well_behaved_kernel_is_not_flagged():
    assert _backend().run(_kernel(), _case()).status is Status.OK


def test_telemetry_is_recorded_even_on_failure():
    # A kernel that dies after 30s is a different problem from one that dies at once,
    # and the device it died on is part of the finding.
    def boom(**kw):
        raise RuntimeError

    result = _backend().run(_kernel(launcher=boom), _case())
    assert result.telemetry["device_name"] == "fake"
    assert result.telemetry["wall_ms"] >= 0.0


def test_a_plain_callable_is_rejected_as_a_bad_call():
    # A bad call, not bad data: it can only come from a coding error and costs nothing
    # to re-run, so it raises rather than being recorded as a failed execution.
    with pytest.raises(TypeError, match="requires a TritonKernel adapter"):
        _backend().run(lambda **kw: np.zeros((2, 3), dtype=np.float32), _case())


def test_the_case_is_carried_through_unchanged():
    case = _case()
    assert _backend().run(_kernel(), case).case is case


def test_backend_is_named_for_the_report_layer():
    assert TritonBackend.name == "triton"
