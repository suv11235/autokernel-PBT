"""TritonBackend tests, CPU-only.

Status mapping, telemetry assembly and error classification are all structural and are
tested here with an injected fake device probe. The device path itself is covered by
gpu-marked tests in tests/gpu/, run by hand on the instance.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, Status
from autokernel_pbt.props.backends.telemetry import MISSING, TELEMETRY_SCHEMA_VERSION
from autokernel_pbt.props.backends.triton_backend import (
    TritonBackend,
    TritonCompilationError,
    _default_device_probe,
)
from autokernel_pbt.props.backends.triton_kernel import InputMutatedError, TritonKernel
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.table import ExecutionTable


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


def _install_triton_errors(module_name: str) -> type[Exception]:
    """Plant a `CompilationError` where some Triton release keeps it.

    Shaped like the real one: it derives from a Triton base class, not from anything
    the backend defines, so only a backend that looks the type up in Triton itself can
    classify it. The `device_stubs` fixture removes every `triton.*` module afterwards.
    """

    class TritonError(Exception):
        pass

    class CompilationError(TritonError):
        pass

    CompilationError.__module__ = module_name
    parent, _, leaf = module_name.rpartition(".")
    module = types.ModuleType(module_name)
    module.CompilationError = CompilationError
    sys.modules[module_name] = module
    if parent not in sys.modules:
        sys.modules[parent] = types.ModuleType(parent)
    setattr(sys.modules[parent], leaf, module)
    return CompilationError


def _raising(exc_type: type[Exception], msg: str):
    def launcher(**kw):
        raise exc_type(msg)

    return launcher


def test_compile_failure_maps_to_compile_error(device_stubs):
    """The criterion COMPILE_ERROR_IS_DISTINGUISHED_FROM_LAUNCH_ERROR.

    Triton compiles on first call, so a compile error arrives during execution and
    would otherwise be indistinguishable from a launch failure. They mean different
    things: a kernel that never compiled says nothing about numerics, while one that
    launched and crashed may. Status.COMPILE_ERROR has existed unused since phase 1.

    The exception is Triton's OWN `CompilationError`, which is what `jit_fn[grid](...)`
    actually raises. A launcher re-raising it as the backend's type is not something
    any ladder or mutant launcher does, so testing only that type certified a mapping
    that never fired on device.
    """
    compilation_error = _install_triton_errors("triton.compiler.errors")
    launcher = _raising(compilation_error, "at 7:8: arange's range must be a power of 2")

    result = _backend().run(_kernel(launcher=launcher), _case())
    assert result.status is Status.COMPILE_ERROR
    assert "arange's range must be a power of 2" in result.error


def test_compile_failure_is_classified_where_an_older_triton_keeps_it(device_stubs):
    # Probed, not imported from one fixed path: the type's home is not a stable
    # surface, and a version that moved it must not reclassify every compile failure.
    compilation_error = _install_triton_errors("triton.compiler")
    launcher = _raising(compilation_error, "at 3:0: unexpected type")
    assert _backend().run(_kernel(launcher=launcher), _case()).status is Status.COMPILE_ERROR


def test_the_backend_s_own_compile_error_type_maps_to_compile_error():
    # For a launcher that detects a compile failure itself and says so explicitly.
    launcher = _raising(TritonCompilationError, "at 3:0: unexpected type")
    result = _backend().run(_kernel(launcher=launcher), _case())
    assert result.status is Status.COMPILE_ERROR
    assert "unexpected type" in result.error


def test_a_same_named_exception_from_elsewhere_is_not_a_compile_error(device_stubs):
    # Classification is by Triton's type, never by name or message substring.
    _install_triton_errors("triton.compiler.errors")

    class CompilationError(Exception):
        pass

    launcher = _raising(CompilationError, "CompilationError: not from triton")
    assert _backend().run(_kernel(launcher=launcher), _case()).status is Status.LAUNCH_ERROR


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


def test_a_failing_device_probe_degrades_the_device_group_to_missing():
    """A sticky CUDA error poisons every later `torch.cuda.*` call, the probe included.

    The probe runs in the failure handlers too, so before this guard the kernel's own
    illegal-address fault re-raised from inside its error path and escaped `run` --
    aborting a paid run on precisely the row that most needed recording.
    """

    def sticky_probe():
        msg = "CUDA error: an illegal memory access was encountered"
        raise RuntimeError(msg)

    launcher = _raising(RuntimeError, "CUDA error: an illegal memory access was encountered")
    result = _backend(device_probe=sticky_probe).run(_kernel(launcher=launcher), _case())
    assert result.status is Status.LAUNCH_ERROR
    assert result.telemetry["device_name"] is MISSING


def test_an_unencodable_launch_value_survives_the_table_write(tmp_path):
    """The write happens once, after every execution has been paid for.

    So a single value the table cannot encode used to raise from `ExecutionTable.write`
    and discard the whole run. Here it is a launch constexpr, the group that is merged
    rather than probed; the compiled and device groups have their own tests in
    test_telemetry.py. End to end through the real table, because "encodable" means
    whatever the table's writer accepts.
    """

    class _Opaque:
        """A constexpr object a launcher author might pass through as-is."""

    result = _backend().run(_kernel(constexprs={"BLOCK": _Opaque()}), _case())
    assert result.status is Status.OK

    ExecutionTable(tmp_path / "run").write([result])
    (row,) = ExecutionTable(tmp_path / "run").read()
    assert row.telemetry["constexprs"] is None


def _fake_torch(*, driver: object) -> types.ModuleType:
    """torch as `_default_device_probe` sees it, with a chosen driver query."""
    torch = types.ModuleType("torch")
    torch.__version__ = "2.7.0"
    # The CUDA toolkit torch was BUILT against -- not the installed driver.
    torch.version = types.SimpleNamespace(cuda="12.8")
    properties = types.SimpleNamespace(
        name="NVIDIA A10", major=8, minor=6, multi_processor_count=72, total_memory=24 << 30
    )
    torch.cuda = types.SimpleNamespace(
        current_device=lambda: 0, get_device_properties=lambda index: properties
    )
    torch._C = types.SimpleNamespace()
    if driver is not None:
        torch._C._cuda_getDriverVersion = driver
    return torch


def test_driver_version_is_the_installed_driver_not_torch_s_build_toolkit(monkeypatch):
    # cuDriverGetVersion encodes 1000*major + 10*minor; 12090 is a CUDA 12.9 driver.
    # Recorded in runtime_version's own "major.minor" form so the two compare directly.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(driver=lambda: 12090))
    monkeypatch.setitem(sys.modules, "triton", None)
    probed = _default_device_probe()
    assert probed["driver_version"] == "12.9"
    assert probed["runtime_version"] == "12.8"


def _raises():
    msg = "no CUDA driver"
    raise RuntimeError(msg)


@pytest.mark.parametrize(
    "driver",
    [None, _raises, lambda: 0, lambda: "12090"],
    ids=["query-absent", "query-raises", "no-driver", "not-an-int"],
)
def test_an_unreadable_driver_version_is_missing(monkeypatch, driver):
    # Never torch's build-time toolkit as a stand-in: that is the mislabel being fixed.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(driver=driver))
    monkeypatch.setitem(sys.modules, "triton", None)
    assert _default_device_probe()["driver_version"] is MISSING
