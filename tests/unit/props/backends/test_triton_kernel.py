"""TritonKernel adapter tests, all CPU.

The adapter's *structure* is where wiring bugs live -- protocol conformance, launch
config recording, source identity -- and all of it is checkable without a GPU. The
device path is covered by gpu-marked tests run by hand on the instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OutputContractError
from autokernel_pbt.props.backends.triton_kernel import TritonKernel


def _fake_jit(name: str = "fake_kernel"):
    """A stand-in for a @triton.jit function: named, and never actually launched."""

    def kernel():  # pragma: no cover - never called in CPU tests
        msg = "the fake kernel was launched"
        raise AssertionError(msg)

    kernel.__name__ = name
    return kernel


def _adapter(**overrides) -> TritonKernel:
    defaults = {
        "kernel_id": "relu_triton",
        "jit_fn": _fake_jit(),
        "grid": lambda shape, constexprs: (1,),
        "constexprs": {"BLOCK_SIZE": 128},
        # **kw absorbs the grid= and constexprs= the adapter hands in.
        "launcher": lambda **kw: np.zeros((2, 3), dtype=np.float32),
    }
    defaults.update(overrides)
    return TritonKernel(**defaults)


def test_adapter_is_callable_with_numpy_kwargs():
    """The criterion ADAPTER_SATISFIES_THE_BACKEND_PROTOCOL.

    `Backend.run` types the kernel as `Callable[..., np.ndarray]` and calls it as
    `kernel(**kernel_inputs(case))`. The adapter must satisfy that unchanged, or the
    Triton backend would need its own protocol and the two backends would stop being
    substitutable -- which is what makes cross-backend comparison possible at all.
    """
    out = _adapter()(x=np.ones((2, 3), dtype=np.float32))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)


def test_launch_telemetry_records_the_constexprs():
    # BLOCK_SIZE and friends are what the tile compiler specialized on, and are the
    # ISSTA taxonomy's "tile mapping and launch" signal. Losing them costs a run.
    adapter = _adapter()
    adapter(x=np.ones((2, 3), dtype=np.float32))
    assert adapter.launch_telemetry()["constexprs"] == {"BLOCK_SIZE": 128}


def test_launch_telemetry_is_empty_before_the_first_call():
    # Nothing has launched, so there is no geometry to report. Reporting a grid here
    # would describe a launch that never happened.
    assert _adapter().launch_telemetry()["grid"] is None


def test_the_launcher_receives_exactly_the_grid_that_is_recorded():
    """Telemetry must describe the launch, not sit beside it.

    If the launcher computed its own grid, the recorded `grid` would be a label next
    to the behaviour rather than a description of it, and the two could drift apart
    silently -- with the artifacts reporting a launch geometry that never ran. That is
    the "asserted a label rather than the behaviour" defect this repo has hit four
    times, and launch geometry is the ISSTA taxonomy's own fault class.
    """
    seen = {}

    def launcher(*, grid, constexprs, **inputs):
        seen["grid"] = grid
        seen["constexprs"] = constexprs
        return np.zeros((2, 3), dtype=np.float32)

    adapter = _adapter(grid=lambda shape, ce: (7, 3, 1), launcher=launcher)
    adapter(x=np.ones((2, 3), dtype=np.float32))

    assert seen["grid"] == (7, 3, 1)
    assert seen["constexprs"] == {"BLOCK_SIZE": 128}
    assert adapter.launch_telemetry()["grid"] == [7, 3, 1]


def test_the_grid_callable_sees_the_primary_input_shape():
    seen = {}

    def grid(shape, constexprs):
        seen["shape"] = shape
        return (4,)

    _adapter(grid=grid)(x=np.ones((8, 16), dtype=np.float32))
    assert seen["shape"] == (8, 16)


def test_source_hash_distinguishes_two_kernels_sharing_a_name():
    # kernel_id is a label; the identity is the source. Two runs must not be able to
    # both call something "relu_triton" and mean different code.
    a = _adapter(jit_fn=_fake_jit("k"))
    b = _adapter(
        jit_fn=_fake_jit("k"),
        launcher=lambda **kw: np.ones((2, 3), dtype=np.float32),
    )
    assert a.source_hash != b.source_hash


def test_source_hash_is_stable_for_one_kernel():
    adapter = _adapter()
    assert adapter.source_hash == adapter.source_hash


def test_compiled_is_none_before_the_first_call():
    # Triton compiles lazily, so there is no artifact to read telemetry from until
    # the kernel has run at least once. The backend must not assume otherwise.
    assert _adapter().compiled is None


def test_compiled_is_populated_once_recorded():
    sentinel = object()
    adapter = _adapter()
    adapter._record_compiled(sentinel)
    assert adapter.compiled is sentinel


def test_the_launcher_is_given_a_working_record_compiled_callback():
    """Without this the compiled artifact is never captured.

    Triton's launch returns the CompiledKernel; a launcher that discards it leaves
    the adapter with `compiled = None`, and every compiled telemetry field -- n_regs,
    spills, shared memory -- reads MISSING. That is not a hypothetical: the first
    hardware run hit exactly this, and the schema looked complete while carrying
    nothing.
    """
    sentinel = object()

    def launcher(*, grid, constexprs, record_compiled, **inputs):
        record_compiled(sentinel)
        return np.zeros((2, 3), dtype=np.float32)

    adapter = _adapter(launcher=launcher)
    adapter(x=np.ones((2, 3), dtype=np.float32))
    assert adapter.compiled is sentinel


def test_a_launcher_returning_a_non_array_is_a_contract_error():
    with pytest.raises(OutputContractError):
        _adapter(launcher=lambda **kw: None)(x=np.ones((2, 3), dtype=np.float32))


def test_inputs_are_copied_before_reaching_the_launcher():
    """`readonly_inputs` makes the host arrays non-writeable during execution.

    `torch.from_numpy` warns on a non-writeable array, and this project turns
    warnings into errors -- so the adapter copies rather than aliasing. The copy is
    free relative to a host-to-device transfer, and `base.readonly_inputs`' own
    docstring flags this exact hazard as Phase 3's to solve.
    """
    x = np.ones((2, 3), dtype=np.float32)
    x.flags.writeable = False
    seen = {}

    def launcher(*, grid, constexprs, **inputs):
        seen["writeable"] = inputs["x"].flags.writeable
        return np.zeros((2, 3), dtype=np.float32)

    _adapter(launcher=launcher)(x=x)
    assert seen["writeable"] is True


def test_a_mutating_launcher_cannot_corrupt_the_caller_s_array():
    # The copy is not only about the warning: it also means a kernel that writes to
    # its host input cannot reach the recorded case tensors.
    x = np.ones((2, 3), dtype=np.float32)

    def launcher(*, grid, constexprs, **inputs):
        inputs["x"] += 1.0
        return np.zeros((2, 3), dtype=np.float32)

    _adapter(launcher=launcher)(x=x)
    assert np.array_equal(x, np.ones((2, 3), dtype=np.float32))
