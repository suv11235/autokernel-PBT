"""TritonKernel adapter tests, all CPU.

The adapter's *structure* is where wiring bugs live -- protocol conformance, launch
config recording, source identity -- and all of it is checkable without a GPU. The
device path is covered by gpu-marked tests run by hand on the instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OutputContractError
from autokernel_pbt.props.backends.telemetry import MISSING
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


class _FakeJITFunction:
    """Shaped like `triton.runtime.jit.JITFunction`, as far as hashing can see.

    An *instance*, not a function, so `inspect.getsource` raises TypeError on it -- as
    it does on the real one. It wraps the Python function as `.fn`, optionally carries
    the text Triton compiles as `.src`, and, like Triton 3.3, sets `__module__` and
    `__name__` but no `__qualname__`, with a repr naming the function and never its
    body. Anything hashing only those names cannot see an edit to the kernel.
    """

    def __init__(self, fn, src: str | None = None) -> None:
        self.fn = fn
        if src is not None:
            self.src = src
        self.__module__ = "kernels.triton.ladder"
        self.__name__ = "_softmax_kernel"

    def __repr__(self) -> str:
        return "JITFunction(kernels.triton.ladder:_softmax_kernel)"


def _body_correct(x_ptr, y_ptr, n_cols):  # pragma: no cover - hashed, never called
    return "correct body"


def _body_broken(x_ptr, y_ptr, n_cols):  # pragma: no cover - hashed, never called
    return "BROKEN body, different code"


def test_source_hash_covers_the_text_triton_compiles():
    # Same wrapped function, different `.src`: `.src` is what Triton parses and
    # compiles, so it is the body whose edit must change the identity.
    a = _adapter(jit_fn=_FakeJITFunction(_body_correct, src="def k():\n    return 1\n"))
    b = _adapter(jit_fn=_FakeJITFunction(_body_correct, src="def k():\n    return 2\n"))
    assert a.source_hash != b.source_hash


def test_source_hash_falls_back_to_the_wrapped_function_body():
    # No `.src`: the wrapped `.fn` is the next most faithful source of the body.
    a = _adapter(jit_fn=_FakeJITFunction(_body_correct))
    b = _adapter(jit_fn=_FakeJITFunction(_body_broken))
    assert a.source_hash != b.source_hash


@pytest.mark.parametrize("role", ["jit_fn", "launcher"])
def test_source_hash_is_missing_when_a_component_has_no_source(role):
    """A hash of a *name* is not an identity, and must not be recorded as one.

    `len` has no Python source. The old fallback hashed `module.qualname` instead, so
    two different kernels sharing a name shared a hash -- exactly what the field
    exists to rule out. MISSING says "not captured" honestly.
    """
    assert _adapter(**{role: len}).source_hash is MISSING


def test_compiled_is_reset_at_the_start_of_every_call():
    """An artifact describes the call that produced it, never a later one.

    One adapter serves every case of its shape. If a later call fails before
    recording -- a compile failure for a new specialization -- its row would
    otherwise report the previous call's registers and spills as its own.
    """
    sentinel = object()
    calls = []

    def launcher(*, grid, constexprs, record_compiled, **inputs):
        calls.append(None)
        if len(calls) == 1:
            record_compiled(sentinel)
            return np.zeros((2, 3), dtype=np.float32)
        msg = "compile failed for this specialization"
        raise RuntimeError(msg)

    adapter = _adapter(launcher=launcher)
    adapter(x=np.ones((2, 3), dtype=np.float32))
    assert adapter.compiled is sentinel
    with pytest.raises(RuntimeError, match="compile failed for this specialization"):
        adapter(x=np.ones((2, 3), dtype=np.float32))
    assert adapter.compiled is None


def test_a_call_whose_grid_fails_does_not_report_the_previous_grid():
    # Same defect one field over: the grid is recorded before the launch, so a grid
    # callable that raises would leave the previous call's geometry in place.
    def grid(shape, constexprs):
        if shape[0] == 0:
            msg = "no rows to launch over"
            raise ValueError(msg)
        return (shape[0],)

    adapter = _adapter(grid=grid)
    adapter(x=np.ones((2, 3), dtype=np.float32))
    assert adapter.launch_telemetry()["grid"] == [2]
    with pytest.raises(ValueError, match="no rows to launch over"):
        adapter(x=np.ones((0, 3), dtype=np.float32))
    assert adapter.launch_telemetry()["grid"] is None


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
