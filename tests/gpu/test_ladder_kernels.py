"""Device tests for the Triton ladder kernels. Run by hand on the instance.

Deliberately NOT acceptance criteria: a criterion whose evidence only exists on rented
hardware would leave tests/spec red on every developer machine and in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, Status
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.tasks import REFERENCES, TASKS

pytestmark = pytest.mark.gpu

# Module-level, not inside a test: `@triton.jit` compiles from source and resolves
# names against the decorated function's *globals*. A `tl` imported into a test
# function's local scope is invisible to the compiler, which fails with a bare
# `NameError('tl is not defined')` from inside the Triton frontend. importorskip
# keeps a machine without triton skipping the module rather than failing collection.
triton = pytest.importorskip("triton", reason="triton is not installed")
tl = pytest.importorskip("triton.language", reason="triton is not installed")


@triton.jit
def _vandal_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    """Writes to its INPUT pointer: the defect the integrity check exists for."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    tl.store(x_ptr + row * n_cols + offs, x + 1.0, mask=mask)
    tl.store(y_ptr + row * n_cols + offs, x, mask=mask)


@triton.jit
def _uncompilable_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    """`tl.arange` requires a power-of-two range, so this is refused at compile time."""
    offs = tl.arange(0, 3)
    tl.store(y_ptr + offs, tl.load(x_ptr + offs))


@pytest.mark.parametrize("task_id", ["relu", "softmax", "layernorm"])
def test_the_triton_kernel_agrees_with_its_numpy_reference(task_id, torch_cuda, triton_module):
    from autokernel_pbt.props.backends.triton_backend import TritonBackend
    from kernels.triton.ladder import KERNELS

    backend = TritonBackend()
    task = TASKS[task_id]
    for group in Generator(task.domain, seed=0).generate(len(task.domain.shapes)):
        for case in group.cases:
            result = backend.run(KERNELS[task_id](case.shape[-1]), case)
            assert result.status is Status.OK, result.error
            expected = REFERENCES[task_id](x=case.tensors["x"])
            # Loose on purpose: this asserts the port is not grossly wrong. How close
            # it *should* be is the measurement the run exists to make, not a
            # threshold to assume in advance.
            assert np.allclose(result.outputs[OUTPUT_NAME], expected, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("task_id", ["relu", "softmax", "layernorm"])
def test_compiled_telemetry_is_populated_on_device(task_id, torch_cuda, triton_module):
    """The whole reason for the schema: these must not come back MISSING.

    If they do, `_COMPILED_FIELDS` needs another probe location for this Triton
    version -- which is exactly the kind of thing the smoke session exists to find
    while the instance is still up and the discovery is still cheap.
    """
    from autokernel_pbt.props.backends.telemetry import MISSING
    from autokernel_pbt.props.backends.triton_backend import TritonBackend
    from kernels.triton.ladder import KERNELS

    task = TASKS[task_id]
    group = Generator(task.domain, seed=0).generate(len(task.domain.shapes))[0]
    result = TritonBackend().run(KERNELS[task_id](group.base.shape[-1]), group.base)
    assert result.status is Status.OK, result.error
    for key in ("n_regs", "shared_bytes", "num_warps"):
        assert result.telemetry[key] is not MISSING, f"{key} came back MISSING on device"


def test_a_kernel_that_fails_to_compile_is_a_compile_error_on_device(torch_cuda, triton_module):
    """A REAL Triton compile failure, classified by the real backend.

    The CPU test plants a stand-in `triton.compiler.errors`; only this one proves the
    installed Triton raises a type `_triton_compile_error_types` actually finds. If it
    fails as LAUNCH_ERROR, this release keeps `CompilationError` somewhere new -- add
    the location to `_TRITON_COMPILE_ERROR_LOCATIONS`.
    """
    from autokernel_pbt.props.backends.triton_backend import TritonBackend
    from autokernel_pbt.props.backends.triton_kernel import TritonKernel
    from autokernel_pbt.props.case import Case
    from kernels.triton.ladder import _launcher

    kernel = TritonKernel(
        kernel_id="uncompilable",
        jit_fn=_uncompilable_kernel,
        grid=lambda shape, ce: (1,),
        constexprs={"BLOCK": 4},
        launcher=_launcher(_uncompilable_kernel),
    )
    case = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="relu",
        dtype="float32",
        shape=(1, 4),
        tensors={"x": np.ones((1, 4), dtype=np.float32)},
    )
    result = TritonBackend().run(kernel, case)
    assert result.status is Status.COMPILE_ERROR, result.error


def test_a_kernel_that_writes_to_its_input_is_caught_on_device(torch_cuda, triton_module):
    """The integrity check firing for real -- it cannot be exercised off-device.

    The CPU test in tests/unit/ only proves the backend CLASSIFIES the error. This
    proves it is actually raised, which is the half that could silently never fire.
    """
    from autokernel_pbt.props.backends.triton_kernel import InputMutatedError, TritonKernel
    from kernels.triton.ladder import _launcher, block_for

    kernel = TritonKernel(
        kernel_id="vandal",
        jit_fn=_vandal_kernel,
        grid=lambda shape, ce: (shape[0],),
        constexprs={"BLOCK": block_for(8)},
        launcher=_launcher(_vandal_kernel),
    )
    with pytest.raises(InputMutatedError):
        kernel(x=np.ones((4, 8), dtype=np.float32))


def test_correct_variants_are_rejected_by_the_admission_gate(torch_cuda, triton_module):
    """The false-positive population must be verified correct, not asserted correct.

    A "correct variant" the gate admits is not a variant but a mutant -- and one
    written for this population, ``layernorm_sumsq``, turned out to be exactly that.
    See ``docs/measurements/2026-08-19-false-positive-rate.md``.

    `verdict is not True` alone was vacuous: a variant that never ran (every row a
    LAUNCH_ERROR) is rejected as "not judgeable" and passed it. The rejection must be
    the specific one -- "not broken" -- and must rest on every group having run.
    """
    from autokernel_pbt.corpus.gate import Rejection, admit
    from autokernel_pbt.props.backends.triton_backend import TritonBackend
    from kernels.mutants.correct_variants import TASK_OF, VARIANTS, correct_variant

    backend = TritonBackend()
    for name in VARIANTS:
        task = TASKS[TASK_OF[name]]
        groups = Generator(task.domain, seed=0).generate(len(task.domain.shapes))
        rows = [
            backend.run(correct_variant(name, case.shape[-1]), case)
            for group in groups
            for case in group.cases
        ]
        verdict = admit(rows, reference_fn=REFERENCES[TASK_OF[name]])
        assert isinstance(verdict, Rejection), f"{name} was admitted as broken"
        assert verdict.reason.startswith("not broken"), f"{name}: {verdict.reason}"
        assert verdict.groups_judgeable == len(groups), (
            f"{name} ran to OK on only {verdict.groups_judgeable} of {len(groups)} groups"
        )
