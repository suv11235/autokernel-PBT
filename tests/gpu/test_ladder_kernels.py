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


@pytest.mark.parametrize("task_id", ["relu", "softmax", "layernorm"])
def test_the_triton_kernel_agrees_with_its_numpy_reference(task_id, torch_cuda, triton_module):
    from autokernel_pbt.props.backends.triton_backend import TritonBackend
    from kernels.triton.ladder import KERNELS

    backend = TritonBackend()
    kernel = KERNELS[task_id]()
    task = TASKS[task_id]
    for group in Generator(task.domain, seed=0).generate(len(task.domain.shapes)):
        for case in group.cases:
            result = backend.run(kernel, case)
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
    result = TritonBackend().run(KERNELS[task_id](), group.base)
    assert result.status is Status.OK, result.error
    for key in ("n_regs", "shared_bytes", "num_warps"):
        assert result.telemetry[key] is not MISSING, f"{key} came back MISSING on device"


def test_a_kernel_that_writes_to_its_input_is_caught_on_device(torch_cuda, triton_module):
    """The integrity check firing for real -- it cannot be exercised off-device.

    The CPU test in tests/unit/ only proves the backend CLASSIFIES the error. This
    proves it is actually raised, which is the half that could silently never fire.
    """
    import triton
    import triton.language as tl

    from autokernel_pbt.props.backends.triton_kernel import InputMutatedError, TritonKernel
    from kernels.triton.ladder import BLOCK_SIZE, _launcher

    @triton.jit
    def _vandal(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
        # Writes to the INPUT pointer: the defect the check exists for.
        tl.store(x_ptr + row * n_cols + offs, x + 1.0, mask=mask)
        tl.store(y_ptr + row * n_cols + offs, x, mask=mask)

    kernel = TritonKernel(
        kernel_id="vandal",
        jit_fn=_vandal,
        grid=lambda shape, ce: (shape[0],),
        constexprs={"BLOCK": BLOCK_SIZE},
        launcher=_launcher(_vandal),
    )
    with pytest.raises(InputMutatedError):
        kernel(x=np.ones((4, 8), dtype=np.float32))
