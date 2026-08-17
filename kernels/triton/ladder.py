"""Triton ports of the development ladder.

One fixed launch configuration per task, deliberately. Sweeping BLOCK_SIZE would
multiply hardware time and confound the tier-1 transfer question -- does a property
that holds on NumPy hold on Triton -- with a block-size study. That study is Phase 3b.

Each kernel matches its task's existing `kernels/tasks/<id>/acceptance.yaml`. The
softmax and layernorm kernels subtract the row max and the row mean respectively, for
the same reason their NumPy references do: without it the declarative arm's shift
invariance property would be judging a kernel that is wrong for a reason unrelated to
the backend, which is not the question this run asks.

This module imports torch and triton at module scope, so nothing CPU-side may import
it. The device tests import it inside the test body for exactly that reason.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from autokernel_pbt.props.backends.triton_kernel import (
    InputMutatedError,
    TritonKernel,
    device_digest,
)

#: One row per program, with the whole row in registers. Covers the ladder (max
#: reduction length 129) and the tolerance sweep (max 16384) with one configuration,
#: which is what keeps the first run a clean transfer measurement rather than a
#: block-size study.
BLOCK_SIZE = 16384


@triton.jit
def _relu_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    tl.store(y_ptr + row * n_cols + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    # other=-inf so masked lanes cannot win the max on a partial tile.
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    # Masked lanes are exp(-inf) = 0 and contribute nothing to the sum.
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def _layernorm_kernel(x_ptr, y_ptr, n_cols, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    # Re-mask after centering: masked lanes hold 0.0, and 0 - mean is not 0, so an
    # unmasked centered tile would poison the variance with its tail.
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    tl.store(y_ptr + row * n_cols + offs, centered / tl.sqrt(var + EPS), mask=mask)


def _rows_grid(shape, constexprs):
    """One program per row. `shape` is the primary input's shape."""
    return (shape[0],)


def _launcher(jit_kernel):
    """Build a launcher that uses the grid and constexprs the adapter handed it.

    Not its own: the adapter records what it passes as launch telemetry, so a launcher
    that recomputed either could make the artifacts describe a launch that never
    happened.
    """

    def launch(*, grid, constexprs, **inputs):
        x = inputs["x"]
        cols = x.shape[-1]
        xd = torch.as_tensor(x, device="cuda")
        yd = torch.empty_like(xd)

        # The input-mutation check, on the device buffer -- the only place it can
        # actually fire. See InputMutatedError for why the backend cannot do this.
        before = device_digest(xd)
        jit_kernel[grid](xd, yd, cols, **constexprs)
        if device_digest(xd) != before:
            msg = "kernel modified its input tensor(s) ['x'] on device"
            raise InputMutatedError(msg)

        # No astype: the output dtype should be whatever the kernel produced, so a
        # dtype defect stays visible instead of being cast away here.
        return yd.cpu().numpy()

    return launch


def relu_kernel() -> TritonKernel:
    return TritonKernel(
        kernel_id="relu_triton",
        jit_fn=_relu_kernel,
        grid=_rows_grid,
        constexprs={"BLOCK": BLOCK_SIZE},
        launcher=_launcher(_relu_kernel),
    )


def softmax_kernel() -> TritonKernel:
    return TritonKernel(
        kernel_id="softmax_triton",
        jit_fn=_softmax_kernel,
        grid=_rows_grid,
        constexprs={"BLOCK": BLOCK_SIZE},
        launcher=_launcher(_softmax_kernel),
    )


def layernorm_kernel() -> TritonKernel:
    # EPS matches tasks.LAYERNORM_EPS. A kernel targeting a different eps sits
    # systematically away from the reference and would inflate every
    # tolerance-bearing arm's rate for a reason that is not a defect.
    from autokernel_pbt.props.tasks import LAYERNORM_EPS

    return TritonKernel(
        kernel_id="layernorm_triton",
        jit_fn=_layernorm_kernel,
        grid=_rows_grid,
        constexprs={"BLOCK": BLOCK_SIZE, "EPS": LAYERNORM_EPS},
        launcher=_launcher(_layernorm_kernel),
    )


#: The tolerance sweep is softmax over wider reductions, so it reuses that kernel.
KERNELS = {
    "relu": relu_kernel,
    "softmax": softmax_kernel,
    "layernorm": layernorm_kernel,
    "tolerance_sweep": softmax_kernel,
}
