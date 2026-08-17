"""Triton ports of the development ladder.

The tile width is DERIVED from each shape rather than fixed. The first hardware run
used a single BLOCK=16384 for everything, which had two consequences worth recording:
every shape compiled to the same artifact, so all compiled telemetry was constant
across the run and carried no signal; and the kernels were silently wrong for any row
wider than the block, which the corpus never reached. Deriving it fixes both and is
what a competent Triton kernel does anyway.

This is not a block-size *sweep* -- there is still exactly one configuration per
shape, so the tier-1 transfer question stays unconfounded. Sweeping BLOCK independently
of shape remains Phase 3b's.

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

#: Largest tile these kernels support. One program holds a whole row in registers, so
#: a row wider than this needs a two-stage (multi-block) reduction, which is a
#: different kernel and out of scope here -- the guard in `_launcher` refuses it
#: rather than computing a wrong answer.
MAX_BLOCK = 16384


def block_for(n_cols: int) -> int:
    """The tile width for a row of `n_cols`, rounded up to a power of two.

    DERIVED, not fixed. A fixed BLOCK is wrong in both directions: too small and the
    kernel silently drops the tail of every row (measured: BLOCK=2048 on n_cols=4096
    makes softmax rows sum to 1.51 instead of 1.0, with no error raised), too large
    and every shape compiles to the same artifact -- which made `n_regs`, `n_spills`,
    `shared_bytes`, `num_warps` and `num_stages` *constant across the entire first
    hardware run*, since those are properties of the compiled kernel and Triton
    compiles one artifact per constexpr combination.

    Deriving it is also simply what a competent Triton kernel does.
    """
    return max(1 << (n_cols - 1).bit_length(), 1)


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

    def launch(*, grid, constexprs, record_compiled, **inputs):
        x = inputs["x"]
        cols = x.shape[-1]
        block = constexprs["BLOCK"]
        if cols > block:
            # A bad *call*, not bad data: it can only come from a misconfigured
            # kernel, costs nothing to re-run, and the alternative is silent
            # wrongness -- `tl.arange(0, BLOCK)` simply never reaches the tail, so
            # the kernel returns a plausible, incorrect answer.
            msg = (
                f"BLOCK={block} is smaller than n_cols={cols}; this kernel holds a "
                f"whole row in one tile, and a smaller tile would silently drop the "
                f"row's tail. Rows wider than {MAX_BLOCK} need a two-stage reduction."
            )
            raise ValueError(msg)
        xd = torch.as_tensor(x, device="cuda")
        yd = torch.empty_like(xd)

        # The input-mutation check, on the device buffer -- the only place it can
        # actually fire. See InputMutatedError for why the backend cannot do this.
        before = device_digest(xd)
        # Triton's launch RETURNS the CompiledKernel. Discarding it leaves the
        # adapter with no artifact and every compiled telemetry field MISSING.
        compiled = jit_kernel[grid](xd, yd, cols, **constexprs)
        record_compiled(compiled)
        if device_digest(xd) != before:
            msg = "kernel modified its input tensor(s) ['x'] on device"
            raise InputMutatedError(msg)

        # No astype: the output dtype should be whatever the kernel produced, so a
        # dtype defect stays visible instead of being cast away here.
        return yd.cpu().numpy()

    return launch


def relu_kernel(n_cols: int) -> TritonKernel:
    return TritonKernel(
        kernel_id="relu_triton",
        jit_fn=_relu_kernel,
        grid=_rows_grid,
        constexprs={"BLOCK": block_for(n_cols)},
        launcher=_launcher(_relu_kernel),
    )


def softmax_kernel(n_cols: int) -> TritonKernel:
    return TritonKernel(
        kernel_id="softmax_triton",
        jit_fn=_softmax_kernel,
        grid=_rows_grid,
        constexprs={"BLOCK": block_for(n_cols)},
        launcher=_launcher(_softmax_kernel),
    )


def layernorm_kernel(n_cols: int) -> TritonKernel:
    # EPS matches tasks.LAYERNORM_EPS. A kernel targeting a different eps sits
    # systematically away from the reference and would inflate every
    # tolerance-bearing arm's rate for a reason that is not a defect.
    from autokernel_pbt.props.tasks import LAYERNORM_EPS

    return TritonKernel(
        kernel_id="layernorm_triton",
        jit_fn=_layernorm_kernel,
        grid=_rows_grid,
        constexprs={"BLOCK": block_for(n_cols), "EPS": LAYERNORM_EPS},
        launcher=_launcher(_layernorm_kernel),
    )


#: The tolerance sweep is softmax over wider reductions, so it reuses that kernel.
KERNELS = {
    "relu": relu_kernel,
    "softmax": softmax_kernel,
    "layernorm": layernorm_kernel,
    "tolerance_sweep": softmax_kernel,
    "softmax_at_scale": softmax_kernel,
}
