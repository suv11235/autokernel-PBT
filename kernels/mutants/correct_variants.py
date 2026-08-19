"""Correct kernels that are not bit-identical to their references.

The false-positive denominator. A correct kernel that reproduces the reference
bit-for-bit measures nothing: every tolerance-bearing arm is handed a residual of
exactly zero and passes vacuously. The realistic false-positive risk is a kernel that
is *right* but differs in the last few ulps, and that risk is already measured --
`allclose` flags 5 of 9 layernorm groups for exactly this reason.

Each variant is algebraically equivalent to its reference and numerically different.
The admission gate is the check: a correct variant must be REJECTED as "not broken on
any group". A variant the gate admits is not a correct variant, it is a mutant.
"""

from __future__ import annotations

import triton
import triton.language as tl

from autokernel_pbt.props.backends.triton_kernel import TritonKernel
from kernels.triton.ladder import _launcher, _rows_grid, block_for

#: log2(e). exp(x) == exp2(x * LOG2E) exactly in real arithmetic; in floating point
#: the two differ in the last bits, which is the point.
LOG2E = 1.4426950408889634


@triton.jit
def _softmax_exp2(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    """Correct: exponentiates base-2 rather than base-e.

    Hardware has a native exp2, so this is what a performance-minded author writes.
    Algebraically identical, numerically different in the final bits.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = (x - tl.max(x, axis=0)) * 1.4426950408889634
    e = tl.exp2(x)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def _softmax_reciprocal(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    """Correct: multiplies by the reciprocal instead of dividing.

    One reciprocal and N multiplies rather than N divides -- a standard optimization,
    and it rounds differently because the reciprocal is itself rounded first.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    inv = 1.0 / tl.sum(e, axis=0)
    tl.store(y_ptr + row * n_cols + offs, e * inv, mask=mask)


@triton.jit
def _layernorm_rsqrt(x_ptr, y_ptr, n_cols, EPS: tl.constexpr, BLOCK: tl.constexpr):
    """Correct: multiplies by the reciprocal square root instead of dividing by sqrt.

    Hardware has a native rsqrt, so this is the form a performance-minded author
    writes. Same two-pass variance as the reference -- only the final scaling differs,
    which is exactly the "right but off in the last bits" case a false-positive rate
    needs. Unlike the sum-of-squares formulation it does not lose significance.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    tl.store(y_ptr + row * n_cols + offs, centered * (1.0 / tl.sqrt(var + EPS)), mask=mask)


_SOFTMAX_VARIANTS = {
    "softmax_exp2": _softmax_exp2,
    "softmax_reciprocal": _softmax_reciprocal,
}
_LAYERNORM_VARIANTS = {"layernorm_rsqrt": _layernorm_rsqrt}

VARIANTS = {**_SOFTMAX_VARIANTS, **_LAYERNORM_VARIANTS}


def correct_variant(name: str, n_cols: int) -> TritonKernel:
    jit_fn = VARIANTS[name]
    constexprs = {"BLOCK": block_for(n_cols)}
    if name in _LAYERNORM_VARIANTS:
        from autokernel_pbt.props.tasks import LAYERNORM_EPS

        constexprs["EPS"] = LAYERNORM_EPS
    return TritonKernel(
        kernel_id=name,
        jit_fn=jit_fn,
        grid=_rows_grid,
        constexprs=constexprs,
        launcher=_launcher(jit_fn),
    )


TASK_OF = {
    "softmax_exp2": "softmax",
    "softmax_reciprocal": "softmax",
    "layernorm_rsqrt": "layernorm",
}

# --------------------------------------------------------------------------- #
# Intended as a correct variant; the gate said otherwise.
# --------------------------------------------------------------------------- #


@triton.jit
def _layernorm_sumsq(x_ptr, y_ptr, n_cols, EPS: tl.constexpr, BLOCK: tl.constexpr):
    """Variance as E[x^2] - E[x]^2. Written as a correct variant, ADMITTED as a mutant.

    Algebraically identical to the reference and a standard textbook optimization --
    one pass instead of two. On device against a uniform(-10, 10) input it crosses the
    correctness threshold, because the two terms are close in magnitude and their
    difference loses most of its significant bits. Catastrophic cancellation is
    textbook; that it is severe enough to be a *bug* at this scale was not predicted
    here, and the gate is what caught it.

    Kept deliberately. It is a defect an optimizing agent would plausibly introduce
    while believing it was refactoring, which is the class of bug this corpus is for.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    sq = tl.where(mask, x * x, 0.0)
    var = tl.sum(sq, axis=0) / n_cols - mean * mean
    centered = tl.where(mask, x - mean, 0.0)
    tl.store(y_ptr + row * n_cols + offs, centered / tl.sqrt(var + EPS), mask=mask)


#: Not a variant. Verified broken by the admission gate; see the docstring above.
FOUND_MUTANTS = {"layernorm_sumsq": _layernorm_sumsq}
