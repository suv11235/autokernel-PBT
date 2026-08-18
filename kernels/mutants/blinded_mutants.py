"""Triton mutants authored under the blinding protocol.

Each was produced by a separate agent given exactly two things: the correct Triton
softmax kernel, and one verbatim subcategory description from Table 2 of the ISSTA
2026 study. None saw `props/properties.py`, any `acceptance.yaml`, any tolerance, any
other mutant, or any indication that an oracle layer exists at all. Each was told not
to read or search the repository.

The kernel bodies below are **verbatim** as returned. They are not edited, tidied or
corrected -- editing them would reintroduce exactly the contamination the protocol
exists to exclude, and a mutant improved by someone who knows the properties is no
longer a blinded mutant.

Contrast `triton_mutants.py`, whose author had seen the whole oracle layer.
"""

from __future__ import annotations

import triton
import triton.language as tl

from autokernel_pbt.corpus.mutant import Mutant
from autokernel_pbt.props.backends.triton_kernel import TritonKernel
from kernels.triton.ladder import _launcher, block_for

TAXONOMY = {
    "operator_implementation": (
        "Operator logic is incorrect or incomplete after specialization for type or tile shape."
    ),
    "data_type_semantics": (
        "Loss of numeric meaning or precision from implicit casts or inconsistent "
        "mixed-precision handling across devices."
    ),
    "indexing_and_stride": (
        "Incorrect address computation due to invalid strides, broadcasts, or layout "
        "transformations."
    ),
    "branch_predication": (
        "Incorrect predicates or loop guards cause wrong control-flow decisions."
    ),
    "special_value_handling": (
        "Arithmetic fails to correctly propagate or detect NaN, infinities, or denormal values."
    ),
}


@triton.jit
def _b_operator(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    if BLOCK >= 1024:
        # wide rows: two-stage reduction over halves to keep registers down
        half = BLOCK // 2
        lo = tl.where(offs < half, x, -float("inf"))
        hi = tl.where(offs >= half, x, -float("inf"))
        m = tl.maximum(tl.max(lo, axis=0), tl.max(hi, axis=0))
        x = x - m
        e = tl.exp(x)
        d = tl.sum(tl.where(offs < half, e, 0.0), axis=0)
    else:
        x = x - tl.max(x, axis=0)
        e = tl.exp(x)
        d = tl.sum(e, axis=0)
    tl.store(y_ptr + row * n_cols + offs, e / d, mask=mask)


@triton.jit
def _b_dtype(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x).to(tl.float16)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def _b_indexing(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    row_start = row * BLOCK
    x = tl.load(x_ptr + row_start + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row_start + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def _b_predicate(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs <= n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def _b_special(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    m = tl.max(tl.where(mask, x, 0.0), axis=0)
    x = x - m
    e = tl.where(mask, tl.exp(x), 0.0)
    d = tl.sum(e, axis=0)
    tl.store(y_ptr + row * n_cols + offs, e / (d + 1e-6), mask=mask)


_KERNELS = {
    "operator_implementation": _b_operator,
    "data_type_semantics": _b_dtype,
    "indexing_and_stride": _b_indexing,
    "branch_predication": _b_predicate,
    "special_value_handling": _b_special,
}

SUBCATEGORIES = tuple(_KERNELS)


def blinded_mutant(subcategory: str, n_cols: int) -> tuple[Mutant, TritonKernel]:
    jit_fn = _KERNELS[subcategory]
    adapter = TritonKernel(
        kernel_id=f"softmax_{subcategory}_blinded",
        jit_fn=jit_fn,
        grid=lambda shape, ce: (shape[0],),
        constexprs={"BLOCK": block_for(n_cols)},
        launcher=_launcher(jit_fn),
    )
    record = Mutant(
        kernel_id=adapter.kernel_id,
        task_id="softmax",
        intended_class=f"type_and_operator/{subcategory}",
        taxonomy_quote=TAXONOMY[subcategory],
        backend="triton",
        fn=adapter,
    )
    return record, adapter
