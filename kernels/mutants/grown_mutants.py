"""Fifteen Triton mutants authored under the blinding protocol, three per class.

Each of five agents received exactly two things: the correct Triton softmax kernel and
one verbatim ISSTA subcategory description. None saw the property set, any contract,
any tolerance, any other agent's output, or any indication that an oracle layer
exists. Each was asked for three MECHANISTICALLY DIFFERENT routes to its fault class,
and for at least one subtle mutant whose output is close to correct.

Bodies are VERBATIM as returned, including one that does not compile. Editing them
would reintroduce the contamination the protocol excludes, and a mutant repaired by
someone who knows the properties is no longer a blinded mutant -- the gate is where
unusable candidates are removed, not the editor.

KNOWN LIMITATION, recorded rather than hidden: three mutants from one agent are less
independent than three from three agents. That was a deliberate trade for tractability
and belongs in any threats section quoting these numbers.

CROSS-CLASS COLLISION, measured: `operator_1` and `indexing_1` are character-identical
despite being authored from different subcategory descriptions. The `intended_class`
label is therefore demonstrably not a partition -- direct evidence for why the gate
does not attempt class verification.
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
def _operator_1(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    row_start = row * BLOCK
    x = tl.load(x_ptr + row_start + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row_start + offs, e / tl.sum(e, axis=0), mask=mask)

@triton.jit
def _operator_2(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    dtype = x_ptr.dtype.element_ty
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = (x - tl.max(x, axis=0)).to(dtype)
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    tl.store(y_ptr + row * n_cols + offs, (e / s).to(dtype), mask=mask)

@triton.jit
def _operator_3(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)

@triton.jit
def _dtype_1(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x.to(tl.float16)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    tl.store(y_ptr + row * n_cols + offs, (e / denom).to(x.dtype), mask=mask)

@triton.jit
def _dtype_2(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    neg_big = tl.full((1,), -1.0, tl.float32) * tl.zeros((1,), tl.float32).dtype.element_ty(0)
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-3.0e38)
    x = x.to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp((x - m).to(x_ptr.dtype.element_ty).to(tl.float32))
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)

@triton.jit
def _dtype_3(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0).to(x_ptr.dtype.element_ty)
    shifted = (x.to(x_ptr.dtype.element_ty) - m)
    e = tl.exp(shifted.to(tl.float32))
    s = tl.sum(e.to(x_ptr.dtype.element_ty).to(tl.float32), axis=0)
    tl.store(y_ptr + row * n_cols + offs, e / s, mask=mask)

@triton.jit
def _indexing_1(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * BLOCK + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row * BLOCK + offs, e / tl.sum(e, axis=0), mask=mask)

@triton.jit
def _indexing_2(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    rows = row + tl.arange(0, 1)
    offs = tl.arange(0, BLOCK)
    ptrs = x_ptr + rows[:, None] * n_cols + offs[None, :]
    mask = offs[None, :] < n_cols
    x = tl.load(ptrs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + rows[:, None] * n_cols + offs[None, :], e / tl.sum(e, axis=0), mask=mask)

@triton.jit
def _indexing_3(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    row_stride = n_cols + 1
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)

@triton.jit
def _predicate_1(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs <= n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)

@triton.jit
def _predicate_2(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0))

@triton.jit
def _predicate_3(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.where(mask, tl.exp(x), 1.0)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)

@triton.jit
def _special_1(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    x = tl.maximum(x, -12.0)
    e = tl.exp(x)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)

@triton.jit
def _special_2(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x) / tl.exp(m)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)

@triton.jit
def _special_3(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = tl.where(x < float("inf"), x, 0.0)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)

_KERNELS = {
    "operator_1": _operator_1,
    "operator_2": _operator_2,
    "operator_3": _operator_3,
    "dtype_1": _dtype_1,
    "dtype_2": _dtype_2,
    "dtype_3": _dtype_3,
    "indexing_1": _indexing_1,
    "indexing_2": _indexing_2,
    "indexing_3": _indexing_3,
    "predicate_1": _predicate_1,
    "predicate_2": _predicate_2,
    "predicate_3": _predicate_3,
    "special_1": _special_1,
    "special_2": _special_2,
    "special_3": _special_3,
}

SUBCATEGORIES = tuple(_KERNELS)

CLASS_OF = {
    "operator_1": "operator_implementation",
    "operator_2": "operator_implementation",
    "operator_3": "operator_implementation",
    "dtype_1": "data_type_semantics",
    "dtype_2": "data_type_semantics",
    "dtype_3": "data_type_semantics",
    "indexing_1": "indexing_and_stride",
    "indexing_2": "indexing_and_stride",
    "indexing_3": "indexing_and_stride",
    "predicate_1": "branch_predication",
    "predicate_2": "branch_predication",
    "predicate_3": "branch_predication",
    "special_1": "special_value_handling",
    "special_2": "special_value_handling",
    "special_3": "special_value_handling",
}


def grown_mutant(name: str, n_cols: int) -> tuple[Mutant, TritonKernel]:
    jit_fn = _KERNELS[name]
    adapter = TritonKernel(
        kernel_id=f"softmax_{name}",
        jit_fn=jit_fn,
        grid=lambda shape, ce: (shape[0],),
        constexprs={"BLOCK": block_for(n_cols)},
        launcher=_launcher(jit_fn),
    )
    record = Mutant(
        kernel_id=adapter.kernel_id,
        task_id="softmax",
        intended_class=f"type_and_operator/{CLASS_OF[name]}",
        taxonomy_quote=TAXONOMY[CLASS_OF[name]],
        backend="triton",
        fn=adapter,
    )
    return record, adapter
