"""Triton mutants, one per CPU-reachable ISSTA subcategory, for the softmax task.

PROVISIONAL -- THE BLINDING IS VIOLATED. `docs/protocol/mutant-authoring.md` requires
the authoring agent to see only the taxonomy quote and the correct reference, never
the property set. These were authored by the same agent that wrote
`props/properties.py` and every task contract, so they are contaminated by exactly
the knowledge the blinding exists to exclude.

They exist to prove the corpus pipeline end to end -- authoring, gating, recording,
scoring -- on real hardware. **No detection rate computed from them may be reported.**
The corpus must be re-authored by an agent that has not seen the oracle layer before
any number leaves this repository. See `docs/measurements/` for the standing caveat.

Each mutant carries the verbatim Table 2 description it was written from, so a
re-authored replacement can target the same class.
"""

from __future__ import annotations

import triton
import triton.language as tl

from autokernel_pbt.corpus.mutant import Mutant
from autokernel_pbt.props.backends.triton_kernel import TritonKernel
from kernels.triton.ladder import _launcher, block_for

#: subcategory -> verbatim Table 2 description (ISSTA 2026, arXiv:2605.19652).
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
def _m_operator(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    """Operator implementation: the normalization step is simply absent."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(y_ptr + row * n_cols + offs, e, mask=mask)


@triton.jit
def _m_dtype(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    """Data type semantics: the denominator is accumulated in float16."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    denom = tl.sum(e.to(tl.float16), axis=0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + offs, e / denom, mask=mask)


@triton.jit
def _m_indexing(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    """Indexing and stride: the row stride is off by one element."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    base = row * (n_cols - 1)
    x = tl.load(x_ptr + base + offs, mask=mask, other=-float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(y_ptr + base + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def _m_predicate(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    """Branch predication: the tail guard admits one lane too many."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs <= n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def _m_special(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    """Special value handling: masked lanes fill with 0.0 instead of -inf.

    A masked lane then contributes exp(0 - max) to the denominator rather than zero,
    so a partial tile is normalized by too large a sum.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)


_KERNELS = {
    "operator_implementation": _m_operator,
    "data_type_semantics": _m_dtype,
    "indexing_and_stride": _m_indexing,
    "branch_predication": _m_predicate,
    "special_value_handling": _m_special,
}


def triton_mutant(subcategory: str, n_cols: int) -> tuple[Mutant, TritonKernel]:
    """The mutant record and the adapter that runs it."""
    jit_fn = _KERNELS[subcategory]
    adapter = TritonKernel(
        kernel_id=f"softmax_{subcategory}_triton",
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


SUBCATEGORIES = tuple(_KERNELS)
