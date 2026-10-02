"""The comparison an agent's spec is read with belongs to the harness."""

from __future__ import annotations

import numpy as np

from autokernel_pbt.formal.comparisons import Comparisons


def _softmax32(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


X = np.random.default_rng(3).normal(size=(8, 8)).astype(np.float32)
FLOAT32_ROWS_OF_8 = Comparisons(dtype=np.dtype(np.float32), n=8)


def test_close_broadcasts_a_scalar_and_nothing_else():
    """Row sums can be compared with 1.0; a (8, 1) column is not stretched over (8, 8)."""
    y = _softmax32(X)
    assert FLOAT32_ROWS_OF_8.close(np.sum(y, axis=-1), 1.0)
    assert not FLOAT32_ROWS_OF_8.close(y, y[:, :1])


def test_the_budget_is_the_outputs_dtype_not_the_specs_arithmetic():
    """Summing float32 outputs in float64 must not tighten the budget to float64's.

    The same float64 row sums are close to one under the float32 budget the harness binds,
    and far from it under a float64 budget — so the budget comes from the bound dtype, never
    from whatever the spec chose to compute in. And `for_output` binds the output's dtype
    and the input's last-axis length, which is what the reference arm normalizes by.
    """
    sums = np.sum(_softmax32(X).astype(np.float64), axis=-1)
    ones = np.ones_like(sums)  # not the scalar 1.0, so broadcasting is pinned elsewhere, alone
    assert FLOAT32_ROWS_OF_8.close(sums, ones)
    assert not Comparisons(dtype=np.dtype(np.float64), n=8).close(sums, ones)
    assert Comparisons.for_output({"x": X}, np.float32) == FLOAT32_ROWS_OF_8


def test_the_budget_grows_with_the_accumulation_length():
    """The budget is 30 eps times log2(n), floored at log2 = 1, as the reference arm's is.

    A residual of 60 float32 eps is a test ratio of 20 over rows of 8 and of 60 over rows of
    1, so it is close in the first and not in the second. Without the length, a correct
    kernel's pairwise-summation error over a wide row would be held to a one-element budget.
    """
    # float32 throughout, so this test pins the length and not the dtype rule. 1 + 60 ulp is
    # exact in float32.
    expected = np.ones(8, dtype=np.float32)
    actual = expected + np.float32(60) * np.finfo(np.float32).eps
    assert FLOAT32_ROWS_OF_8.close(actual, expected)
    assert not Comparisons(dtype=np.dtype(np.float32), n=1).close(actual, expected)
