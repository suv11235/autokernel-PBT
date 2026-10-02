"""The comparison an agent's spec is evaluated with: the harness's, never the agent's.

A spec over the reals says ``y = softmax(x)``. Its float reading cannot say that and mean it,
because rounding makes exact equality false for every correct kernel, so something has to decide
how close is close enough. If the agent decided, it could widen the margin until its spec accepted
anything — the formal twin of a tolerance tuned until the test passes. So the harness binds the
budget for each case, in the reference arm's own units, and the spec only ever asks whether two
things are close.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from autokernel_pbt.props.backends.base import PRIMARY_INPUT
from autokernel_pbt.props.tolerance import residual_ratio, within_threshold


@dataclass(frozen=True)
class Comparisons:
    """Closeness for one case, budgeted by the harness.

    ``dtype`` is the kernel output's dtype — the precision whose rounding is forgiven — and
    deliberately not the dtype of whatever the spec computes with: a spec that sums float32
    outputs in float64 would otherwise be held to float64's budget and reject every correct
    kernel. ``n`` is the accumulation length, read from the input as the reference arm reads it.
    """

    dtype: np.dtype
    n: int

    @classmethod
    def for_output(cls, inputs: Mapping[str, np.ndarray], dtype: np.dtype | type) -> Comparisons:
        x = inputs[PRIMARY_INPUT]
        n = x.shape[-1] if x.ndim and x.shape[-1] >= 1 else 1
        return cls(dtype=np.dtype(dtype), n=n)

    def close(self, actual: np.ndarray, expected: np.ndarray | float) -> bool:
        """Whether ``actual`` is within the rounding budget of ``expected``.

        A scalar ``expected`` is broadcast, so a spec can ask whether row sums are close to one.
        Nothing else is: broadcasting a (2, 1) array over (2, 3) is how ``np.allclose`` hides a
        wrong-shaped output, and a shape mismatch here is simply not close.
        """
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        if expected.ndim == 0:
            expected = np.broadcast_to(expected, actual.shape)
        return within_threshold(residual_ratio(actual, expected, dtype=self.dtype, n=self.n))

    def at_most(self, actual: np.ndarray, bound: np.ndarray | float) -> bool:
        """Whether every element of ``actual`` is at most ``bound``, up to the rounding budget.

        Only the excess over the bound is budgeted, through the same normalized ratio as
        closeness, so an order law forgives exactly the rounding an equality law does and no
        more. A scalar bound is broadcast; an array bound must match ``actual``'s shape.
        """
        actual = np.asarray(actual)
        bound = np.asarray(bound)
        if bound.ndim == 0:
            bound = np.broadcast_to(bound, actual.shape)
        elif bound.shape != actual.shape:
            return False
        return self.close(actual, np.minimum(actual, bound))

    def at_least(self, actual: np.ndarray, bound: np.ndarray | float) -> bool:
        """Whether every element of ``actual`` is at least ``bound``, up to the rounding budget."""
        return self.at_most(-np.asarray(actual), -np.asarray(bound))
