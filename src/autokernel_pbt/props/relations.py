"""Metamorphic relations: derive a partner case from a base case."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

import numpy as np

from autokernel_pbt.props.backends.base import HELPER_PREFIX
from autokernel_pbt.props.case import Case

# The backend filters tensors by this prefix before calling the kernel; the name
# is derived from it so the two cannot drift apart.
PERM_KEY = f"{HELPER_PREFIX}perm{HELPER_PREFIX}"


class Relation(Protocol):
    """Derives one metamorphic partner from a base case."""

    name: str

    def derive(self, base: Case, rng: np.random.Generator) -> Case: ...


def _derived(base: Case, relation: str, tensors: dict[str, np.ndarray]) -> Case:
    return replace(
        base,
        case_id=f"{base.case_id}::{relation}",
        relation=relation,
        tensors=tensors,
    )


def _overflow_scale(dtype: np.dtype) -> float:
    """Half the exponent at which ``exp`` overflows in ``dtype``.

    44.4 for float32, 5.5 for float16. Shifts drawn at this scale reach the
    regime where a softmax without max-subtraction actually overflows.
    """
    return 0.5 * float(np.log(np.finfo(dtype).max))


class ShiftRows:
    """x -> x + c, one constant per row. Consumed by shift-invariance properties.

    The shift scale is tied to the dtype's overflow point, not to unit variance.
    A softmax without max-subtraction is *mathematically* shift-invariant; it
    only breaks once ``exp`` overflows, which in float32 needs ``x > 88.72``.
    Draws from N(0, 1) never come close (max |shift| over 100k draws: ~4.5), so
    a unit-scale shift makes the property vacuous — it would pass on an unstable
    kernel every time. At the default scale, measured over 400 float32 trials:
    15.0% of groups catch a naive kernel, with 0% false alarms on a correct one.

    Requires a 2-D ``x``. A 1-D ``x`` is ambiguous (one row of n, or n rows of
    one?) and an ``(n, 1)`` shift silently outer-broadcasts it to ``(n, n)``,
    so anything other than 2-D is rejected rather than silently mis-shaped.

    Caveat for reduced precision: in float16 a wide shift produces *genuine*
    false alarms, because ``x + c`` destroys the row's mantissa detail — so
    shift-invariance holds only approximately there. Phase 1 is float32-only,
    where the dtype-aware scale is safe, but ``ShiftInvariance`` will need a
    dtype-aware tolerance before fp16 lands.
    """

    name = "shift_rows"

    def __init__(self, scale: float | None = None) -> None:
        # None means "derive from the tensor's dtype at derive() time".
        self.scale = scale

    def derive(self, base: Case, rng: np.random.Generator) -> Case:
        tensors = dict(base.tensors)
        x = tensors["x"]
        if x.ndim != 2:
            msg = (
                f"relation {self.name!r} requires a 2-D 'x', got shape "
                f"{tuple(x.shape)} in case {base.case_id!r}"
            )
            raise ValueError(msg)
        scale = _overflow_scale(x.dtype) if self.scale is None else self.scale
        shift = rng.normal(0.0, scale, size=(x.shape[0], 1)).astype(x.dtype)
        tensors["x"] = x + shift
        return _derived(base, self.name, tensors)


class PermuteLastAxis:
    """Permute the last axis. Consumed by equivariance properties.

    The permutation is stored under ``__perm__`` so the property can undo it.
    Note that ``case.shape`` and ``case.dtype`` describe ``x`` only; helper
    tensors such as ``__perm__`` carry their own shape and dtype.
    """

    name = "permute_last_axis"

    def derive(self, base: Case, rng: np.random.Generator) -> Case:
        tensors = dict(base.tensors)
        x = tensors["x"]
        perm = rng.permutation(x.shape[-1])
        tensors["x"] = np.take(x, perm, axis=-1)
        tensors[PERM_KEY] = perm.astype(np.int64)
        return _derived(base, self.name, tensors)


RELATIONS: dict[str, Callable[[], Relation]] = {
    ShiftRows.name: ShiftRows,
    PermuteLastAxis.name: PermuteLastAxis,
}
