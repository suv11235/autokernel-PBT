"""Metamorphic relations: derive a partner case from a base case."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Protocol

import numpy as np

from autokernel_pbt.props.case import Case


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


class ShiftRows:
    """x -> x + c, one constant per row. Consumed by shift-invariance properties.

    Requires a 2-D ``x``. A 1-D ``x`` is ambiguous (one row of n, or n rows of
    one?) and an ``(n, 1)`` shift silently outer-broadcasts it to ``(n, n)``,
    so anything other than 2-D is rejected rather than silently mis-shaped.
    """

    name = "shift_rows"

    def derive(self, base: Case, rng: np.random.Generator) -> Case:
        tensors = dict(base.tensors)
        x = tensors["x"]
        if x.ndim != 2:
            msg = (
                f"relation {self.name!r} requires a 2-D 'x', got shape "
                f"{tuple(x.shape)} in case {base.case_id!r}"
            )
            raise ValueError(msg)
        shift = rng.normal(0.0, 1.0, size=(x.shape[0], 1)).astype(x.dtype)
        tensors["x"] = (x + shift).astype(x.dtype)
        return _derived(base, self.name, tensors)


class PermuteLastAxis:
    """Permute the last axis. Consumed by equivariance properties.

    The permutation is stored under ``__perm__`` so the property can undo it.
    """

    name = "permute_last_axis"

    def derive(self, base: Case, rng: np.random.Generator) -> Case:
        tensors = dict(base.tensors)
        x = tensors["x"]
        perm = rng.permutation(x.shape[-1])
        tensors["x"] = np.take(x, perm, axis=-1)
        tensors["__perm__"] = perm.astype(np.int64)
        return _derived(base, self.name, tensors)


RELATIONS: dict[str, Callable[[], Relation]] = {
    ShiftRows.name: ShiftRows,
    PermuteLastAxis.name: PermuteLastAxis,
}
