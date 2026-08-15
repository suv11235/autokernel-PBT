"""Serializable description of a task's input space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Phase 1 is NumPy-only; bfloat16 has no native NumPy dtype and arrives with the
# device backends in Phase 3.
SUPPORTED_DTYPES = ("float16", "float32", "float64")
SUPPORTED_DISTRIBUTIONS = ("normal", "uniform", "zeros", "ones")


@dataclass(frozen=True)
class TensorSpec:
    """One input tensor's value distribution. Shape comes from the domain.

    Args:
        name: Tensor name.
        dtype: NumPy dtype string (float16, float32, float64).
        distribution: Value distribution ("normal", "uniform", "zeros", "ones").
        low: Lower bound for uniform distribution (ignored for others).
        high: Upper bound for uniform distribution (ignored for others).
    """

    name: str
    dtype: str
    distribution: str = "normal"
    low: float = -1.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_DTYPES:
            msg = (
                f"unsupported dtype {self.dtype!r}; "
                f"expected one of {SUPPORTED_DTYPES}"
            )
            raise ValueError(msg)
        if self.distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"unsupported distribution {self.distribution!r}")
        if self.high < self.low:
            raise ValueError(
                f"tensor {self.name!r}: low {self.low} exceeds high {self.high}"
            )

    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "distribution": self.distribution,
            "low": self.low,
            "high": self.high,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorSpec:
        return cls(**data)


@dataclass(frozen=True)
class InputDomain:
    """Everything needed to regenerate a task's case set from a seed."""

    task_id: str
    tensors: tuple[TensorSpec, ...]
    shapes: tuple[tuple[int, ...], ...]
    relations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Normalize sequences to canonical tuple form so that hand construction and
        # from_dict produce equal objects, and so numpy int64 dimensions become Python ints.
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(
            self,
            "shapes",
            tuple(tuple(int(d) for d in s) for s in self.shapes),
        )
        object.__setattr__(self, "relations", tuple(self.relations))

        if not self.shapes:
            raise ValueError("domain needs at least one shape")
        if not self.tensors:
            raise ValueError("domain needs at least one tensor spec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tensors": [t.to_dict() for t in self.tensors],
            "shapes": [list(s) for s in self.shapes],
            "relations": list(self.relations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputDomain:
        return cls(
            task_id=data["task_id"],
            tensors=tuple(TensorSpec.from_dict(t) for t in data["tensors"]),
            shapes=tuple(tuple(s) for s in data["shapes"]),
            relations=tuple(data.get("relations", ())),
        )
