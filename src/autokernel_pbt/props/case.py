"""Generated inputs and their group identity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

BASE_RELATION = "base"


@dataclass(frozen=True)
class Case:
    """One executable input set."""

    case_id: str
    group_id: str
    relation: str
    task_id: str
    dtype: str
    shape: tuple[int, ...]
    tensors: dict[str, np.ndarray] = field(compare=False)

    def __post_init__(self) -> None:
        # Normalize shape to tuple of int
        if not isinstance(self.shape, tuple):
            object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))

    def metadata(self) -> dict[str, Any]:
        """Everything except tensor payloads — this is what lands in Parquet."""
        return {
            "case_id": self.case_id,
            "group_id": self.group_id,
            "relation": self.relation,
            "task_id": self.task_id,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class CaseGroup:
    """A base case plus its metamorphic partners, sharing one group_id."""

    group_id: str
    cases: tuple[Case, ...]

    def __post_init__(self) -> None:
        # Normalize cases to tuple
        if not isinstance(self.cases, tuple):
            object.__setattr__(self, "cases", tuple(self.cases))

        bases = [c for c in self.cases if c.relation == BASE_RELATION]
        if len(bases) != 1:
            msg = (
                f"group {self.group_id} needs exactly one base case, "
                f"got {len(bases)}"
            )
            raise ValueError(msg)
        for case in self.cases:
            if case.group_id != self.group_id:
                msg = (
                    f"group_id mismatch: case {case.case_id} has "
                    f"{case.group_id!r}, group is {self.group_id!r}"
                )
                raise ValueError(msg)

    @property
    def base(self) -> Case:
        return next(c for c in self.cases if c.relation == BASE_RELATION)

    def by_relation(self, relation: str) -> Case | None:
        return next((c for c in self.cases if c.relation == relation), None)
