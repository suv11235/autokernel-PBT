"""The reducible description of one case group.

A ``CaseSpec`` is everything needed to rebuild a group *given the run's domain*: the
seed, which group it is, the shape it was assigned, and the ordered transforms
applied to its base case. The domain is deliberately not duplicated here — a spec
carrying its own copy of the dtypes and distributions could disagree with the domain
the run was actually generated under, and then a "regenerated" case would be a
different case wearing the same id.

WHY THIS EXISTS NOW, with no shrinker to use it.
``reference/PBT-property-based-testing/NOTES.md`` §5.3 records the spirv-fuzz result:
if metamorphic transformations are small and independent, plain delta debugging over
the *transformation sequence* gives reduction for free — and that is an architectural
decision which must be made up front rather than retrofitted. Once a corpus is
recorded against cases that cannot be described, shrinking means re-executing on
hardware to explore, which is precisely the cost the record/replay architecture
exists to avoid.

Shrinking a *tensor* is deliberately not supported. The reduction is over
``transforms``, which is a short list of names, and ``without_transform`` is its unit
move.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaseSpec:
    """A regeneration recipe for one case group, relative to the run's domain."""

    seed: int
    task_id: str
    group_index: int
    shape: tuple[int, ...]
    transforms: tuple[str, ...]

    def __post_init__(self) -> None:
        # Normalize unconditionally, as InputDomain and Case do. A guard on
        # `isinstance(..., tuple)` would let an already-tuple shape skip int()
        # coercion, so np.int64 dims would survive construction and only fail later
        # at json.dumps, in a persistence path far from the mistake.
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))
        object.__setattr__(self, "transforms", tuple(self.transforms))
        if self.group_index < 0:
            msg = f"group_index must be non-negative, got {self.group_index}"
            raise ValueError(msg)
        # CaseGroup rejects duplicate relations, because by_relation() would otherwise
        # return the first of several and make the rest unreachable. Catching it here
        # names the spec that is wrong; catching it there names only a group id.
        if len(set(self.transforms)) != len(self.transforms):
            msg = f"spec has duplicate transforms: {list(self.transforms)}"
            raise ValueError(msg)

    def without_transform(self, name: str) -> CaseSpec:
        """This spec with one transform dropped — the unit move of a future shrinker."""
        if name not in self.transforms:
            msg = (
                f"spec for group {self.group_index} does not carry transform {name!r}; "
                f"it has {list(self.transforms)}"
            )
            raise ValueError(msg)
        return CaseSpec(
            seed=self.seed,
            task_id=self.task_id,
            group_index=self.group_index,
            shape=self.shape,
            transforms=tuple(t for t in self.transforms if t != name),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "task_id": self.task_id,
            "group_index": self.group_index,
            "shape": list(self.shape),
            "transforms": list(self.transforms),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseSpec:
        return cls(
            seed=data["seed"],
            task_id=data["task_id"],
            group_index=data["group_index"],
            shape=tuple(data["shape"]),
            transforms=tuple(data["transforms"]),
        )

    def to_json(self) -> str:
        # sort_keys so the encoding is stable: a spec persisted twice must produce
        # identical bytes, or a corpus fingerprint over it would differ for no reason.
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> CaseSpec:
        return cls.from_dict(json.loads(text))
