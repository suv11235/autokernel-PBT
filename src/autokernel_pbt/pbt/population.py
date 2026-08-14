"""Kernel candidate representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class KernelCandidate:
    id: str
    source_path: str
    backend: str
    generation: int
    parent_id: str | None = None
    lineage_id: str = ""
    fitness: float = float("-inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "backend": self.backend,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "lineage_id": self.lineage_id or self.id,
            "fitness": self.fitness,
        }
