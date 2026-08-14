"""Mutation hook (skeleton)."""

from __future__ import annotations

import uuid

from autokernel_pbt.pbt.population import KernelCandidate


def mutate(parent: KernelCandidate, generation: int) -> KernelCandidate:
    """Explore: new candidate id and lineage, same source path (placeholder)."""
    new_id = str(uuid.uuid4())[:8]
    return KernelCandidate(
        id=new_id,
        source_path=parent.source_path,
        backend=parent.backend,
        generation=generation,
        parent_id=parent.id,
        lineage_id=new_id,
    )
