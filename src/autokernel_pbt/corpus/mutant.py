"""One member of the mutation corpus.

The attribute is `intended_class`, never `fault_class`. Nothing in this project
verifies that an agent-authored kernel exhibits the class its prompt asked for --
automatic defect classification is a research problem of its own, and a weak
classifier would mislabel exactly the cases that matter. Naming the attribute for
what it actually is means a table built from it inherits the caveat by its own column
name, rather than depending on a footnote nobody reads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

KNOWN_BACKENDS = ("numpy", "triton")


@dataclass(frozen=True)
class Mutant:
    """A deliberately-broken kernel, traceable to the taxonomy row it samples."""

    kernel_id: str
    task_id: str
    #: The subcategory the authoring prompt asked for. INTENDED, not verified.
    intended_class: str
    #: The paper's own words for that subcategory, so a reader can check the mutant
    #: against the class it claims without re-deriving what the class meant.
    taxonomy_quote: str
    backend: str
    fn: Callable[..., Any] = field(compare=False)

    def __post_init__(self) -> None:
        if not self.taxonomy_quote.strip():
            msg = (
                f"mutant {self.kernel_id!r} carries no taxonomy_quote; a mutant with no "
                f"provenance is untraceable to the corpus it claims to sample"
            )
            raise ValueError(msg)
        if self.backend not in KNOWN_BACKENDS:
            msg = f"unknown backend {self.backend!r}; expected one of {KNOWN_BACKENDS}"
            raise ValueError(msg)
