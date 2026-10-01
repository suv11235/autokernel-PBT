"""Where each sampled subcategory sits in the ISSTA 2026 taxonomy (arXiv:2605.19652).

The parents are Table 2's categories, as transcribed in
`docs/superpowers/specs/2026-08-16-phase-2a-instrument-design.md` section 2.2, slugged
the way the first label already was (`type_and_operator`): lowercase, spaces to
underscores. That table files operator logic, dtype semantics and special-value
handling under Type and Operator, indexing and stride under Memory, and boundary-
predicate and mask errors under Control Flow and Scheduling.

One map for every mutant module. Each module used to prefix `type_and_operator/`
itself, which filed indexing_and_stride and branch_predication under the wrong parent.

Imports nothing from torch or triton, so it is readable on CPU.
"""

from __future__ import annotations

TYPE_AND_OPERATOR = "type_and_operator"
MEMORY = "memory"
CONTROL_FLOW_AND_SCHEDULING = "control_flow_and_scheduling"

#: subcategory -> its Table 2 parent category.
PARENT_OF = {
    "operator_implementation": TYPE_AND_OPERATOR,
    "data_type_semantics": TYPE_AND_OPERATOR,
    "special_value_handling": TYPE_AND_OPERATOR,
    "indexing_and_stride": MEMORY,
    "branch_predication": CONTROL_FLOW_AND_SCHEDULING,
}


def intended_class(subcategory: str) -> str:
    """The `Mutant.intended_class` label for a subcategory: `<parent>/<subcategory>`.

    Refuses an unmapped subcategory rather than defaulting its parent: a default is
    exactly how two of the five came to be filed under the wrong one.
    """
    if subcategory not in PARENT_OF:
        msg = (
            f"no Table 2 parent recorded for {subcategory!r}; add it to PARENT_OF "
            f"from the taxonomy before registering a mutant of that class"
        )
        raise ValueError(msg)
    return f"{PARENT_OF[subcategory]}/{subcategory}"
