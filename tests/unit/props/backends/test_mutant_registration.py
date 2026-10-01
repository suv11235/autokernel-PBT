"""Mutant registration metadata, checked on CPU against stubbed torch and triton.

Only the scaffolding around the mutant bodies is under test. The bodies themselves are
recorded verbatim and never launched here.
"""

from __future__ import annotations

import importlib

import pytest

#: Table 2 of the ISSTA 2026 taxonomy, as transcribed in
#: docs/superpowers/specs/2026-08-16-phase-2a-instrument-design.md section 2.2: indexing
#: and stride is a Memory bug, predicate and mask errors are Control Flow and
#: Scheduling, and the other three are Type and Operator. Written out rather than read
#: from the implementation's map, so this checks the map against the paper.
EXPECTED_CLASS = {
    "operator_implementation": "type_and_operator/operator_implementation",
    "data_type_semantics": "type_and_operator/data_type_semantics",
    "special_value_handling": "type_and_operator/special_value_handling",
    "indexing_and_stride": "memory/indexing_and_stride",
    "branch_predication": "control_flow_and_scheduling/branch_predication",
}


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [("triton_mutants", "triton_mutant"), ("blinded_mutants", "blinded_mutant")],
)
def test_one_per_class_mutants_carry_their_table_2_parent(device_stubs, module_name, factory_name):
    module = importlib.import_module(f"kernels.mutants.{module_name}")
    factory = getattr(module, factory_name)
    assert set(module.SUBCATEGORIES) == set(EXPECTED_CLASS)
    for subcategory in module.SUBCATEGORIES:
        record, _ = factory(subcategory, 8)
        assert record.intended_class == EXPECTED_CLASS[subcategory], subcategory


def test_grown_mutants_carry_their_table_2_parent(device_stubs):
    module = importlib.import_module("kernels.mutants.grown_mutants")
    assert set(module.CLASS_OF.values()) == set(EXPECTED_CLASS)
    for name in module.SUBCATEGORIES:
        record, _ = module.grown_mutant(name, 8)
        assert record.intended_class == EXPECTED_CLASS[module.CLASS_OF[name]], name


def test_an_unmapped_subcategory_is_refused_rather_than_defaulted():
    # A new subcategory must get an explicit parent. Defaulting it is how two of five
    # landed under the wrong one.
    from kernels.mutants.taxonomy import intended_class

    with pytest.raises(ValueError, match="no Table 2 parent recorded for 'resource_allocation'"):
        intended_class("resource_allocation")
