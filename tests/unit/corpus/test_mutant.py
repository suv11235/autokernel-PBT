"""Mutant identity and labelling tests."""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.corpus.mutant import Mutant


def _fn(x):
    return x * 2.0


def _mutant(**overrides) -> Mutant:
    defaults = {
        "kernel_id": "softmax_dtype_semantics",
        "task_id": "softmax",
        "intended_class": "type_and_operator/data_type_semantics",
        "taxonomy_quote": "Loss of numeric meaning or precision from implicit casts",
        "backend": "numpy",
        "fn": _fn,
    }
    defaults.update(overrides)
    return Mutant(**defaults)


def test_fault_class_is_recorded_as_intended():
    """The criterion INTENDED_CLASS_IS_LABELLED_AS_INTENDED.

    Nothing verifies that a returned kernel exhibits the class the prompt asked for.
    The attribute is named `intended_class`, not `fault_class`, so a reader of the
    code cannot mistake a construction for a verification -- and any table built from
    it inherits the caveat by its own column name.
    """
    m = _mutant()
    assert m.intended_class == "type_and_operator/data_type_semantics"
    assert not hasattr(m, "fault_class")
    assert not hasattr(m, "verified_class")


def test_the_taxonomy_quote_is_carried_verbatim():
    # The paper's own words, so a reader can check the mutant against the class it
    # claims without re-deriving what the class meant.
    assert "implicit casts" in _mutant().taxonomy_quote


def test_a_mutant_without_a_taxonomy_quote_is_rejected():
    # A mutant with no provenance is untraceable to the corpus it claims to sample.
    with pytest.raises(ValueError, match="taxonomy_quote"):
        _mutant(taxonomy_quote="")


def test_backend_must_be_known():
    with pytest.raises(ValueError, match="unknown backend"):
        _mutant(backend="cuda_cpp")


def test_the_callable_is_reachable():
    assert _mutant().fn(np.ones(2)).tolist() == [2.0, 2.0]
