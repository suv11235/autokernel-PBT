"""Generator determinism and coverage tests."""

import warnings

import numpy as np
import pytest

from autokernel_pbt.props.domain import InputDomain, TensorSpec
from autokernel_pbt.props.generator import Generator

DOMAIN = InputDomain(
    task_id="softmax",
    tensors=(TensorSpec(name="x", dtype="float32"),),
    shapes=((2, 4), (3, 5), (1, 7)),
    relations=("shift_rows", "permute_last_axis"),
)


def test_same_seed_gives_identical_tensors():
    a = Generator(DOMAIN, seed=123).generate(4)
    b = Generator(DOMAIN, seed=123).generate(4)
    for ga, gb in zip(a, b):
        assert np.array_equal(ga.base.tensors["x"], gb.base.tensors["x"])


# Mandated test: calls generate(1) on the 3-shape DOMAIN, so it legitimately trips
# the shape-coverage warning. Ignored rather than shown so the suite carries no
# standing warnings; the warning itself is pinned by the two dedicated tests below.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_different_seed_gives_different_tensors():
    a = Generator(DOMAIN, seed=1).generate(1)
    b = Generator(DOMAIN, seed=2).generate(1)
    assert not np.array_equal(a[0].base.tensors["x"], b[0].base.tensors["x"])


def test_every_shape_visited_before_repeat():
    groups = Generator(DOMAIN, seed=0).generate(3)
    assert {g.base.shape for g in groups} == set(DOMAIN.shapes)


# Mandated test: calls generate(1) on the 3-shape DOMAIN, so it legitimately trips
# the shape-coverage warning. Ignored rather than shown so the suite carries no
# standing warnings; the warning itself is pinned by the two dedicated tests below.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_group_contains_base_plus_each_relation():
    groups = Generator(DOMAIN, seed=0).generate(1)
    relations = {c.relation for c in groups[0].cases}
    assert relations == {"base", "shift_rows", "permute_last_axis"}


def test_group_ids_are_unique():
    groups = Generator(DOMAIN, seed=0).generate(6)
    assert len({g.group_id for g in groups}) == 6


# Mandated test: calls generate(1) on the 3-shape DOMAIN, so it legitimately trips
# the shape-coverage warning. Ignored rather than shown so the suite carries no
# standing warnings; the warning itself is pinned by the two dedicated tests below.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dtype_is_honoured():
    groups = Generator(DOMAIN, seed=0).generate(1)
    assert groups[0].base.tensors["x"].dtype == np.float32


def test_every_case_is_byte_identical_across_same_seeded_runs():
    """Determinism must cover relation-derived partners, not just the base.

    Task 12 asserts two oracle arms see byte-identical inputs; those arms read
    every case in the group, including helper tensors such as ``__perm__``.
    """
    a = Generator(DOMAIN, seed=7).generate(5)
    b = Generator(DOMAIN, seed=7).generate(5)
    assert [g.group_id for g in a] == [g.group_id for g in b]
    for ga, gb in zip(a, b):
        assert [c.case_id for c in ga.cases] == [c.case_id for c in gb.cases]
        for ca, cb in zip(ga.cases, gb.cases):
            assert ca.metadata() == cb.metadata()
            assert ca.tensors.keys() == cb.tensors.keys()
            for name in ca.tensors:
                left, right = ca.tensors[name], cb.tensors[name]
                assert left.dtype == right.dtype
                assert left.shape == right.shape
                assert left.tobytes() == right.tobytes()


def test_prefix_is_stable_across_generation_count():
    """generate(10) reproduces the first 4 groups of generate(4) exactly.

    Each group draws from its own ``default_rng([seed, index])`` stream, so group
    i's bytes are a pure function of (seed, i) -- nothing about the requested count
    can reach them. Recorded runs stay replayable when a case budget is widened.
    """
    short = Generator(DOMAIN, seed=99).generate(4)
    long = Generator(DOMAIN, seed=99).generate(10)
    for ga, gb in zip(short, long[:4]):
        assert ga.group_id == gb.group_id
        for ca, cb in zip(ga.cases, gb.cases):
            assert ca.case_id == cb.case_id
            for name in ca.tensors:
                assert ca.tensors[name].tobytes() == cb.tensors[name].tobytes()


def _base_bytes(groups):
    return [g.base.tensors["x"].tobytes() for g in groups]


def test_adding_a_tensor_does_not_perturb_existing_tensors():
    """A corpus must be extensible: adding 'y' must not rewrite 'x'.

    Under a single shared rng stream this failed for every group after the first,
    because 'y' consumed draws that shifted 'x' in all later groups.
    """
    one = InputDomain(
        task_id="softmax",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=((4, 4),),
        relations=("shift_rows",),
    )
    two = InputDomain(
        task_id="softmax",
        tensors=(
            TensorSpec(name="x", dtype="float32"),
            TensorSpec(name="y", dtype="float32"),
        ),
        shapes=((4, 4),),
        relations=("shift_rows",),
    )
    assert _base_bytes(Generator(one, seed=42).generate(4)) == _base_bytes(
        Generator(two, seed=42).generate(4)
    )


def test_reordering_relations_does_not_perturb_base_tensors():
    """Relation order is a domain detail; it must not rewrite the base inputs.

    The samplers consume variable numbers of 64-bit words, so under one shared
    stream reordering desynchronized and resynchronized it unpredictably -- there
    was not even a stable "first k groups match" prefix.
    """
    forward = InputDomain(
        task_id="softmax",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=((4, 4),),
        relations=("shift_rows", "permute_last_axis"),
    )
    reversed_ = InputDomain(
        task_id="softmax",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=((4, 4),),
        relations=("permute_last_axis", "shift_rows"),
    )
    assert _base_bytes(Generator(forward, seed=42).generate(8)) == _base_bytes(
        Generator(reversed_, seed=42).generate(8)
    )


def test_zero_groups_is_empty_and_negative_is_rejected():
    assert Generator(DOMAIN, seed=0).generate(0) == []
    with pytest.raises(ValueError, match="n_groups must be non-negative"):
        Generator(DOMAIN, seed=0).generate(-3)


def test_too_few_groups_warns_about_unexercised_shapes():
    """Boundary shape coverage is the recall mechanism; a never-visited shape is silent loss."""
    with pytest.warns(UserWarning, match=r"\(1, 7\)"):
        Generator(DOMAIN, seed=0).generate(2)


def test_full_shape_coverage_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Generator(DOMAIN, seed=0).generate(3)


def test_zero_groups_does_not_warn_about_coverage():
    """Asking for nothing cannot lose shape coverage -- there is nothing to cover."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Generator(DOMAIN, seed=0).generate(0) == []


def test_coverage_warning_points_at_the_caller():
    """The warning must name the misconfiguring call site, not generator.py."""
    with pytest.warns(UserWarning) as record:
        Generator(DOMAIN, seed=0).generate(2)
    assert record[0].filename == __file__


def test_unknown_relation_raises_value_error():
    domain = InputDomain(
        task_id="softmax",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=((2, 4),),
        relations=("shift_rowz",),
    )
    with pytest.raises(ValueError, match="shift_rowz") as excinfo:
        Generator(domain, seed=0).generate(1)
    message = str(excinfo.value)
    assert "softmax" in message
    assert "shift_rows" in message
    assert "permute_last_axis" in message


def test_relation_shape_mismatch_raises_clear_value_error():
    """A 1-D shape under ``shift_rows`` must name the relation, not crash obscurely."""
    domain = InputDomain(
        task_id="softmax",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=((8,),),
        relations=("shift_rows",),
    )
    with pytest.raises(ValueError, match="shift_rows") as excinfo:
        Generator(domain, seed=0).generate(1)
    assert "2-D" in str(excinfo.value)
