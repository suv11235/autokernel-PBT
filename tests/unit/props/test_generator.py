"""Generator determinism and coverage tests."""

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


def test_different_seed_gives_different_tensors():
    a = Generator(DOMAIN, seed=1).generate(1)
    b = Generator(DOMAIN, seed=2).generate(1)
    assert not np.array_equal(a[0].base.tensors["x"], b[0].base.tensors["x"])


def test_every_shape_visited_before_repeat():
    groups = Generator(DOMAIN, seed=0).generate(3)
    assert {g.base.shape for g in groups} == set(DOMAIN.shapes)


def test_group_contains_base_plus_each_relation():
    groups = Generator(DOMAIN, seed=0).generate(1)
    relations = {c.relation for c in groups[0].cases}
    assert relations == {"base", "shift_rows", "permute_last_axis"}


def test_group_ids_are_unique():
    groups = Generator(DOMAIN, seed=0).generate(6)
    assert len({g.group_id for g in groups}) == 6


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

    The rng is drawn from a fresh ``default_rng(seed)`` per ``generate`` call and
    consumed in a fixed order per group, so group i depends only on the seed and
    on i -- never on how many groups were requested. Recorded runs stay
    replayable when a case budget is widened.
    """
    short = Generator(DOMAIN, seed=99).generate(4)
    long = Generator(DOMAIN, seed=99).generate(10)
    for ga, gb in zip(short, long[:4]):
        assert ga.group_id == gb.group_id
        for ca, cb in zip(ga.cases, gb.cases):
            assert ca.case_id == cb.case_id
            for name in ca.tensors:
                assert ca.tensors[name].tobytes() == cb.tensors[name].tobytes()


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
