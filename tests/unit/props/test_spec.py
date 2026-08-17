"""CaseSpec round-trip and regeneration tests.

The spec exists so a recorded group can be rebuilt without re-executing anything on
hardware, which is what makes offline shrinking possible later. So the load-bearing
assertion here is not that the spec serializes — it is that a group rebuilt from one
is *byte-identical* to the group that was recorded. A spec that regenerated something
merely similar would let a future shrinker report a minimal case the run never ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.domain import InputDomain, TensorSpec
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.spec import CaseSpec

DOMAIN = InputDomain(
    task_id="softmax",
    tensors=(TensorSpec(name="x", dtype="float32"),),
    shapes=((2, 4), (3, 5)),
    relations=("shift_rows",),
)


def test_spec_round_trips_through_json():
    spec = CaseSpec(
        seed=7, task_id="softmax", group_index=3, shape=(2, 4), transforms=("shift_rows",)
    )
    assert CaseSpec.from_json(spec.to_json()) == spec


def test_spec_json_is_stable_across_encodings():
    # A corpus fingerprint over the persisted spec would differ for no reason if key
    # order varied between two encodings of the same spec.
    spec = CaseSpec(seed=7, task_id="t", group_index=0, shape=(2,), transforms=("a", "b"))
    assert spec.to_json() == CaseSpec.from_json(spec.to_json()).to_json()


def test_spec_normalizes_shape_to_ints():
    # np.int64 dims otherwise survive construction and fail later at json.dumps, in a
    # persistence path far from the mistake.
    spec = CaseSpec(
        seed=1, task_id="t", group_index=0, shape=(np.int64(2), np.int64(4)), transforms=()
    )
    assert spec.shape == (2, 4)
    assert all(type(d) is int for d in spec.shape)


def test_spec_rejects_a_negative_group_index():
    with pytest.raises(ValueError, match="group_index must be non-negative"):
        CaseSpec(seed=1, task_id="t", group_index=-1, shape=(2,), transforms=())


def test_spec_rejects_duplicate_transforms():
    # CaseGroup rejects duplicate relations because by_relation() would return the
    # first of several and make the rest unreachable. Caught here, the error names
    # the spec; caught there, it names only a group id.
    with pytest.raises(ValueError, match="duplicate transforms"):
        CaseSpec(
            seed=1,
            task_id="t",
            group_index=0,
            shape=(2,),
            transforms=("shift_rows", "shift_rows"),
        )


def test_generator_stamps_a_spec_on_every_group():
    groups = Generator(DOMAIN, seed=11).generate(2)
    assert groups[0].spec == CaseSpec(
        seed=11, task_id="softmax", group_index=0, shape=(2, 4), transforms=("shift_rows",)
    )
    assert groups[1].spec is not None
    assert groups[1].spec.group_index == 1


def test_group_from_spec_is_byte_identical():
    """The criterion CASE_SPEC_REGENERATES: a spec is a complete recipe.

    Byte-identical, not merely close: a regenerated case that differs in the last
    ulp would let a shrinker report a minimal reproducer that the recorded run never
    executed, and no downstream check would notice.
    """
    generator = Generator(DOMAIN, seed=11)
    original = generator.generate(3)[2]
    assert original.spec is not None
    rebuilt = generator.group_from_spec(original.spec)

    assert rebuilt.group_id == original.group_id
    assert len(rebuilt.cases) == len(original.cases)
    for a, b in zip(original.cases, rebuilt.cases, strict=True):
        assert a.case_id == b.case_id
        assert a.relation == b.relation
        assert a.shape == b.shape
        assert sorted(a.tensors) == sorted(b.tensors)
        for name, array in a.tensors.items():
            assert array.dtype == b.tensors[name].dtype, name
            # array_equal, not allclose: the claim is bitwise identity.
            assert np.array_equal(array, b.tensors[name]), name


def test_group_from_spec_honours_a_reduced_transform_list():
    """Dropping a transform is the unit move of a shrinker; the base is unchanged."""
    generator = Generator(DOMAIN, seed=11)
    # generate(len(DOMAIN.shapes)), not generate(1): asking for fewer groups than
    # shapes emits the unexercised-shapes warning, and filterwarnings=["error"]
    # makes that a failure. The reduction under test is per-group, so which group
    # is irrelevant.
    full = generator.generate(len(DOMAIN.shapes))[0]
    assert full.spec is not None
    reduced = generator.group_from_spec(full.spec.without_transform("shift_rows"))

    assert {c.relation for c in reduced.cases} == {"base"}
    assert np.array_equal(reduced.base.tensors["x"], full.base.tensors["x"])


def test_without_transform_leaves_the_original_spec_alone():
    # CaseSpec is frozen, but a shrinker exploring several reductions must be able to
    # branch from one spec repeatedly without the first branch mutating the second.
    spec = CaseSpec(
        seed=1, task_id="t", group_index=0, shape=(2,), transforms=("a", "b")
    )
    assert spec.without_transform("a").transforms == ("b",)
    assert spec.transforms == ("a", "b")


def test_without_transform_rejects_an_absent_name():
    spec = CaseSpec(
        seed=1, task_id="t", group_index=0, shape=(2,), transforms=("shift_rows",)
    )
    with pytest.raises(ValueError, match="does not carry transform 'permute_last_axis'"):
        spec.without_transform("permute_last_axis")


def test_group_from_spec_rejects_a_spec_for_another_task():
    # The rebuilt group would stamp ids from the spec's task while the tensors came
    # from this generator's domain, producing rows that cannot join back to either.
    generator = Generator(DOMAIN, seed=11)
    other = CaseSpec(seed=11, task_id="relu", group_index=0, shape=(2, 4), transforms=())
    with pytest.raises(ValueError, match="spec is for task 'relu'"):
        generator.group_from_spec(other)


def test_a_hand_built_group_may_carry_no_spec():
    # A group assembled from literal tensors has nothing to regenerate it from, and
    # requiring a recipe would force tests to invent one that regenerates something
    # else entirely.
    from autokernel_pbt.props.case import Case, CaseGroup

    case = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="t",
        dtype="float32",
        shape=(2,),
        tensors={"x": np.zeros(2, dtype=np.float32)},
    )
    assert CaseGroup(group_id="g0", cases=(case,)).spec is None
