"""Relation tests."""

import numpy as np
import pytest

from autokernel_pbt.props.case import BASE_RELATION, Case
from autokernel_pbt.props.relations import RELATIONS, PermuteLastAxis, ShiftRows


def _base() -> Case:
    return Case(
        case_id="c0",
        group_id="g0",
        relation=BASE_RELATION,
        task_id="softmax",
        dtype="float32",
        shape=(2, 4),
        tensors={"x": np.arange(8, dtype=np.float32).reshape(2, 4)},
    )


def test_shift_rows_adds_per_row_constant():
    rng = np.random.default_rng(0)
    derived = ShiftRows().derive(_base(), rng)
    diff = derived.tensors["x"] - _base().tensors["x"]
    # Every element in a row shifted by the same amount.
    assert np.allclose(diff, diff[:, :1])


def test_shift_rows_sets_relation_and_group():
    derived = ShiftRows().derive(_base(), np.random.default_rng(0))
    assert derived.relation == "shift_rows"
    assert derived.group_id == "g0"
    assert derived.case_id == "c0::shift_rows"


def test_permute_last_axis_is_a_permutation():
    derived = PermuteLastAxis().derive(_base(), np.random.default_rng(0))
    assert np.array_equal(
        np.sort(derived.tensors["x"], axis=-1), np.sort(_base().tensors["x"], axis=-1)
    )


def test_permute_records_its_index_map():
    derived = PermuteLastAxis().derive(_base(), np.random.default_rng(0))
    perm = derived.tensors["__perm__"]
    assert sorted(perm.tolist()) == [0, 1, 2, 3]


def test_relations_registry_is_keyed_by_name():
    for name, factory in RELATIONS.items():
        assert factory().name == name


def test_relations_are_deterministic_for_a_seed():
    a = ShiftRows().derive(_base(), np.random.default_rng(7))
    b = ShiftRows().derive(_base(), np.random.default_rng(7))
    assert np.array_equal(a.tensors["x"], b.tensors["x"])


# --- A. non-mutation of the base case -------------------------------------


def test_relations_do_not_mutate_the_base_case():
    base = _base()
    before = base.tensors["x"].copy()
    shifted = ShiftRows().derive(base, np.random.default_rng(1))
    permuted = PermuteLastAxis().derive(base, np.random.default_rng(1))

    assert np.array_equal(base.tensors["x"], before)
    assert base.tensors["x"].tobytes() == before.tobytes()
    assert shifted.tensors["x"] is not base.tensors["x"]
    assert permuted.tensors["x"] is not base.tensors["x"]
    # The tensor dict itself is rebound, not shared.
    assert shifted.tensors is not base.tensors
    assert permuted.tensors is not base.tensors
    assert "__perm__" not in base.tensors


# --- B. 1-D inputs --------------------------------------------------------


def _base_1d() -> Case:
    return Case(
        case_id="c1",
        group_id="g1",
        relation=BASE_RELATION,
        task_id="softmax",
        dtype="float32",
        shape=(8,),
        tensors={"x": np.arange(8, dtype=np.float32)},
    )


def test_shift_rows_rejects_1d_input():
    with pytest.raises(ValueError, match="shift_rows"):
        ShiftRows().derive(_base_1d(), np.random.default_rng(0))


def test_permute_last_axis_supports_1d_input():
    base = _base_1d()
    derived = PermuteLastAxis().derive(base, np.random.default_rng(0))
    assert np.array_equal(np.sort(derived.tensors["x"]), np.sort(base.tensors["x"]))
    assert sorted(derived.tensors["__perm__"].tolist()) == list(range(8))


# --- C. helper-tensor naming convention -----------------------------------


def test_perm_uses_the_helper_prefix():
    derived = PermuteLastAxis().derive(_base(), np.random.default_rng(0))
    helpers = [k for k in derived.tensors if k.startswith("__")]
    assert helpers == ["__perm__"]


# --- D. determinism across a shared rng -----------------------------------


def _apply_both(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = _base()
    shifted = ShiftRows().derive(base, rng)
    permuted = PermuteLastAxis().derive(base, rng)
    return shifted.tensors["x"], permuted.tensors["x"], permuted.tensors["__perm__"]


def test_sequential_relations_share_an_rng_deterministically():
    a_shift, a_perm_x, a_perm = _apply_both(11)
    b_shift, b_perm_x, b_perm = _apply_both(11)
    assert np.array_equal(a_shift, b_shift)
    assert np.array_equal(a_perm_x, b_perm_x)
    assert np.array_equal(a_perm, b_perm)
