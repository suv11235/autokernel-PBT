"""Relation tests."""

from dataclasses import replace

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
    # ...and the rows are shifted by *different* amounts. Without this, a
    # scalar shift (size=(1, 1) or size=()) also passes.
    assert diff[0, 0] != diff[1, 0]


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


# (Seed determinism is covered by
# test_sequential_relations_share_an_rng_deterministically, which pins the same
# base and seed with stricter assertions.)


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


# --- anti-vacuity: the shift must be big enough to expose the defect --------


def _softmax_naive(x: np.ndarray) -> np.ndarray:
    """The defect under test: no max-subtraction, so exp() can overflow."""
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _softmax_correct(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sweep(kernel, groups: int = 50, seed: int = 3) -> int:
    """Count groups where the kernel's shifted output diverges from its base."""
    rng = np.random.default_rng(seed)
    diverged = 0
    for i in range(groups):
        base = replace(
            _base(),
            case_id=f"c{i}",
            tensors={"x": rng.normal(0.0, 1.0, size=(4, 8)).astype(np.float32)},
        )
        partner = ShiftRows().derive(base, rng)
        with np.errstate(over="ignore", invalid="ignore"):
            out_base = kernel(base.tensors["x"])
            out_partner = kernel(partner.tensors["x"])
        if not np.allclose(out_base, out_partner, atol=1e-5, equal_nan=False):
            diverged += 1
    return diverged


def test_shift_scale_can_actually_catch_an_unstable_softmax():
    # If this fails, the relation has gone vacuous: the shift is no longer
    # large enough to push exp() past the float32 overflow point (~88.72).
    assert _sweep(_softmax_naive) > 0


def test_shift_scale_does_not_false_alarm_on_a_correct_softmax():
    assert _sweep(_softmax_correct) == 0


def test_default_shift_scale_tracks_the_dtype_overflow_point():
    # Probe with many rows: a 2-row sample says little about the scale.
    wide = replace(_base(), tensors={"x": np.zeros((4000, 4), dtype=np.float32)})
    shifts = ShiftRows().derive(wide, np.random.default_rng(0)).tensors["x"][:, 0]

    overflow = float(np.log(np.finfo(np.float32).max))  # ~88.72
    # The shift must land in the band that actually reaches overflow. A
    # unit-scale shift sits at 0.011x of this and makes the property vacuous.
    assert 0.25 * overflow < shifts.std() < 0.75 * overflow


def test_shift_scale_is_overridable():
    derived = ShiftRows(scale=0.0).derive(_base(), np.random.default_rng(0))
    assert np.array_equal(derived.tensors["x"], _base().tensors["x"])
