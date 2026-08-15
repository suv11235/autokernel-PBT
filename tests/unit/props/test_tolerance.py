"""Test-ratio tolerance tests."""

import numpy as np
import pytest

from autokernel_pbt.props.tolerance import (
    DEFAULT_THRESH,
    machine_eps,
    residual_ratio,
    within_threshold,
)


def test_identical_arrays_give_zero_ratio():
    x = np.ones((4, 4), dtype=np.float32)
    assert residual_ratio(x, x) == 0.0


def test_ratio_is_dimensionless_across_scale():
    """Scaling both arrays must not change the ratio.

    Two factors, for two different reasons. 1024.0 is exactly representable, so
    scaling introduces no rounding of its own and the ratio must come back
    *bitwise* identical — that is the property, asserted with no slack. 1000.0 is
    not, so ``cand * 1000`` and ``ref * 1000`` each round before they are
    subtracted; the perturbation is kept well above the ulp of the scaled values
    so that self-inflicted noise stays under the tolerance. (At ``1e-12``, the
    residual is only a few thousand ulps of 2000.0 and the scaled ratio differs
    in the 4th digit — an artifact of the construction, not of the ratio.)
    """
    ref = np.linspace(1.0, 2.0, 16, dtype=np.float64).reshape(4, 4)
    cand = ref + 1e-6
    small = residual_ratio(cand, ref)
    assert residual_ratio(cand * 1024.0, ref * 1024.0) == small
    large = residual_ratio(cand * 1000.0, ref * 1000.0)
    assert np.isclose(small, large, rtol=1e-6)
    # Down as well as up, across six orders of magnitude.
    assert np.isclose(residual_ratio(cand * 1e-6, ref * 1e-6), small, rtol=1e-6)


def test_float32_rounding_stays_under_threshold():
    ref = np.random.default_rng(0).normal(size=(64, 64))
    cand = ref.astype(np.float32).astype(np.float64)
    assert residual_ratio(cand, ref, dtype=np.float32) < DEFAULT_THRESH


def test_gross_error_exceeds_threshold():
    ref = np.ones((8, 8), dtype=np.float32)
    cand = ref.copy()
    cand[0, 0] = 5.0
    assert residual_ratio(cand, ref) > DEFAULT_THRESH


def test_nan_in_candidate_gives_infinite_ratio():
    ref = np.ones((4,), dtype=np.float32)
    cand = np.array([np.nan, 1.0, 1.0, 1.0], dtype=np.float32)
    assert np.isinf(residual_ratio(cand, ref))


def test_zero_reference_does_not_divide_by_zero():
    ref = np.zeros((4,), dtype=np.float32)
    cand = np.zeros((4,), dtype=np.float32)
    assert np.isfinite(residual_ratio(cand, ref))


def test_machine_eps_matches_numpy():
    assert machine_eps(np.float32) == np.finfo(np.float32).eps


# --- dtype handling ---------------------------------------------------------


def test_ratio_is_dtype_sensitive():
    """The unit roundoff must actually divide the residual.

    A refactor that ignored ``dtype`` would make these two calls equal; they must
    instead differ by exactly the ratio of the two dtypes' eps (2**29).
    """
    ref = np.linspace(1.0, 2.0, 16, dtype=np.float64)
    cand = ref + 1e-9
    r32 = residual_ratio(cand, ref, dtype=np.float32)
    r64 = residual_ratio(cand, ref, dtype=np.float64)
    expected = np.finfo(np.float32).eps / np.finfo(np.float64).eps
    assert np.isclose(r64 / r32, expected, rtol=1e-9)


def test_integer_candidate_is_rejected_with_a_clear_message():
    ref = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="no unit roundoff"):
        residual_ratio(ref, ref)


def test_python_list_candidate_is_rejected():
    # np.asarray([1, 2, 3]).dtype is int64: the same integer path, reached by
    # accident rather than on purpose.
    with pytest.raises(ValueError, match="no unit roundoff"):
        residual_ratio([1, 2, 3], [1, 2, 3])


def test_integer_candidate_is_accepted_with_an_explicit_dtype():
    cand = np.array([1, 2, 3])
    assert residual_ratio(cand, cand, dtype=np.float32) == 0.0


def test_bool_candidate_is_rejected():
    x = np.array([True, False])
    with pytest.raises(ValueError, match="no unit roundoff"):
        residual_ratio(x, x)


# --- shape and degenerate inputs -------------------------------------------


def test_shape_mismatch_is_infinite():
    assert np.isinf(residual_ratio(np.ones((4,)), np.ones((5,))))


def test_scalar_reference_is_normalized_like_the_persisted_output():
    # A reduction's persisted output is (1,) while np.sum gives (). Normalizing
    # only one side would return inf for every reduction case.
    assert residual_ratio(np.array([3.0]), np.float64(3.0)) == 0.0


def test_empty_arrays_agree_vacuously():
    empty = np.zeros((0,), dtype=np.float32)
    assert residual_ratio(empty, empty) == 0.0


def test_huge_values_do_not_warn_on_overflow():
    # cand - ref overflows float64 here; filterwarnings=["error"] turns the
    # RuntimeWarning into a failure unless the arithmetic is guarded.
    big = np.finfo(np.float64).max
    cand = np.array([big, 0.0])
    ref = np.array([-big, 0.0])
    assert np.isinf(residual_ratio(cand, ref))


# --- threshold --------------------------------------------------------------


def test_within_threshold_rejects_non_finite_ratios():
    assert within_threshold(0.0)
    assert not within_threshold(float("inf"))
    assert not within_threshold(float("nan"))
    assert not within_threshold(DEFAULT_THRESH)


def test_single_ulp_per_element_stays_under_threshold():
    """One ulp of float32 error on every element must not trip the threshold.

    This is the claim DEFAULT_THRESH = 30.0 rests on: a correct kernel that
    rounds differently from the reference is not a bug.
    """
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(64, 64)).astype(np.float32)
    cand = np.nextafter(ref, np.float32(np.inf))
    assert residual_ratio(cand, ref) < DEFAULT_THRESH
