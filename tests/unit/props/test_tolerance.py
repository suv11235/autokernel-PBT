"""Test-ratio tolerance tests."""

import numpy as np
import pytest

from autokernel_pbt.props.tolerance import (
    DEFAULT_THRESH,
    ExactDtypeError,
    residual_ratio,
    within_threshold,
)

EPS32 = np.finfo(np.float32).eps
EPS64 = np.finfo(np.float64).eps


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


# --- the n normalization ----------------------------------------------------


def test_ratio_normalizes_by_sqrt_of_the_accumulation_length():
    """Pin the normalization law: sqrt(n), not n and not 1.

    This is the most consequential constant in the module — it sets the detection
    floor, and a silent change to linear n would quadruple that floor at n=4096
    without failing anything else in this file.
    """
    # Exactly representable in float32, so the residual is exactly delta and the
    # absolute assertion below can be tight rather than approximate.
    delta = 2.0**-13
    short = np.ones((64,), dtype=np.float32)
    long = np.ones((4096,), dtype=np.float32)
    r_short = residual_ratio(short + delta, short)
    r_long = residual_ratio(long + delta, long)
    # Same residual, same scale: the ratios differ only by the normalization.
    assert np.isclose(r_short / r_long, np.sqrt(4096 / 64), rtol=1e-9)
    # And the absolute value follows the stated formula.
    assert np.isclose(r_short, delta / (EPS32 * np.sqrt(64)), rtol=1e-6)


def test_explicit_n_overrides_the_last_axis():
    """A caller comparing a derived quantity must be able to state the real n.

    Task 10 compares row sums of an (R, C) output: the array it passes has shape
    (R,), so the default would normalize by the row count instead of the
    reduction length.
    """
    sums = np.full((8,), 1.0, dtype=np.float32)
    cand = sums + 1e-4
    default = residual_ratio(cand, sums)
    explicit = residual_ratio(cand, sums, n=4096)
    assert np.isclose(default / explicit, np.sqrt(4096 / 8), rtol=1e-9)


def test_non_positive_n_is_rejected():
    x = np.ones((4,), dtype=np.float32)
    with pytest.raises(ValueError, match="positive accumulation length"):
        residual_ratio(x, x, n=0)


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
    assert np.isclose(r64 / r32, EPS32 / EPS64, rtol=1e-9)


def test_integer_candidate_is_rejected_with_a_clear_message():
    ref = np.array([1, 2, 3])
    with pytest.raises(ExactDtypeError, match="no unit roundoff"):
        residual_ratio(ref, ref)


def test_python_list_candidate_is_rejected():
    # np.asarray([1, 2, 3]).dtype is int64: the same integer path, reached by
    # accident rather than on purpose.
    with pytest.raises(ExactDtypeError, match="no unit roundoff"):
        residual_ratio([1, 2, 3], [1, 2, 3])


def test_integer_candidate_is_accepted_with_an_explicit_dtype():
    cand = np.array([1, 2, 3])
    assert residual_ratio(cand, cand, dtype=np.float32) == 0.0


def test_bool_candidate_is_rejected():
    x = np.array([True, False])
    with pytest.raises(ExactDtypeError, match="no unit roundoff"):
        residual_ratio(x, x)


def test_exact_dtype_is_rejected_even_when_the_shapes_also_mismatch():
    """The raise must not be bypassable by a second defect.

    Every other early return yields inf, and inf becomes FAIL downstream. If the
    dtype check ran after the shape check, an int-returning kernel with a shape
    bug would be recorded as a caught bug rather than as inapplicable.
    """
    with pytest.raises(ExactDtypeError):
        residual_ratio(np.array([1, 2, 3]), np.array([1, 2]))


def test_exact_dtype_error_is_narrower_than_value_error():
    # Task 11 catches this specifically; a bare `except ValueError` there would
    # swallow the n-validation error too and deflate the detection denominator.
    assert issubclass(ExactDtypeError, ValueError)


# --- shape and degenerate inputs -------------------------------------------


def test_shape_mismatch_is_infinite():
    assert np.isinf(residual_ratio(np.ones((4,)), np.ones((5,))))


def test_scalar_reference_is_normalized_like_the_persisted_output():
    # A reduction's persisted output is (1,) while np.sum gives (). Normalizing
    # only one side would return inf for every reduction case.
    assert residual_ratio(np.array([3.0]), np.float64(3.0)) == 0.0


def test_empty_arrays_are_not_a_pass():
    """Zero elements is not evidence, so it must not read as a pass.

    NaN rather than 0.0 so the guard is structural at the call site the oracles
    write, and distinguishable from the inf of a real mismatch.
    """
    empty = np.zeros((0,), dtype=np.float32)
    assert np.isnan(residual_ratio(empty, empty))
    assert not within_threshold(residual_ratio(empty, empty))
    # A zero dimension anywhere, not just a 1-D empty.
    assert np.isnan(residual_ratio(np.zeros((4, 0)), np.zeros((4, 0))))


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


def test_threshold_is_bracketed_from_both_sides():
    """Pin DEFAULT_THRESH itself, not just "well under it".

    The rounding tests below pass by ~1000x and would pass at THRESH=0.01, so
    they constrain the constant only from above. These two perturbations sit
    either side of the detection floor that 30.0 implies.
    """
    ref = np.ones((64,), dtype=np.float32)
    floor = DEFAULT_THRESH * EPS32 * np.sqrt(64)
    assert residual_ratio(ref + np.float32(floor * 0.5), ref) < DEFAULT_THRESH
    assert residual_ratio(ref + np.float32(floor * 2.0), ref) > DEFAULT_THRESH


def test_correctly_rounded_float32_has_wide_margin():
    """A correct kernel that merely rounds differently must be nowhere near the
    threshold. Measured ~0.004: three orders of margin, not a near miss."""
    ref = np.random.default_rng(0).normal(size=(64, 64))
    cand = ref.astype(np.float32).astype(np.float64)
    assert residual_ratio(cand, ref, dtype=np.float32) < DEFAULT_THRESH / 100


def test_single_ulp_per_element_has_wide_margin():
    """One ulp of float32 error on every element is not a bug, and must not be
    anywhere near the threshold either. Measured ~0.07 at n=64."""
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(64, 64)).astype(np.float32)
    cand = np.nextafter(ref, np.float32(np.inf))
    assert residual_ratio(cand, ref) < DEFAULT_THRESH / 100
