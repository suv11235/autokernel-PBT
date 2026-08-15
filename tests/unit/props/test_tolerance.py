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


def test_ratio_normalizes_by_log2_of_the_accumulation_length():
    """Pin the normalization law: log2(n), not sqrt(n), not n, not 1.

    This is the most consequential decision in the module — it sets the detection
    floor. log2 is the pairwise-summation bound, which is what the backends here
    actually compute; the alternatives are the bound for sequential accumulation
    (n), a folk-statistical guess (sqrt), and no normalization at all. All four
    are pinned apart, so a silent change to any of them fails here.
    """
    # Exactly representable in float32, so the residual is exactly delta and the
    # absolute assertion below can be tight rather than approximate.
    delta = 2.0**-13
    short = np.ones((64,), dtype=np.float32)
    long = np.ones((4096,), dtype=np.float32)
    r_short = residual_ratio(short + delta, short)
    r_long = residual_ratio(long + delta, long)
    # Same residual, same scale: the ratios differ only by the normalization.
    observed = r_short / r_long
    assert np.isclose(observed, np.log2(4096) / np.log2(64), rtol=1e-9)
    assert not np.isclose(observed, np.sqrt(4096 / 64), rtol=0.05)
    assert not np.isclose(observed, 4096 / 64, rtol=0.05)
    assert not np.isclose(observed, 1.0, rtol=0.05)
    # And the absolute value follows the stated formula.
    assert np.isclose(r_short, delta / (EPS32 * np.log2(64)), rtol=1e-9)


def test_short_accumulations_never_divide_by_zero():
    # log2(1) is 0 and log2(2) is 1; the divisor floors at 1.0 for both.
    x = np.ones((1,), dtype=np.float32)
    delta = np.float32(2.0**-13)
    assert np.isclose(residual_ratio(x + delta, x), float(delta) / EPS32, rtol=1e-6)
    assert residual_ratio(x + delta, x, n=2) == residual_ratio(x + delta, x, n=1)


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
    assert np.isclose(default / explicit, np.log2(4096) / np.log2(8), rtol=1e-9)


def test_default_n_still_depends_on_memory_layout():
    """The layout asymmetry is reduced by log2, not removed.

    Identical numerical error in the two layouts of the same data still differs
    by log2(4096) = 12x under the default n. That is the reason `n` is explicit:
    this test pins the residual exposure so it cannot silently widen again, and
    documents that a caller relying on the default is choosing an axis.
    """
    delta = 2.0**-13
    row = np.ones((1, 4096), dtype=np.float32)
    col = np.ones((4096, 1), dtype=np.float32)
    wide = residual_ratio(row + delta, row)
    tall = residual_ratio(col + delta, col)
    assert np.isclose(tall / wide, np.log2(4096), rtol=1e-9)
    # Stating n makes the two layouts agree exactly.
    assert residual_ratio(row + delta, row, n=4096) == residual_ratio(col + delta, col, n=4096)


def test_non_positive_n_is_rejected():
    x = np.ones((4,), dtype=np.float32)
    with pytest.raises(ValueError, match="positive accumulation length"):
        residual_ratio(x, x, n=0)


def test_non_integral_n_is_rejected():
    # `int | None` is not enforced at runtime; n=2.5 would otherwise sail into
    # the divisor and produce a plausible-looking ratio.
    x = np.ones((4,), dtype=np.float32)
    with pytest.raises(TypeError, match="integer accumulation length"):
        residual_ratio(x, x, n=2.5)


def test_bad_n_is_rejected_even_when_another_defect_would_short_circuit():
    # Same bypass class as the dtype raise: every other exit yields inf or NaN,
    # so a check that runs after them is unreachable on exactly the inputs where
    # a caller most needs to hear about it.
    with pytest.raises(ValueError, match="positive accumulation length"):
        residual_ratio(np.ones((4,), dtype=np.float32), np.ones((3,), dtype=np.float32), n=0)
    empty = np.zeros((0,), dtype=np.float32)
    with pytest.raises(ValueError, match="positive accumulation length"):
        residual_ratio(empty, empty, n=-5)


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


def test_a_real_low_precision_bug_fails_at_a_long_reduction():
    """Constrain DEFAULT_THRESH from below with a bug, not with a derivation.

    The margin tests below bound the constant from above only, and a floor
    computed from DEFAULT_THRESH itself cannot bound it at all — both sides of
    such a comparison move together, so it holds for any positive value. This
    uses a real defect whose size comes from the hardware (a float16 accumulator
    in an otherwise-float32 softmax over 4096 elements) and asserts it is caught.
    Together with the margin tests, DEFAULT_THRESH is bracketed on both sides by
    measurements rather than by tautology.
    """
    n = 4096
    x = np.random.default_rng(0).normal(size=(n,)).astype(np.float32)
    shifted = (x - x.max()).astype(np.float32)
    exp32 = np.exp(shifted)
    reference = np.exp(shifted.astype(np.float64))
    reference = reference / reference.sum()
    buggy = exp32 / np.float32(exp32.astype(np.float16).sum(dtype=np.float16))
    assert not within_threshold(residual_ratio(buggy, reference, dtype=np.float32))


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
