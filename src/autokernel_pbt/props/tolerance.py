"""Normalized test ratios, after the LAPACK convention (LAWN 41 s7.1.1).

A test ratio is dimensionless: it divides the residual by the scale of the problem
and by the unit roundoff, so a single threshold covers every routine, size and
precision. This replaces per-dtype ``rtol``/``atol`` guesses.
"""

from __future__ import annotations

import numpy as np

# LAPACK uses 30.0 across its entire test suite.
DEFAULT_THRESH = 30.0

# dtype kinds that have a unit roundoff. Everything else (bool, signed and
# unsigned int) is exact and has no rounding budget to normalize by.
_INEXACT_KINDS = "fc"


class ExactDtypeError(ValueError):
    """A test ratio was requested for a dtype that has no unit roundoff.

    Named, rather than a bare ``ValueError``, because the callers that catch this
    (the oracles) map it to INCONCLUSIVE. ``residual_ratio`` can raise
    ``ValueError`` for unrelated reasons, and a bare ``except ValueError`` there
    would convert genuine failures into INCONCLUSIVE and silently deflate the
    denominator of the detection rate.
    """


def _machine_eps(dtype: type | np.dtype) -> float:
    """Unit roundoff of a floating dtype."""
    return float(np.finfo(dtype).eps)


def _resolve_eps(candidate: np.ndarray, dtype: type | np.dtype | None) -> float:
    """Pick the unit roundoff, defaulting to the candidate's own dtype.

    An integer or boolean candidate has no unit roundoff, and neither does a
    Python list (``np.asarray([1, 2, 3]).dtype`` is int64), so the default path
    would hand ``np.finfo`` a dtype it rejects. Raising beats the alternatives:
    substituting float64 eps would report a ratio measured against a rounding
    budget the data never had, and returning ``inf`` would be worse still, since
    ``within_threshold(inf)`` is False and the oracles turn that into a FAIL —
    recording a correct int-returning kernel as a caught bug, a false positive
    injected directly into the headline metric with no downstream signal. Only an
    exception has the arity to say "this question does not apply here".
    """
    resolved = np.dtype(dtype) if dtype is not None else np.asarray(candidate).dtype
    if resolved.kind not in _INEXACT_KINDS:
        msg = (
            f"dtype {resolved!r} is exact and has no unit roundoff, so a test ratio is "
            f"undefined for it; compare exact outputs for equality instead, or pass "
            f"dtype=<floating type> if this is a promoted low-precision computation"
        )
        raise ExactDtypeError(msg)
    return _machine_eps(resolved)


def residual_ratio(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    dtype: type | np.dtype | None = None,
    n: int | None = None,
) -> float:
    """‖candidate - reference‖_inf / (‖reference‖_inf * eps * sqrt(n)).

    Returns inf if either side is non-finite or the shapes disagree, and NaN if the
    inputs are empty.

    ``dtype`` selects the unit roundoff; it defaults to the candidate's dtype. Pass it
    explicitly when comparing a low-precision candidate promoted to float64. Raises
    ``ExactDtypeError`` if the resolved dtype is exact.

    ``n`` is the length of the accumulation the error grew over. It defaults to the
    last-axis length, which is right for this corpus's row-wise kernels (softmax,
    layernorm, reductions) but *only* when the candidate is the kernel's output. A
    caller comparing a derived quantity — row sums of an (R, C) output, shape (R,) —
    must pass ``n=C`` explicitly, because the last axis of what it is handing over is
    the row count, not the reduction length. A contraction length that is not an
    output dimension at all (GEMM's K) is likewise the caller's to supply.

    Empty inputs return NaN rather than 0.0. Agreement over zero elements is
    vacuously true, so 0.0 is arithmetically defensible, but it is not evidence, and
    ``within_threshold`` would read it as a pass at the exact call site the oracles
    write (``within_threshold(residual_ratio(...))``, with no zero-size check in
    between). NaN makes that a non-pass structurally. Properties that want to
    distinguish "nothing was checked" from "the values disagree" — INCONCLUSIVE
    rather than FAIL — can test ``np.isnan`` against the ``inf`` returned for a real
    mismatch. ``_LADDER_SHAPES`` has no zero dimension today, but ``InputDomain``
    does not forbid one.
    """
    # Resolve eps FIRST. Every other early return yields inf, so resolving later
    # would let an exact-dtype candidate that also has a shape mismatch escape as
    # inf -> FAIL, which is the false positive this raise exists to prevent.
    eps = _resolve_eps(candidate, dtype)

    # atleast_1d on BOTH sides. np.ascontiguousarray is documented ndmin=1, so it promotes
    # a 0-d array to (1,) before safetensors ever sees it — safetensors itself round-trips
    # shape [] faithfully. Since the execution table calls ascontiguousarray on every
    # tensor, a reduction's persisted output is (1,) while its in-memory reference from
    # np.sum is (). Normalizing only one side would return inf for every reduction case.
    cand = np.atleast_1d(np.asarray(candidate, dtype=np.float64))
    ref = np.atleast_1d(np.asarray(reference, dtype=np.float64))
    if cand.shape != ref.shape:
        return float("inf")
    if cand.size == 0:
        return float("nan")
    if not np.all(np.isfinite(cand)) or not np.all(np.isfinite(ref)):
        return float("inf")

    # Finite float64 inputs of opposing sign near finfo.max overflow on subtraction,
    # and pyproject sets filterwarnings = ["error"], so an unguarded RuntimeWarning
    # here is a hard failure on data that is merely extreme. Suppress the warning and
    # read the overflow off the result: a difference too large to represent is a
    # disagreement past any threshold, which is what inf already means here. Only
    # `over` is suppressed — IEEE-754 subtraction of two finite operands cannot
    # produce NaN, and the finiteness check above already rejected inf and NaN
    # inputs, so a future refactor that does introduce an invalid path should fail
    # loudly rather than be silently absorbed here.
    with np.errstate(over="ignore"):
        residual = float(np.max(np.abs(cand - ref)))
    if not np.isfinite(residual):
        return float("inf")

    scale = float(np.max(np.abs(ref)))
    # A zero reference has no scale of its own; fall back to unit scale so the
    # ratio stays finite and still measures absolute deviation in units of eps.
    scale = scale if scale > 0.0 else 1.0

    length = cand.shape[-1] if n is None else n
    if length < 1:
        msg = f"n must be a positive accumulation length, got {length}"
        raise ValueError(msg)
    # sqrt(n), not n. Linear n is LAPACK's worst-case bound for *sequential*
    # accumulation, where every rounding error is assumed to align. Every backend in
    # this corpus reduces pairwise, and rounding errors of alternating sign random-walk
    # rather than accumulate, so the realistic bound is statistical. Measured float32
    # sum relative error: 8.3e-8 at n=64, 1.3e-7 at n=4096, 1.9e-7 at n=16384 — a 2.3x
    # rise across a 256x rise in n, nowhere near the 256x that linear n budgets for.
    # Over-normalizing is not a harmless safety margin: it raises the detection floor
    # in step with n, and at n=4096 linear n let a softmax whose denominator was 0.3%
    # wrong score 6.1 and pass, a bug the field-default allclose catches. A reference
    # arm that loses to the baseline it exists to beat inverts the whole comparison.
    return residual / (scale * eps * length**0.5)


def within_threshold(ratio: float, thresh: float = DEFAULT_THRESH) -> bool:
    return bool(np.isfinite(ratio) and ratio < thresh)
