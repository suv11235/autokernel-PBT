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


def machine_eps(dtype: type | np.dtype) -> float:
    """Unit roundoff of a floating dtype. Raises ValueError for exact dtypes."""
    return float(np.finfo(dtype).eps)


def _resolve_eps(candidate: np.ndarray, dtype: type | np.dtype | None) -> float:
    """Pick the unit roundoff, defaulting to the candidate's own dtype.

    An integer or boolean candidate has no unit roundoff, and neither does a
    Python list (``np.asarray([1, 2, 3]).dtype`` is int64), so the default path
    would hand ``np.finfo`` a dtype it rejects. Rather than let that surface as
    numpy's bare "not compatible with finfo", or silently substitute float64 eps
    and report a ratio computed against a rounding budget the data never had,
    this raises with the dtype named and the two ways out spelled: compare exact
    dtypes exactly, or pass ``dtype=`` when a low-precision computation was
    promoted before it got here. Callers that evaluate whole runs (the oracles)
    should treat this as INCONCLUSIVE for the case, not as a kernel failure.
    """
    resolved = np.dtype(dtype) if dtype is not None else np.asarray(candidate).dtype
    if resolved.kind not in _INEXACT_KINDS:
        msg = (
            f"dtype {resolved!r} is exact and has no unit roundoff, so a test ratio is "
            f"undefined for it; compare exact outputs for equality instead, or pass "
            f"dtype=<floating type> if this is a promoted low-precision computation"
        )
        raise ValueError(msg)
    return machine_eps(resolved)


def residual_ratio(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    dtype: type | np.dtype | None = None,
) -> float:
    """‖candidate - reference‖_inf / (‖reference‖_inf * eps * n), or inf if non-finite.

    ``dtype`` selects the unit roundoff; it defaults to the candidate's dtype. Pass it
    explicitly when comparing a low-precision candidate promoted to float64.

    Two empty arrays of the same shape return 0.0: elementwise agreement over zero
    elements is vacuously true, and the shape check has already rejected empty
    against non-empty. That 0.0 is *not* evidence of correctness, and this module
    returns floats, so it cannot say so — a case whose tensors have a zero dimension
    is INCONCLUSIVE and the caller must classify it as such before consulting the
    ratio. ``_LADDER_SHAPES`` contains no zero dimension today, but ``InputDomain``
    does not forbid one, so the vacuous pass is reachable by construction.
    """
    # atleast_1d on BOTH sides. np.ascontiguousarray is documented ndmin=1, so it promotes
    # a 0-d array to (1,) before safetensors ever sees it — safetensors itself round-trips
    # shape [] faithfully. Since the execution table calls ascontiguousarray on every
    # tensor, a reduction's persisted output is (1,) while its in-memory reference from
    # np.sum is (). Normalizing only one side would return inf for every reduction case.
    cand = np.atleast_1d(np.asarray(candidate, dtype=np.float64))
    ref = np.atleast_1d(np.asarray(reference, dtype=np.float64))
    if cand.shape != ref.shape:
        return float("inf")
    if not np.all(np.isfinite(cand)) or not np.all(np.isfinite(ref)):
        return float("inf")

    eps = _resolve_eps(candidate, dtype)
    # Finite float64 inputs of opposing sign near finfo.max overflow on subtraction,
    # and pyproject sets filterwarnings = ["error"], so an unguarded RuntimeWarning
    # here is a hard failure on data that is merely extreme. Suppress the warning and
    # read the overflow off the result: a difference too large to represent is a
    # disagreement past any threshold, which is what inf already means here.
    with np.errstate(over="ignore", invalid="ignore"):
        residual = float(np.max(np.abs(cand - ref))) if cand.size else 0.0
    if not np.isfinite(residual):
        return float("inf")
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    # A zero reference has no scale of its own; fall back to unit scale so the
    # ratio stays finite and still measures absolute deviation in units of eps.
    scale = scale if scale > 0.0 else 1.0
    # n normalizes by the length of the accumulation, per LAPACK. Phase 1's corpus
    # accumulates along the last axis (softmax, layernorm, row-wise reductions), and
    # for elementwise kernels n is a harmless small constant. A contraction whose
    # length is not an output dimension (GEMM's K) is not expressible from the output
    # shape at all and would need its own normalization when that arrives.
    n = max(cand.shape[-1], 1) if cand.ndim else 1
    return residual / (scale * eps * n)


def within_threshold(ratio: float, thresh: float = DEFAULT_THRESH) -> bool:
    return bool(np.isfinite(ratio) and ratio < thresh)
