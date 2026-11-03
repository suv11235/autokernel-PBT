# Feature 0015: The affine-tie law, layernorm's completing law (E2, second half)

## Problem

Every law in the layernorm bundle reads an output alone (finite, zero row mean, unit row variance)
or compares an output with another output (shift invariance). None ties an output to the input it
came from, so the bundle admits −LN, every output permutation P LN(x) Q, and row-wise sorting of
LN(x) (the paper's Proposition 1(c)). The gate decides on the declarative arm, and it admits all of
them. Proposition 1(c) says that adding the **affine-tie law** leaves LN alone on every non-constant
row: for each row i there is an s_i > 0 with y_ij − y_ik = (x_ij − x_ik) / s_i. Over the reals that
is exact. E2 needs it as a floating-point check that rejects those families and raises no new false
positive on a correct kernel, device runs included.

The headroom record (`docs/measurements/2026-11-02-affine-tie-headroom-prereg.md`) measured the law's
statistic on every layernorm we hold. A correct kernel's worst ratio is 0.673 on the CPU and 0.708 on
the devices, with no trend in the input's offset. The wrong families reach at least 6.28e6, except
−LN, whose tie is exact and whose slope is what is wrong.

## The law's float form

One recorded case: input `x` and output `y`, same shape, rows on the last axis, both read in float64.
`eps` and `tiny` are the unit roundoff and the smallest normal of **y's** dtype, and `c = 30`. In
each row i, with x̃ = x − mean x and ỹ = y − mean y:

- the **slope** is a_i = (x̃ · ỹ) / (x̃ · x̃), the least-squares estimate of 1 / s_i, on a row with a
  **varying** input (max x > min x), and 0 on a row with a constant input;
- the **residual** is r_ij = ỹ_ij − a_i x̃_ij.

The checks, in this order; the first one violated is the FAIL and its detail:

1. **The tie.** On every row, `max_j |r_ij| / (eps · max_j |y_ij| + tiny) < c`.
2. **The slope.** On every varying row, `a_i > 0`.

A PASS reports the largest ratio.

### Why c = 30, and why max |y| with no range term

- **c is the project's one accuracy budget, not a fitted number.** It is `DEFAULT_THRESH`, the
  LAPACK test-ratio threshold the other laws and the reference arm already use, and it was chosen
  before any of the numbers above existed.
- **eps · max |y| models the dominant error term** (CLAUDE.md: a tolerance must model the dominant
  mechanism). A correct kernel computes y_j = (x_j − μ̂) / σ̂. The errors in μ̂ and σ̂ are row
  constants, so they move the intercept and the slope, which the law leaves free. What remains is
  the rounding of one subtraction and one division, each relative to the entry itself, about
  eps · |y_j|. Softmax's law needs a range term because its rounding grows with |x − max x| in log
  space. Nothing here does: across 42 multi-column device executions the ratio's correlation with
  the input's offset is −0.257.
- **tiny is the floor where y has lost relative precision.** A device that flushes subnormals has an
  absolute error of up to tiny there. It also gives the zero output of a constant row a ratio of 0
  rather than 0/0.
- **The margins on everything recorded.** The worst correct kernel is 42× below c (0.708), and the
  weakest wrong ratio is 2.1e5× above it (6.28e6).

### Why the slope is a separate check

- **−LN's tie is exact.** It has the ratio of a correct kernel (0.381), so only the slope rejects
  it. This is the condition s_i > 0 in Proposition 1(c), and it is what excludes −LN there.
- **A flat output on a varying row has no residual.** A zeroed buffer has r = 0 and a = 0. The law
  as stated has no finite s_i for it, so a = 0 is a violation, not a limit.
- **The slope alone is not enough.** Sorted rows have a positive slope on every row in 4 of their 14
  differing gate cases.

The slope carries no tolerance. A correct kernel's slope is about 1 / σ, and it is at least 0.1486
on everything recorded.

### Least squares, and what stays free

The law says only that some s_i exists, so the check must not assume its value. The least-squares
slope finds the s_i most favourable to the kernel, and any tie that holds at all has a small residual
under it. The bundle's other laws then pin the free parameters: zero mean fixes the intercept, and
unit variance fixes s_i. So the undivided kernel x − mean, whose tie is exact with slope 1, passes
this law and stays the unit-variance law's to reject.

### Constant and one-column rows

- **A constant input must give a flat output.** The law gives y_ij − y_ik = 0 for every s_i, and on
  such a row the slope is 0, so check 1 reads the output's own spread. A correct kernel subtracts
  one μ̂ from identical entries and divides by one σ̂, so its row is exactly flat. Unit variance
  abstains on these rows, so without this check a kernel could write anything with zero mean there.
- **A one-column row holds vacuously.** It has ratio 0 and no slope. So the law PASSes on the
  ladder's rungs (1, 1) and (17, 1), as the log-ratio law does, and it adds no abstention.

### Budget and limits

- **The budget admits distributed division on the gate's domain.** x / σ̂ − μ̂ / σ̂ is correct over the
  reals, and its rounding is relative to |x| / σ̂, not |y|. Its ratio is 4.72 on the gate's cases and
  climbs with the offset. E2 found that it crosses c between row offsets of 100 and 1e3, where the
  reference arm rejects it too on all but one input
  (`docs/measurements/2026-11-02-e2-affine-tie-law-prereg.md`, P9). So the law adds no stricter accuracy
  budget of its own.
- **A device that flushes subnormal differences to zero** can output a flat row for a varying input
  whose range is around 1e-35 or below, and the slope check would FAIL that. No domain draws such a
  row, and the law takes no extra case for it.

### INCONCLUSIVE, never a raise

The output is unusable (failed status or missing), empty, non-finite (`outputs_are_finite` owns that,
as declared in `defers_nonfinite_to`), or of a non-float dtype. Or there is no primary input of the
output's shape, the input is not real-valued, or the input contains NaN or ±∞ (the law is undefined
there). Or the arithmetic itself runs out of range, which takes float64 values near its limits, so
the ratio or slope is not finite.

## Scope

1. **The property**: `RowsHaveAffineTie` (`rows_have_affine_tie`) in
   `src/autokernel_pbt/props/properties.py`, tier 1 (portable), not tolerance-free, deferring
   non-finite output to `outputs_are_finite`, registered.
2. **A completed layernorm contract**: `kernels/tasks/layernorm/acceptance.completed.yaml`, the
   current contract plus one criterion. The current `kernels/tasks/layernorm/acceptance.yaml` does
   not change. The driver, every recorded run and `LAYERNORM_CASE_PROPERTIES` stay on it.
3. **The gate scores either bundle** for layernorm too, through
   `pbt_gate.check_task(kernel, "layernorm", bundle=...)`. The default stays `"current"`, so the loop
   and every recorded number are unchanged.
4. **A layernorm saboteur matrix**, recorded under both bundles beside softmax's: the reference, −LN,
   columns reversed, each row sorted, rows rolled, and the undivided positive control.
5. **E2 for layernorm**, pre-registered under `docs/measurements/` before it runs: that matrix, the
   26 loop candidates, the recorded device runs, and correct kernels on wider, narrower and more
   offset inputs than the gate draws.

## Non-goals

- Mechanizing Proposition 1(c) (E2's formal half)
- Generating wide or large-offset inputs in the gate's or the corpus's domain. That changes
  regenerated bytes ("never change the stream key or draw order"). The law's behaviour on them is
  unit-tested here and measured offline in E2.
- Switching the loop's gate or the driver to the completed bundle: E3 compares the two
- The A10 runs of the correct `_layernorm_rsqrt` rewrite behind the paper's 54-group figure, which
  are not versioned in `runs/`

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
