# Feature 0014: The log-ratio law, softmax's completing law (E2, first half)

## Problem

Every law in the softmax bundle reads an output alone (finite, in [0, 1], rows sum to one) or
compares an output with another output (shift invariance). None ties an output to the input it
came from, so the bundle admits softmax(βx) at every β and every output permutation
P softmax(x) Q (the paper's Proposition 1(a)). The gate decides on the declarative arm, and
it admits softmax(2x), softmax(−x), rows rolled by one, columns reversed, and the constant 1/n
fill, which is 52–54× faster than the loop's softmax. Proposition 1(b) says that adding the
**log-ratio law** leaves softmax and nothing else: y_ij > 0, and log y_ij − x_ij is constant
in j for each row i. Over the reals that is exact. E2 needs it as a floating-point check that
rejects those families and raises no new false positive on a correct kernel, device runs
included. The paper leaves the float form open.

The headroom record (`docs/measurements/2026-10-14-log-ratio-headroom-prereg.md`) measured the
law's statistic on every softmax we hold. Its finding: a correct kernel's spread grows with the
row's input range (r = 0.92 on the A10), so a flat threshold would run out on the wide rows a
real call site sends. Normalized by (1 + row range), the worst cases are 1.43 eps for a correct
kernel (A10; 1.05 on the CPU), 17.2 for fast-exp, and at least 6.38e6 for every wrong output.
(The headroom record printed 8.97 for fast-exp; E2 found that figure was taken from fast-exp's
worst-spread case rather than its worst ratio, and the record carries an erratum.)

## The law's float form

One recorded case: input `x` and output `y`, same shape, rows on the last axis, both read in
float64. `eps` and `tiny` are the unit roundoff and the smallest normal of **y's** dtype, and
`c = 30`. In each row i, every entry is exactly one of:

- **masked**: x_ij = −∞;
- **judged**: x_ij finite and y_ij ≥ tiny;
- **underflowed**: x_ij finite and 0 ≤ y_ij < tiny (zero or subnormal).

A row is **live** if it has a finite input, and **anchored** if it has a judged entry. On
judged entries, d_ij = log y_ij − x_ij, and κ_i = max_j d_ij.

The checks, in this order; the first one violated is the FAIL and its detail:

1. **Positivity.** y_ij ≥ 0 wherever x_ij is finite. A negative output is outside the law
   (Proposition 1(b) requires y > 0; an exact zero is check 4's question).
2. **Masked entries carry no mass.** In a live row, y_ij = 0 exactly wherever x_ij = −∞:
   exp(−∞ + κ) = 0 for every finite κ. No tolerance is involved.
3. **Spread.** On every anchored row,
   `(max_j d_ij − min_j d_ij) / ((1 + max_j x_ij − min_j x_ij) · eps) < c`,
   both maxima and minima taken over the row's judged entries.
4. **Underflow is predicted, not excused.** Every underflowed entry of an anchored row has
   `(x_ij + κ_i − log tiny) / ((1 + max_k x_ik − x_ij) · eps) < c`, the maximum over the row's
   finite inputs. That is, the law's own prediction for that entry is below the smallest normal,
   up to the same budget.

A case with no anchored row is INCONCLUSIVE. A PASS reports the largest spread ratio.

### Why c = 30 and (1 + range)

- **c is the project's one accuracy budget, not a fitted number.** It is `DEFAULT_THRESH`, the
  LAPACK test-ratio threshold that `rows_sum_to_one`, `shift_invariance` and the reference arm
  already use, and it was chosen before any of the numbers above existed. The statistic is a test
  ratio in the same sense: a residual divided by eps and by a model of how the error grows.
- **(1 + range) models the dominant error term** (CLAUDE.md: a tolerance must model the dominant
  mechanism). For a correct float kernel, d_ij is off by the rounding of x − max, about u·|x − max|
  (twice that when exp is computed as exp2(x · log2 e)), plus a few ulps from exp itself. The
  division by the row sum is a row constant and cancels in the spread. So the spread is about
  eps · (a · range + b), with a between 1 and 2 and b a few units.
- **The margins on everything recorded.** The worst correct kernel is 21× below c (1.43, on the
  A10), and the weakest wrong output is 2.1e5× above it (6.38e6).
- **Fast-exp is admitted, and that is the stated budget.** Its ratio is 17.2: its exp is off by up
  to 2.6e-6 relative, about 22 ulp. Under the same budget the reference arm, allclose and the
  current bundle all admit it, so a c below 17.2 would make the completed arm the only one to
  reject it. That would be a stricter accuracy budget than the reference arm's, adopted inside a
  law. The alternative was considered, with c between 1.43 and 17.2. It is not taken: c would
  then be picked between two measured numbers, on one GPU and one Triton kernel, instead of fixed
  in advance. E2 found the choice matters only on the gate's own domain. On rows that underflow,
  or that carry −∞ masks, the law rejects this fast-exp at any c below about 2e6. Its argument
  floor leaves the smallest normal float where softmax underflows, and about 1e-39 on a masked
  score (`docs/measurements/2026-10-14-e2-log-ratio-law-prereg.md`, P10).

### Why the range is taken over judged entries only

An additive mask of −1e9 (or the dtype's lowest value) is a finite input. Over the whole row, the
range would be 1e9, and the bound 30 · 1e9 · eps ≈ 3.6e3 in log space, which is vacuous: a wrong
kernel would pass any row that carries such a mask. A correct kernel's judged entries lie within
about 87 of the row's maximum (below that, float32 underflows), so the range over them stays
bounded, and it is exactly the range the spread's error grows over. A wrong kernel cannot inflate
the range by giving a masked-like entry a normal output: its d there is off by about the range
itself, so the ratio stays near 1/eps.

### Masked and underflowed entries

- **x = −∞ (a masked score)** must give exactly 0. A kernel that leaks probability to a masked
  position is wrong whatever the leak's size, and float softmax produces an exact 0 there.
- **A fully masked row is not judged.** Over the reals its softmax is 0/0, and kernels differ in
  what they return (NaN, zeros, uniform). That is a convention the contract, not this law, has to
  decide. A NaN there is `outputs_are_finite`'s finding.
- **Zero and subnormal outputs at finite inputs** are excluded from the spread, since a subnormal
  has lost relative precision and a device that flushes to zero returns 0 there. They are still
  checked against the law's prediction, by check 4. Excluding them unchecked would let a kernel
  that zeros every entry but the row's largest, and renormalizes, pass: one judged entry per row
  has no spread at all.
- **A live row with no judged entry** (every output below normal) is consistent with the law for
  a very negative κ, so it is not judged. Its row sum is far from one, which `rows_sum_to_one`
  owns. For a correct softmax the row's largest output is at least 1/n, so this never happens.

### INCONCLUSIVE, never a raise

The output is unusable (failed status or missing), empty, non-finite (`outputs_are_finite` owns
that, as declared in `defers_nonfinite_to`), or of a non-float dtype. Or there is no primary input
of the output's shape, the input is not real-valued, the input contains NaN or +∞ (the law is
undefined there), or no row is anchored.

## Scope

1. **The property**: `RowsHaveConstantLogRatio` (`rows_have_constant_log_ratio`) in
   `src/autokernel_pbt/props/properties.py`, tier 1 (portable), not tolerance-free, deferring
   non-finite output to `outputs_are_finite`, registered.
2. **A completed softmax contract**: `kernels/tasks/softmax/acceptance.completed.yaml`, the
   current contract plus one criterion. The current `kernels/tasks/softmax/acceptance.yaml`
   does not change. The driver, every recorded run and `SOFTMAX_CASE_PROPERTIES` stay on it.
3. **The gate scores either bundle**: `pbt_gate.check_task(kernel, task, bundle=...)`, with
   `"current"` (the default, so the loop and every recorded number are unchanged) and
   `"completed"`.
4. **The saboteur matrix, re-recorded** under the completed bundle beside its current rows,
   which stay as measured.
5. **E2 for softmax**, pre-registered under `docs/measurements/` before it runs: the saboteur
   matrix, the two softmax symmetry kernels, the 26 loop candidates, the recorded A10 runs, and
   correct kernels on wider and masked inputs than the gate draws.

## Non-goals

- Layernorm's affine-tie law, the other half of E2's "complete the bundle"
- Mechanizing the rest of Proposition 1 (E2's formal half)
- Generating masked or wide inputs in the gate's or the corpus's domain. That changes regenerated
  bytes ("never change the stream key or draw order"). The law's handling of them is unit-tested
  here and measured offline in E2.
- What a kernel should output on a fully masked row
- Switching the loop's gate or the driver to the completed bundle: E3 compares the two
- The A10 runs of the correct `exp2` and reciprocal-multiply Triton rewrites behind the paper's
  54-group figure, which are not versioned in `runs/`

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
