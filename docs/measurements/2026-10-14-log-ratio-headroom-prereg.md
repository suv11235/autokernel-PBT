# Pre-registration: does one tolerance for the log-ratio law accept every correct softmax, device runs included, and reject every wrong family?

**Date:** 2026-10-14 (registered before any code for this measurement exists; results are
appended below the line, never edited above it)
**Segment:** generation, as design input for E2 (complete the bundle). Translation inherits the
answer, because the same law would gate a port with no reference.
**Instance:** Apple M4 Pro, CPU only; the project's pinned environment (NumPy 1.26.4)
**Companions:** `2026-10-01-softmax-temperature-blind-spot.md` (the log-ratio column, first
printed), `2026-10-14-symmetry-probe.md` (two more softmax families), and
`2026-08-17-first-hardware-run.md` / `2026-08-17-at-scale-hardware-run.md` (the device runs
re-scored here)

## 0. The question

Proposition 1(b) says the softmax bundle plus the **log-ratio law** (log y_ij − x_ij constant
along each row) admits softmax and nothing else, over the reals. E2 must turn that law into a
floating-point check, and its pre-registered success criterion includes *no new false positive*
on correct kernels, device groups among them. The paper leaves the law's float form open. This
measurement asks the narrow question that decides whether a simple form can work: **is there one
threshold on the law's statistic that every correct softmax we hold stays under, on the CPU and
on the recorded A10 runs, while every wrong family we hold exceeds it?** It changes no bundle, no
arm and no gate.

## 1. Construction

**The statistic.** For a case with recorded float32 input x (m × n) and output y, in float64:
d_ij = log y_ij − x_ij over the entries with y_ij > 0; per row, spread_i = max_j d_ij − min_j
d_ij; the case's spread is max_i spread_i divided by float32 eps (2⁻²³). Entries with y_ij = 0
are excluded from the spread and counted separately; so are subnormal y_ij. A one-column row has
spread 0 by definition. This is `saboteurs.log_ratio_spread_eps` with the underflow made
explicit instead of turning the spread into ∞.

**The threshold examined:** 30, the gate's existing ratio threshold (`DEFAULT_THRESH`). Chosen
now, not fitted to the results.

**Correct kernels (the law must accept every case):**

| group | kernels | cases |
|---|---|---|
| NumPy, gate cases | `softmax_reference` (float64, cast); the float32 stock kernel (`saboteurs.softmax_beta(1.0)`); NumPy float32 analogues of the two correct Triton rewrites, `exp2(x·log2 e)` and multiply-by-reciprocal; every loop candidate's `softmax` in `candidates/` except fast-exp, and `best/train.py`'s | `pbt_gate` cases: seed 42, 9 groups, base + `shift_rows`, 18 per kernel |
| A10, recorded | the stock Triton softmax in `runs/gpu-softmax/`, `runs/gpu-tolerance_sweep/` and `runs/gpu-scale/` | 18 + 7 + 5 executions, read offline with `ExecutionTable` |

**Wrong kernels (the law must reject):** softmax(2x), softmax(0x), softmax(−x), the constant
1/n fill, rows rolled by one, columns reversed; gate cases. **Reported, not classified:** fast-exp
(`candidates/r3_softmax_fastexp.py`), whose admission the paper leaves to a stated accuracy
budget, and the positive control exp(x − max) undivided, whose log-ratios are exact by
construction and which the row-sum law already rejects.

**Reported per case:** shape, the spread in eps, the row range max_j x − min_j x of the worst row,
and the counts of zero and subnormal outputs.

## 2. Already known, so not predictions

From the companion records, on the gate cases: the stock float32 kernel 4.8 eps, fast-exp 48,
the positive control 4.3, the temperature family 5.2e7 (β = 2 and β = 0) and 1.04e8 (β = −1),
rows rolled 6.76e7, columns reversed 8.72e7, the float64 reference 0.956. A throwaway check of the
device runs has not been made; their spreads are unknown to me at registration.

## 3. Predictions

- **P1 (CPU, correct).** Every correct NumPy kernel's worst gate-case spread is at most **8 eps**.
- **P2 (device, correct).** Every recorded A10 execution's spread is below **30 eps**.
- **P3 (no underflow).** No correct case, CPU or device, has an output equal to 0 or subnormal.
  These domains are N(0, 1) inputs, and shifts move a row uniformly.
- **P4 (one threshold works).** At 30: every correct case is accepted, every wrong family is
  rejected on every multi-column group, and fast-exp (48) is rejected.
- **P5 (margin).** The smallest wrong-family spread exceeds the largest correct spread by at
  least **10⁶**.

**Falsifier.** Any correct case at or above any wrong family's spread: then no single threshold on
this statistic separates them, and E2 needs a different float form (for example, a bound that
scales with the row range).

## 4. What is NOT measured

- **Masked rows.** The model's real call site sends −∞ scores, where y = 0 and log y − x is
  undefined. No recorded run or gate case contains −∞; the law's handling of masked entries is
  E2's design, not this record's.
- **The correct Triton rewrites on device.** The A10 runs of the `exp2` and reciprocal-multiply
  kernels behind the paper's 54-group false-positive figure are not versioned in `runs/`, so only
  their NumPy analogues are measured. E2's device criterion is therefore only partly answered.
- **Layernorm.** Its completing law, the affine tie, needs its own float form.
- **Any agent or loop.** This is an offline measurement over fixed kernels.

## 5. Deviation rules

- The statistic, the threshold and the kernel list do not change after the first number is seen.
- A correct case that underflows is reported with its count, never dropped.
- If a kernel cannot be loaded or run, it is reported as such, not replaced.

---

## Results (appended 2026-10-14, after the registration above was committed)

**Run:** `cd experiments/generation/log-ratio-headroom && ../../../.venv/bin/python headroom.py`,
in the registered environment (Apple M4 Pro, Python 3.12.2, NumPy 1.26.4). The script prints
every case; the numbers below are its summary. Nothing above the line was edited.

| kernels | cases | worst spread (eps) | where |
|---|---|---|---|
| correct, CPU: the reference, the float32 stock kernel, the `exp2` and reciprocal-multiply analogues, `best/train.py`, and the 24 non-fast-exp files in `candidates/` (29 kernels) | 522 | **7.29** | `exp2` analogue, (7, 129) |
| correct, A10: the stock Triton softmax in `gpu-softmax`, `gpu-tolerance_sweep`, `gpu-scale` | 30 | **12.3** | `softmax_at_scale`, (1024, 8192) |
| fast-exp (reported) | 18 | 48 | (7, 129) |
| exp(x − max), undivided (reported) | 18 | 4.3 | (7, 129) |
| wrong families, weakest case with nonzero spread, as registered | — | **0.681** | rows rolled by one, (1, 64) |
| wrong families, weakest case whose output differs from the reference (post hoc) | — | 2.67e7 | softmax(2x) and the fill, (8, 8) |

Zero and subnormal outputs across all 552 correct cases: **0 and 0**. All 24 candidate files,
`best/train.py` and the reciprocal-multiply analogue reach the stock kernel's 4.8 eps exactly,
so the 29 CPU kernels show only three distinct worst cases: 0.956, 4.8 and 7.29.

### Predictions

- **P1 holds, narrowly.** The largest correct CPU spread is 7.29 eps, under the registered 8. It
  is the `exp2` analogue; every other correct NumPy kernel is at or below 4.8.
- **P2 holds.** The largest A10 spread is 12.3 eps, under 30.
- **P3 holds.** No correct case, CPU or device, underflows to zero or to a subnormal.
- **P4 fails as registered.** Every correct case is accepted at 30, and fast-exp (48) is
  rejected. But one wrong-family case is accepted: rows rolled by one, on the single-row rung
  (1, 64), at 0.681 eps. On that case rolling the rows is the identity, so the kernel's output
  equals the reference's exactly; the symmetry probe already counts it as uncatchable by every
  arm (6 of 9). The registered wording, "every multi-column group", should have excluded it and
  did not. With a criterion chosen **after** seeing the numbers, counting only cases where a wrong
  kernel's output differs from the reference's, every wrong case is rejected, the weakest at
  2.67e7 eps.
- **P5 fails as registered** for the same case: margin 0.0554. On the post-hoc criterion the
  margin is 2.17e6, above the registered 10⁶.
- **The falsifier** fires as registered, on that same case, and on no case where a kernel is
  actually wrong.

### What it says for E2

1. **On everything recorded, one threshold works.** At 30 eps the log-ratio statistic accepts
   every correct softmax we hold, the A10 runs included, and rejects every wrong output and
   fast-exp. No underflow handling was needed on these domains.
2. **The device headroom is small, and it shrinks with the input's range.** It is 2.4× (12.3
   against 30). Across the 26 multi-column A10 executions the spread rises with the worst row's
   input range (max x − min x): Pearson r = 0.92, about 1.6 eps per unit of range, over ranges
   1.97 to 8.40. Extrapolated, a correct Triton softmax reaches 30 eps near a range of 20. The
   gate draws N(0, 1) rows; real attention scores are not N(0, 1) and can span more than that.
   **If the trend holds, a raw threshold of 30 would raise false alarms at a real call site.**
3. **Exploratory, not registered:** the spread divided by (1 + row range) is at most 1.43 on the
   A10 and 1.05 on the CPU, 8.97 for fast-exp, and at least 6.38e6 for every differing wrong
   case. A range-aware bound, spread ≤ c · (1 + range) eps, separates every case here for any c
   between 1.43 and 8.97, and admits fast-exp too for any c up to 6.38e6. This is the float form
   E2 should test first, on inputs with a wider range than any recorded here.
4. **Hypothesis, untested:** the range dependence comes from computing exp as exp2(x · log2 e),
   where rounding the product adds an error that grows with |x − max x|. The NumPy `exp2`
   analogue shows the same pattern on the same cases (7.29 eps against the stock kernel's 4.8).

### Deviations

- **Post-hoc criterion** for which wrong cases count: output not equal to the reference's. It is
  reported beside the registered verdict, not instead of it.
- **Post-hoc exploratory statistic,** spread / (1 + row range), with the linear fit above. The
  registered statistic, threshold and kernel list did not change, and the script's registered
  tables print the same bytes with and without the additions.
- **The two symmetry kernels are built in `headroom.py`,** the same way as in
  `symmetry_probe.py`, because this branch does not contain that script yet.

### Threats to validity

- **One device, one compiler, one Triton kernel:** an A10, Triton 3.3.0, the stock softmax. The
  correct Triton rewrites are measured only as NumPy analogues.
- **Narrow inputs.** Every recorded row range is 8.4 or less, and nothing is masked. Point 2's
  extrapolation and point 3's bound are untested beyond that.
- **Author-chosen wrong families,** as in the companion records.
- **Few distinct correct behaviours** on the CPU: three distinct worst cases from 29 kernels.

### Erratum (appended 2026-10-14, found by E2)

Point 3 of "What it says for E2" gives fast-exp's spread / (1 + row range) as **8.97**. That
value is the statistic on fast-exp's worst-*spread* case, (7, 129). `headroom.py` prints
`reported[label].per_range`, where `reported[label] = worst(spreads)` is chosen by raw spread.
Fast-exp's largest value of the statistic over its 18 gate cases is **17.19**, on (8, 8)
(spread 43.7 eps over a row range of 2.89). The other three numbers in that point (1.43, 1.05,
6.38e6) are true maxima and minima and are unaffected. So is every registered verdict above the
line. With the corrected figure, the range of c that separates every case and rejects fast-exp is
1.43 to 17.2, and fast-exp is admitted for any c from 17.2 to 6.38e6. Found and confirmed by
`docs/measurements/2026-10-14-e2-log-ratio-law-prereg.md` (P0). `headroom.py` is left as it was,
so that its output still matches the tables this record quotes.
