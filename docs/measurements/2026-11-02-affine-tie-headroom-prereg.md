# Pre-registration: does one tolerance for the affine-tie law accept every correct layernorm, device runs included, and reject every wrong family?

**Date:** 2026-11-02 (registered before any code for this measurement exists; results are
appended below the line, never edited above it)
**Segment:** generation, as design input for E2 (complete the bundle). Translation inherits the
answer, because the same law would gate a port with no reference.
**Instance:** Apple M4 Pro, CPU only; the project's pinned environment (NumPy 1.26.4)
**Companions:** `2026-10-14-log-ratio-headroom-prereg.md` (the same question for softmax),
`2026-10-14-e2-log-ratio-law-prereg.md` (softmax's half of E2),
`2026-10-12-te1-triton-to-cuda-prereg.md` and `2026-08-17-first-hardware-run.md` (the device runs
re-scored here)

## 0. The question

Proposition 1(c) says the layernorm bundle admits −LN, P LN(x) Q and row-wise sorting of LN(x).
Adding the **affine-tie law** (for each row i there is an s_i > 0 with
y_ij − y_ik = (x_ij − x_ik) / s_i) leaves LN alone on every non-constant row, over the reals. E2
must turn that law into a floating-point check, with no new false positive on a correct kernel.
This measurement asks the narrow question that decides whether a simple form can work: **is
there one threshold on the law's statistic that every correct layernorm we hold stays under, on
the CPU and on the recorded device runs, while every wrong family we hold exceeds it or has the
wrong slope?** It changes no bundle, no arm and no gate.

## 1. Construction

**The statistic.** For a case with recorded float32 input x (m × n) and output y, in float64, per
row: x̃ = x − mean x and ỹ = y − mean y; the slope a = (x̃ · ỹ) / (x̃ · x̃), the least-squares
estimate of 1 / s, or 0 on a row with a constant input; the residual r = ỹ − a x̃. Then

    ratio = max_j |r_j| / (eps · max_j |y_j| + tiny)

with eps and tiny the unit roundoff and the smallest normal of y's dtype. The case's statistic is
its worst row's ratio. A one-column row has ratio 0 by definition. The case's **slope** is the
smallest a over its rows with a non-constant input, and the law requires it to be positive.

**Why this form, before any number exists.** A correct kernel computes y_j = (x_j − μ̂) / σ̂ for
row constants μ̂ and σ̂. Their errors are row constants too, so they move the intercept and the
slope and leave the tie alone. What remains is the rounding of the subtraction and of the
division, each relative to the entry's own magnitude, so |e_j| is about eps · |y_j|, and
max |y_j| is the scale. Unlike softmax's law there is no range term: nothing here grows with the
input's distance from anything. Whether that holds is what P3 checks.

**The threshold examined:** 30, the gate's existing ratio threshold (`DEFAULT_THRESH`). Chosen
now, not fitted to the results.

**Correct kernels (the law must accept every case):**

| group | kernels | cases |
|---|---|---|
| NumPy, gate cases | `layernorm_reference` (float64, cast); the float32 stock kernel; a NumPy float32 analogue of the correct Triton rewrite `_layernorm_rsqrt`, the centered row times the reciprocal of the root; every loop candidate's `layernorm` in `candidates/`, and `best/train.py`'s | `pbt_gate` cases: seed 42, 9 groups, base + `shift_rows`, 18 per kernel |
| devices, recorded | the stock Triton layernorm in `runs/gpu-layernorm` (A10) and `runs/te1-triton-layernorm` (A6000); its CUDA port in `runs/te1-cuda-layernorm` (A6000) | 18 executions each, read offline with `ExecutionTable` |

**Wrong kernels (the law must reject wherever the output differs from the reference's):** −LN(x),
columns reversed, each row sorted, rows rolled by one, each the reference's output with that
symmetry applied; gate cases.

**Reported, not classified:**

- *Distributed division*, x / σ̂ − μ̂ / σ̂ in float32: correct over the reals, with rounding relative
  to |x| / σ̂ instead of |y|. A rewrite an optimizing agent might write, and the kernel most likely
  to show that the bound is about the input's offset as well.
- *Undivided*, x − mean: the positive control. Its tie is exact, with slope 1, and the
  unit-variance law already rejects it.
- *Zeroed*, an all-zero output of the input's shape: its residual is 0 and its slope is 0.

**Reported per case:** shape, ratio, slope, max |y_j| and max |x_j| on the worst row, and whether
the output equals the reference's.

## 2. Already known, so not predictions

- **The current bundle's verdicts on these kernels** are in the symmetry probe's rows (7, 7, 7 and
  6 of 9 against a reference, 0 of 9 declarative). No statistic of this kind has been computed on
  any of them.
- **The cross-setup numbers in TE1** (`2026-10-12-te1-triton-to-cuda-prereg.md`): the layernorm
  reference ratio has a maximum of 1.00 on each setup, and 0.986 between setups. These compare
  whole outputs, row constants included, so they bound this statistic loosely at best. A
  throwaway check of the device runs under this statistic has not been made, and its values are
  unknown to me at registration.

## 3. Predictions

- **P1 (CPU, correct).** Every correct NumPy kernel's worst gate-case ratio is at most **1.5**:
  two half-ulp operations give about eps · |y_j|, and the fit moves that by at most half again.
- **P2 (device, correct).** Every recorded device execution's ratio is below **3**. Device float32
  division need not be correctly rounded, so its budget is about twice the CPU's.
- **P3 (the model's shape).** Across the multi-column device executions, the ratio shows no trend
  in row length or in input offset: |Pearson r| < 0.5 against log2 n and against
  max |x| / (max |x − mean x|) on the worst row.
- **P4 (one threshold works).** At 30, with the slope positive: every correct case is accepted,
  and every wrong case whose output differs from the reference's is rejected. −LN is rejected
  only by its slope, and the other three by the ratio or the slope.
- **P5 (margin).** Over reversed, sorted and rolled cases that differ from the reference's and
  are not rejected by slope alone, the smallest ratio is at least **10⁶**, so the margin over the
  largest correct ratio exceeds **10⁵**.
- **P6 (controls).** −LN's ratio is at most 1.5 with every slope negative. Undivided's ratio is at
  most 1.5 with every slope within 1e-5 of 1. Zeroed's ratio is 0 with slope 0 on every
  multi-column row.
- **P7 (distributed division, reported).** Its worst ratio is above the stock kernel's and below
  30, and its ratio rises with input offset: Pearson r > 0.5 against the offset measure of P3.

**Falsifier.** Any correct case at or above 30, or with a non-positive slope; or any wrong case
that differs from the reference's and is accepted. Then this float form is wrong for E2.

## 4. What is NOT measured

- **Inputs the gate does not draw.** Every recorded row is U(−10, 10) with a shift of at most a
  few hundred. Wide, narrow and large-offset rows are E2's to test.
- **The correct Triton rewrite on device.** The A10 runs of `_layernorm_rsqrt` behind the paper's
  54-group false-positive figure are not versioned in `runs/`, so only its NumPy analogue is
  measured.
- **Long rows on device.** The widest recorded layernorm row is 129; no at-scale layernorm run
  exists.
- **Softmax,** and any agent or loop. This is an offline measurement over fixed kernels.

## 5. Deviation rules

- The statistic, the threshold and the kernel list do not change after the first number is seen.
- A correct case over a predicted bound is reported as a failed prediction, never dropped.
- If a kernel cannot be loaded or run, it is reported as such, not replaced.

---

## Results (appended 2026-11-02, after the registration above was committed)

**Run:** `cd experiments/generation/affine-tie-headroom && ../../../.venv/bin/python tie_headroom.py`,
in the registered environment (Apple M4 Pro, Python 3.12.2, NumPy 1.26.4). The script prints every
case; the numbers below are its summary. Nothing above the line was edited.

| kernels | cases | worst ratio | where |
|---|---|---|---|
| correct, CPU: the reference, the float32 stock kernel, the `rsqrt` analogue, `best/train.py`, and the 25 files in `candidates/` (29 kernels) | 522 | **0.673** | stock form, (5, 33) |
| correct, devices: Triton on the A10, Triton and its CUDA port on the A6000 | 54 | **0.708** | Triton on the A6000, (16, 32), shift partner |
| wrong, weakest case that the ratio rejects | — | **6.28e6** | columns reversed, (3, 7), shift partner |
| −LN(x) (rejected by its slope alone) | 14 differing | 0.381 | every slope at most −0.1486 |
| distributed division (reported) | 18 | 4.72 | (16, 32) |
| undivided (reported) | 18 | 0.523 | every slope within 2.52e-8 of 1 |
| zeroed (reported) | 18 | 0 | every slope 0 |

By device run, the worst ratios are 0.693 (`gpu-layernorm`), 0.708 (`te1-triton-layernorm`) and
0.633 (`te1-cuda-layernorm`). The 23 candidate files with the in-place form, the 2 with the fresh
quotient, `best/train.py` and the stock kernel reach the same 0.673, so the 29 CPU kernels show only
three distinct worst cases: 0.381, 0.626 and 0.673.

### Predictions

- **P1 holds.** The largest correct CPU ratio is 0.673, under the registered 1.5.
- **P2 holds.** The largest device ratio is 0.708, under 3. The devices sit at the CPU's level, not
  at twice it.
- **P3 holds, one part narrowly.** Over 42 multi-column device executions, r(ratio, log2 n) = 0.460,
  under the registered 0.5, and r(ratio, offset) = −0.257. Row lengths reach only 129.
- **P4 holds.** Every correct case is accepted. Of the 54 wrong cases that differ from the
  reference's, none is accepted. −LN's 14 are rejected by the slope alone, and the other 40 by the
  ratio.
- **P5 holds.** The weakest ratio is 6.28e6, above the registered 10⁶. The margin over the largest
  correct ratio is 8.87e6, above the registered 10⁵.
- **P6 holds.** −LN: worst ratio 0.381, and every slope at most −0.1486. Undivided: worst ratio
  0.523, and every slope within 2.52e-8 of 1. Zeroed: ratio 0, and slope 0 on every multi-column
  row, which rejects it on all 14 of its multi-column cases.
- **P7 holds.** Distributed division's worst ratio is 4.72, above the stock kernel's 0.673 and below
  30, and r(ratio, offset) = 0.963.
- **The falsifier does not fire.**

### What it says for E2

1. **One threshold and one sign check work on everything recorded.** The worst correct case is 42×
   below 30, and the weakest ratio rejection is 2.1e5× above it.
2. **The bound needs no range term.** Unlike softmax's law, a correct kernel's ratio does not rise
   with the input's offset. The rounding that remains is relative to each output, so max |y| is
   the scale. This is measured on rows up to 129 wide and offsets up to about 12.5. E2 should
   test it far past both.
3. **The ratio and the slope are separate checks, and neither is redundant.** −LN has a ratio of
   0.381, indistinguishable from a correct kernel, so only the slope rejects it. A zeroed output
   has no residual, so again only the slope rejects it. Sorted rows have a positive slope on every
   row in 4 of their 14 differing cases, so the slope alone would miss those.
4. **The undivided kernel stays the unit-variance law's.** Its tie is exact. The two laws are
   complements, as Proposition 1(c)'s proof uses them.
5. **Distributed division is admitted on the gate's domain, and its ratio climbs with the offset.**
   A linear fit gives 0.43 per unit of offset, which reaches 30 near an offset of 71.
   *Hypothesis, untested:* the budget would reject it there, on a kernel that is correct over the
   reals. This is the same kind of question as fast-exp's, and E2 should report it.

### Deviations

- **The P6 readout was corrected after the first run.** The first run read only each case's
  smallest slope, which cannot show that every slope is negative. The script now reads the largest
  as well. One line of its output changed, from −0.1758 to −0.1486, and no registered statistic or
  verdict moved.
- **Three numbers come from an ad hoc check, not the script:** the 4 of 14 sorted cases, the largest
  offset of 12.5 over every row, and distributed division's linear fit.

### Threats to validity

- **Narrow inputs.** Rows reach 129 wide, and offsets reach about 12.5, on U(−10, 10) with one
  shift per row.
- **Two device models, one kernel form.** The A10 and the A6000 are both sm_86, and the Triton and
  CUDA kernels share one arithmetic form. The correct Triton `rsqrt` rewrite is measured only as a
  NumPy analogue.
- **Author-chosen wrong families,** as in the companion records.
- **Few distinct correct behaviours** on the CPU: three distinct worst cases from 29 kernels.
