# The false-positive rate, and the first arm differentiation

**Date:** 2026-08-19
**Instance:** Lambda A10, sm_86, CUDA 12.8, torch 2.7.0, triton 3.3.0, numpy 1.26.4
**Population:** 6 correct kernels x 9 case groups = **54 groups**
**Companion:** `2026-08-18-arm-differentiation-null-result.md`, which found no detection difference
over 20 mutants

**Headline:** the arms differentiate **decisively** — just not on detection. On a population of
kernels every one of which is verified correct, `allclose` fails **10 of 54** groups and the other
three arms fail **none**.

---

## Why this measurement needed building

Metric 2 had no denominator. The 5-of-9 layernorm false-positive result was real but `n = 1`: one
correct kernel, on one task. A single kernel cannot distinguish "this arm is miscalibrated" from
"this particular kernel is unusual".

A false-positive rate also cannot be measured against a kernel that reproduces its reference
bit-for-bit — every tolerance-bearing arm is handed a residual of exactly zero and passes
vacuously. The population has to be **correct but numerically different**, which is the realistic
case: an optimizing agent rewrites a kernel, the result is right, and it differs in the last bits.

## Construction

Stated before the numbers, per the repo standard.

| | |
|---|---|
| population | 3 correct-but-different variants + the 3 stock ladder kernels |
| correctness check | the **admission gate**, which must *reject* each one as "not broken on any group" |
| unit | the case group, per the module contract |
| false positive | any group an arm summarizes as FAIL |
| shapes | the standard ladder, seed 42 |

The three variants are optimizations a performance-minded author actually writes:

- `softmax_exp2` — `exp2(x·log2e)` rather than `exp(x)`, since hardware has a native exp2
- `softmax_reciprocal` — one reciprocal and N multiplies rather than N divides
- `layernorm_rsqrt` — multiply by reciprocal square root rather than divide by square root

**The gate is the verification, not my assertion.** A "correct variant" the gate admits is not a
variant, it is a mutant — and one of them was, see §3.

## 1. The result

| kernel | task | allclose | reference | declarative | hybrid |
|---|---|---|---|---|---|
| softmax_exp2 | softmax | 0.000 | 0.000 | 0.000 | 0.000 |
| softmax_reciprocal | softmax | 0.000 | 0.000 | 0.000 | 0.000 |
| softmax_stock | softmax | 0.000 | 0.000 | 0.000 | 0.000 |
| relu_stock | relu | 0.000 | 0.000 | 0.000 | 0.000 |
| layernorm_rsqrt | layernorm | **0.556** | 0.000 | 0.000 | 0.000 |
| layernorm_stock | layernorm | **0.556** | 0.000 | 0.000 | 0.000 |
| **pooled** | | **10/54 = 0.185** | **0/54** | **0/54** | **0/54** |

Every kernel in this table is verified correct, so every non-zero entry is a false alarm.

**The 5-of-9 layernorm result replicates on an independent correct kernel.** `layernorm_rsqrt`
shares no arithmetic with `layernorm_stock` past the two-pass variance, and both fail the same
5 of 9 groups. That is the evidence `n = 1` could not supply: the miscalibration belongs to the
arm, not to a kernel.

**Pooled rates depend on task mix.** Two of six kernels are layernorm; a population weighted
differently moves 0.185. The per-task rates are the honest numbers — the pooled figure is
reported because it is the metric the design names, not because it is mix-independent.

## 2. What this changes about the null result

The companion measurement found **zero** arm differentiation over 20 mutants, and concluded that
where a trusted reference exists the declarative arm has no detection advantage. That stands.

This measurement is its complement, and it points the other way: **`allclose` is not less
sensitive than the strengthened reference arm, it is miscalibrated.** It fails correct kernels the
reference arm passes, on 18.5% of groups pooled and 55.6% of layernorm groups.

Read together, the two measurements say something narrower and better-supported than either alone:

> Against the field default, the contribution of this work is **not** catching more bugs. It is
> catching the same bugs without a false-alarm rate of 0.185 — and, for the declarative arm,
> without needing a reference at all.

A tool that flags a correct kernel on more than half of one task's shapes is not usable as a gate,
and `atol=1e-8` against float32 is the field default rather than an unusual choice.

The mechanism is unchanged and remains backend-independent: `atol=1e-8` sits ~12x below float32
eps, and a layernorm output is centered on zero by construction, so every row contains near-zero
elements whose entire error budget collapses to `atol`.

## 3. A variant that turned out to be a bug

`layernorm_sumsq` — variance as `E[x²] − E[x]²` rather than `E[(x − E[x])²]` — was written as a
fourth correct variant. **The gate admitted it**, which means it differs from the reference beyond
tolerance on at least one group. Worst test ratios against a threshold of 30:

| shape | plain | shifted |
|---|---|---|
| (8, 8) | 0.170 | **34.34** |
| (3, 7) | 0.162 | **48.14** |
| (17, 1) | 0.000 | **inf** |
| (7, 129) | 0.150 | 25.64 |
| (16, 32) | 0.183 | 14.45 |

Catastrophic cancellation in the computational formula for variance is textbook. That it is severe
enough to be a **defect** at these shapes was not predicted here, and the gate is what caught it —
recorded as a failed prediction, per the protocol.

Note which column it fails in. On plain input the formulation is fine; it is the **shifted**
metamorphic partner, where the row mean is large and the two terms nearly cancel, that exposes it.
That is a point for the transform, not for any particular arm: the shifted case is in the corpus
because the generator puts it there, and all four arms see it.

It is kept as `FOUND_MUTANTS` rather than deleted. A defect an optimizing agent would plausibly
introduce while believing it was refactoring is exactly the class this corpus is for.

## Threats

- Six kernels, three of them stock, on one GPU, one dtype, one seed.
- Only three tasks, and only one of them (layernorm) produces a zero-centered output — which is
  the mechanism under study. The pooled rate is therefore a rate over a population chosen partly
  because it contains the effect.
- `allclose`'s rtol/atol are the field defaults and were not tuned. The claim is about the default,
  not about the best `np.allclose` can do — a practitioner who sets `atol` per task gets a
  different number, and that tuning cost is precisely what the reference arm removes.
- The variants are correct on *these* shapes. Correctness is established by the gate at ladder
  scale, not proven.
