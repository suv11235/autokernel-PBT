# Provisional corpus pilot — and why it is a null result

**Date:** 2026-08-17
**Instance:** Lambda A10, sm_86, torch 2.7.0, triton 3.3.0
**Corpus:** 5 Triton mutants (one per CPU-reachable ISSTA subcategory) + 1 correct kernel, softmax
**Status:** **PROVISIONAL. The blinding was violated. No number here may be reported.**

---

## What this is

A pilot to prove the Phase 2b pipeline end to end on real hardware: author → gate → record →
score through four arms → compute rates. It does that. It is **not** a measurement of oracle
strategies, for the reason in §4.

## Results

| kernel | allclose | reference | declarative | hybrid | tol-free (declarative) |
|---|---|---|---|---|---|
| **correct softmax** | 0.000 | 0.000 | 0.000 | 0.000 | — |
| operator_implementation | 0.778 | 0.778 | 0.778 | 0.778 | 0.000 |
| data_type_semantics | 0.778 | 0.778 | 0.778 | 0.778 | 0.000 |
| indexing_and_stride | 0.778 | 0.778 | 0.778 | 0.778 | **0.444** |
| branch_predication | 0.333 | 0.333 | 0.333 | 0.333 | 0.000 |
| special_value_handling | 0.333 | 0.333 | 0.333 | 0.333 | 0.000 |

Detection is the fraction of the 9 case groups on which the arm reaches FAIL.

Gate: **5 of 5 admitted, rejection rate 0.00.**

## 1. What genuinely works

- **Zero false positives.** Every arm passed the correct kernel on all nine groups.
- **The gate ran on device** and admitted every candidate.
- **The driver records and scores on either substrate** through one injected backend.
- **0.778 is the known ladder deflation** — 7 of 9, the other two being the single-column rungs
  where softmax is identically 1.0 for any implementation. It reproduces exactly, on a different
  backend, from an independent corpus.

## 2. The result that matters: no arm differentiates

**All four arms are identical on every mutant.** The declarative arm shows no advantage over
plain `allclose`, and the hybrid arm adds nothing to either.

The project exists to test whether a declarative property set catches bugs a reference comparison
misses. On this corpus it does not — every arm catches every mutant, at exactly the same rate.

## 3. Tolerance-free detection is ~zero

The sharpest claim — *bugs found without a tolerance argument* — scores **0.000 on four of five
classes**. Only `indexing_and_stride` reaches 0.444, presumably because a stride defect can push
values outside `[0, 1]` and trip `values_in_unit_interval`, which is genuinely tolerance-free.

Every other mutant is caught only by tolerance-bearing properties, which is the regime where the
declarative arm has no structural advantage over a reference comparison.

## 4. Why this is a null result and not a finding

**The corpus was authored by the same agent that wrote the property set.**
`docs/protocol/mutant-authoring.md` requires the author to see only the taxonomy quote and the
reference implementation. These mutants were written with full knowledge of
`props/properties.py` and every task contract.

The degenerate outcome is what that contamination predicts. Knowing the properties, the author
wrote defects that break the row-sum invariant — which is a gross numerical error that *any* arm
catches, including `allclose`. A blinded author, given only "operator logic is incorrect or
incomplete after specialization", has no reason to converge on that particular failure and would
plausibly produce defects that separate the arms.

So the honest reading is: **this corpus cannot distinguish the arms, and the most likely
explanation is how it was authored.** The alternative — that these fault classes are genuinely
undifferentiating — cannot be ruled out from here, and that is precisely the ambiguity blinding
exists to remove.

The 0.00 rejection rate points the same way. A blinded author's attempts at a named fault class
should sometimes miss; five for five is what an author who knew the answer would produce.

## 5. What to do

Re-author the corpus with an agent that has not seen the oracle layer, then re-run. The pipeline
is proven, so the second pass costs only authoring and one GPU session.

Until then, the entries above are a smoke test of the machinery, not evidence about oracles.

## Threats

Beyond the blinding violation, which dominates: one task (softmax), one backend, one seed, five
mutants, one per class. Per-class rates rest on a single kernel each. The `branch_predication` and
`special_value_handling` mutants score 0.333 rather than 0.778, and whether that is a property of
those classes or of these two particular kernels cannot be told from n=1.
