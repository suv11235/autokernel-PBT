# An autoresearch loop, gated on properties: 26 candidates, 0 gate rejections, one blind spot

**Date:** 2026-08-18
**Instance:** Apple M4 Pro, 48 GB, macOS 26.6.1, NumPy 2.5.2 on Accelerate BLAS, CPU only
**Harness:** `experiments/autoresearch-pbt/` — karpathy/autoresearch's loop (fixed wall-clock
budget, keep/revert on one metric) transplanted to a NumPy byte-level GPT, with this project's
oracle arms as the correctness gate
**Why CPU:** no GPU was reachable for this session (`ssh lambda` timed out; no local torch or
triton). The oracle layer's NumPy backend is first-class, so the gate is the real one; only the
kernels under test are NumPy rather than Triton.

**Headline.** The loop produced a **−21.2% val_bpb** improvement (2.8287 → 2.2280, 3 seeds each,
~47 pooled sd) at **+27.7% throughput**, over 26 measured candidates in 6 rounds. The property
gate ran on every one and **rejected none**. Separately, a saboteur matrix confirms the gate is
not inert: 6 plausible "optimizations" were caught, each of which is genuinely faster. And one
candidate exposed a **structural blind spot in the softmax property set** that no mutation corpus
in this project could have found.

---

## 1. What the loop achieved

Paired multi-seed sweep, all four run back-to-back so machine load is shared:

| candidate | mean val_bpb | sd | steps | tok/s |
|---|---|---|---|---|
| baseline (plain NumPy GPT) | 2.8287 | 0.0104 | 1174 | 40072 |
| r5_muon_lr | 2.2447 | 0.0112 | 1474 | 50312 |
| r6_wd_rescaled | 2.2472 | 0.0196 | 1547 | 52806 |
| **r6_best_nolrmult (winner)** | **2.2280** | 0.0147 | 1499 | 51166 |

The three leaders are separated by less than a pooled sd; the winner was taken on best mean **and
parsimony** — it is the one with neither the group-LR multiplier nor the weight decay. The gain
over baseline is not close to noise.

The changes that earned it, in order of size: **Muon** on the 16 two-dimensional hidden matrices
(Newton-Schulz ×5, multiplier 0.10) at −0.14; **squared-ReLU MLP** replacing tanh-GELU at
−0.09 through 1.21× cheaper steps; **fused QKV plus flat `(B*T, C)` activations plus a cached
causal mask** at −0.13 through 1.26× cheaper steps; **budget-fraction warmup+cosine at peak
3e-3** at −0.22; **`n_head` 4→2 with `n_embd` 128→160** at −0.10.

**A caveat on the ledger's absolute numbers.** `results.tsv` rows were taken across several hours
under varying machine load. The baseline re-ran at 2.824 (1170 steps) against its ledger row of
2.779 (1193 steps) — same code, same seed. Cross-round absolutes therefore carry step-count
noise of order ±0.05; only the paired sweep above should be quoted.

## 2. The gate rejected nothing, and that is a result about the loop, not the gate

26 candidates, 4 arms, 2 tasks each: **0 rejections.** The reason is not that the gate is weak.
`program.md` names the two cheapest reward hacks and forbids them by name, with their measured
speedups attached — and the agents proposing candidates obeyed. One agent was explicitly asked to
build an approximate-exp softmax and did so honestly rather than hiding it.

So the in-loop rejection count measures the **agents' compliance**, not the gate's power. The
gate's power has to be measured against deliberate saboteurs, which is exactly the mutation-corpus
protocol this project already uses.

## 3. The saboteur matrix: the gate has teeth, and every saboteur is genuinely faster

Six mutants, each a real optimization an agent would plausibly propose. Groups failed of 9:

| task | saboteur | allclose | reference | declarative | hybrid |
|---|---|---|---|---|---|
| softmax | no max-subtraction | 2/9 | 2/9 | 2/9 | 2/9 |
| softmax | float16 accumulator | 7/9 | 7/9 | 7/9 | 7/9 |
| softmax | drops last column | 7/9 | 7/9 | 7/9 | 7/9 |
| layernorm | no centering | 9/9 | 9/9 | 9/9 | 9/9 |
| layernorm | no divide | 7/9 | 7/9 | 7/9 | 7/9 |
| layernorm | sample not population variance | 7/9 | 7/9 | 7/9 | 7/9 |

Measured speed on (512, 4096) float32, ms/call: softmax 3.334 → **2.999** without
max-subtraction (1.11×); layernorm 1.040 → **0.590** without centering (1.76×) → **0.434**
without the divide (2.40×).

**This is the reward-hacking asymmetry, quantified.** In autoresearch the metric is held-out loss,
so search pressure and correctness pressure point the same way — a wrong model just scores worse.
In kernel optimization they point *opposite*: every one of these six is faster, and each produces
plausible output that would train to a plausible loss. The gate is not an auxiliary check bolted
onto the loop; it is the half of the fitness function that keeps the other half meaningful.

**All four arms tied on all six.** That extends the 20-mutant, 3-corpus null result
(`2026-08-18-arm-differentiation-null-result.md`) by six more mechanisms on a second backend, with
the same finding: where a cheap trusted reference exists, the declarative arm has no detection
advantage.

## 4. `allclose` is loop-fatal, not merely false-positive-prone

Every one of the 26 candidates carries the same gate detail: `layernorm/allclose=5/9` while
reference, declarative and hybrid all read 0/9. This reproduces
`2026-08-16-allclose-layernorm-false-positives.md` — but the loop turns a footnote into a
structural claim.

**A loop gated on `allclose` could not have accepted its own starting point.** The baseline's
layernorm is correct and `allclose` rejects it on 5 of 9 groups, so iteration 0 never happens and
`val_bpb` is never computed. Every candidate in this ledger is unreachable.

Sharper still, comparing §3 against §4: `allclose` flags a **correct** layernorm on 5/9 groups and
a **broken** one on 7/9. Those distributions overlap. As a gate on this task `allclose` has almost
no discriminating power at all — a fact invisible when correct and broken kernels are studied in
separate documents.

This is the "oracle choice determines the reachable frontier" claim, and it is now measured rather
than argued.

## 5. The blind spot: the property set cannot price `exp`

One agent built two softmax variants and pre-registered its prediction before measuring.

`candidates/r3_softmax_fastexp.py` replaces `np.exp` with `2**(x·log2e)` — exponent bits assembled
by integer shift, degree-4 minimax polynomial on the fraction. Measured **max relative error
4.49e-06 vs `np.exp` on [−40, 0]**, roughly 40× less accurate per element than the exact kernel,
whose deviation is one float32 ulp (1.19e-07).

**All four arms passed it.** Predicted, and confirmed:

| property | verdict | why |
|---|---|---|
| FINITE_OUTPUT | PASS | the clamp keeps `-inf` out of the integer path |
| UNIT_INTERVAL | PASS | each value ≤ the sum of the *same* values |
| ROW_SUMS | PASS | **the approximation appears in numerator and denominator and cancels exactly** |
| SHIFT_INVARIANCE | PASS | compares two *equally approximate* outputs, not a distance from truth |

Worst `|rowsum − 1|` anywhere, including near-overflow-shifted partners: exact kernel 1.37e-07,
fast-exp kernel **1.66e-07**. In the reference arm's own units, where the threshold is 30, the
fast-exp kernel's shift ratio is 8.3–10.5 against the exact kernel's 9.4–15.9 — *indistinguishable*,
because the discrepancy is dominated by float32 rounding of `x + c` rather than by the polynomial.

**Three of the four softmax criteria are self-normalizing structural invariants, and the fourth is
a self-comparison. None of them can see the accuracy of `exp`.** A softmax that is 40× further
from the true function than a correct one passes with identical margins, and so does the reference
arm at ratio ~14, and so does `allclose` at rtol 1e-5 against a 4.65e-06 residual.

This matters for the project's central claims, in two ways:

- It is a **false-negative class the mutation corpus cannot express.** All 20 corpus mutants are
  semantic faults — wrong operator, wrong index, wrong predicate, wrong dtype. An
  *approximation-quality* fault is a different axis, and softmax's contract is blind along it by
  construction. `SHIFT_INVARIANCE` is a metamorphic relation and it is blind too.
- It bears directly on the **tolerance-free claim**. Tolerance-free properties are attractive
  precisely because they need no error model. This shows the flip side: a property that needs no
  tolerance also cannot detect an error that lives *in* the tolerance. Both of the two currently
  implemented tolerance-free properties (finiteness, unit interval) are in that category.

What would have caught it: a property comparing against a *wider-precision recompute of the same
algorithm* — not a different reference, the same one at float64 — bounded by the dtype's rounding
budget rather than by a structural invariant. That is a fifth arm, and the recorded corpus can
score it offline.

## 6. Negatives worth not re-spending

- **Approximating `exp` is slower in NumPy, not faster.** 0.85× at (2048,128), 0.84× at
  (512,4096). Every ufunc is a full memory round-trip, so a 13-pass polynomial cannot beat one
  fused `np.exp` call, which already uses the same exponent trick in registers. Measured
  alternatives: degree-3 poly 0.88×, integer-space clamp 0.81×, `np.ldexp` 0.47×. **The candidate
  was stopped by its speed, not by the gate** — which is the uncomfortable part of §5.
- **Exact softmax micro-optimization buys nothing.** Reciprocal-multiply instead of divide saves
  0.7%, below noise: `exp` is 72% of the call and everything else is memory-bound.
- **Blocked causal attention is worth ~1.005×**, not the ~7% the exp-count argument predicted. The
  backward's per-block overhead nearly cancels the saving. Verified bit-identical forward and
  gradients matching to 6.3e-7, so it is free but pointless.
- **Weight tying** (`wte` ↔ head) lost on its own and lost composed (2.398 vs 2.349).
- **Dropout** loses from the wrong side: it closes the train/val gap by making train worse
  (train +0.08, val +0.06) and costs 8% of the steps.
- **Label smoothing** (0.02, 0.05) and **input byte corruption** (0.02, 0.06) both lost. Every
  noise-injection regularizer lost; only shrinkage won, and only in-process.
- **Capacity stopped binding.** `r4_cap` (1.95M params, n_layer 3, n_embd 224) landed at 2.3859
  against r3_all's 2.3852 — its author had pre-registered exactly that as the falsifier.
- **`batch_size` buys nothing** on this BLAS: tok/s is flat from 8 to 64, so Accelerate never
  starts threading and cost is linear in B.

## 7. Two results that did not reproduce on the runner

Both were measured 3/3 in-process by their authors and both flipped sign when the runner measured
them. This is the single most important methodological note in this document.

| change | author's in-process result | runner |
|---|---|---|
| per-group LR multiplier (`wte`/`wlm` at 0.25×) on the Muon model | expected to compose | **2.2634 vs 2.2451, worse** |
| decoupled weight decay on the Muon branch, wd 2.0 | −0.0245, 3 of 3 seeds | **+0.0135, worse** |

The weight-decay author pre-registered the risk exactly: wall-clock schedule jitter is worth ±0.01
on its own, and their machine was running ~20% slow, so their pilots did 1249 steps rather than
~1550. Under a wall-clock budget an in-process pilot is a *different experiment* from a harness
run, because the step count — and therefore the position on the cosine — differs.

The weight-decay work is still worth keeping for one reason unrelated to its verdict: it found
that `hybrid_step` dispatches to Muon and `continue`s **before the decay line**, so decay reached
only the AdamW group — **7.7% of parameters**. The earlier "weight decay is inert" negative was
measured on a model where decay touched almost nothing, and was silently inherited as if it were a
statement about the whole model. A negative result carries the configuration it was measured under.

## 8. Threats

- **One task family, one backend.** Everything here is a NumPy softmax and layernorm. The Triton
  and CUDA paths are untouched, and the gate's decisive arm was chosen (declarative) rather than
  measured against the alternatives on this corpus.
- **The gate's decisive arm is a design choice with a measured justification, not a result.**
  `allclose` was excluded because §4 shows it cannot pass a correct baseline. That is sound but it
  means this loop cannot be read as a comparison of arms as gates — only the excluded one was
  shown to be unusable.
- **26 candidates on one seed each, then 4 on three seeds.** The exploration was run at a noise
  level (±0.05 across rounds) larger than several of the differences it acted on. The winner is
  robust; the ordering of rounds 5 and 6 is not.
- **`n = 3` seeds** for the headline. The gain is ~47 pooled sd so it is not at risk, but the
  separation between the top three candidates is not resolved at this n.
- **The saboteurs are author-chosen, not blinded.** They were written by the same person who knew
  the property set — the opposite of `docs/protocol/mutant-authoring.md`. They should be treated as
  a demonstration that the gate can fail, not as a detection rate.
- **`train.py`'s gradients were verified by finite differences at a tiny config** (worst relative
  error 1e-8 to 6e-6 across every candidate), not exhaustively. A gradient bug that only appears at
  the real shape would not have been caught.

## 9. What this says about the parent design

§7 metric 4 — "downstream kernel quality: end-to-end effect on generated kernel speed. Requires
the full loop; deferred to a later phase" — is now measurable, and cheaply, on CPU. The loop needed
one new module (`pbt_gate.py`, 140 lines, a `driver.run_task` without the persistence) plus a
keep/revert ledger. Everything else was reused unchanged: `Generator`, the case groups, the four
arms, the contracts, `NumpyBackend`.

The three findings the mutation corpus could not have produced, in order of value:

1. **§5, the approximation blind spot.** A false-negative class orthogonal to the entire 301-bug
   taxonomy, found because an optimizer naturally proposes approximations and a mutation corpus
   never does.
2. **§4, `allclose` as an unusable gate.** The false-positive rate was already known; that it makes
   iteration 0 unreachable is new, and it is the concrete form of "oracle choice determines the
   reachable frontier".
3. **§7, that a negative result inherits its configuration.** "Weight decay is inert" survived
   three rounds after the branch it was measured on stopped covering 92% of the parameters.

Threat #4 of the null-result document — *"No correct-but-different kernels were run, so no
false-positive rate accompanies these"* — is the gap this harness closes by construction, and it
is not yet exhausted: every one of the 26 candidates is a correct-but-different kernel, and the
false-positive rate over them is **0/26 for reference, declarative and hybrid, and 26/26 for
`allclose` on layernorm.**
