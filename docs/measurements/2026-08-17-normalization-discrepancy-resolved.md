# Resolving the normalization discrepancy

**Question:** parent design §5.3 recorded two measurement harnesses disagreeing about which
normalization of the reference arm's test ratio is *flattest* in `n`, with "a consistent 5×"
magnitude gap and "a setup difference — reference construction or residual scaling — [that]
remains unidentified."

**Answer:** the gap is `log2(n_reduction) / log2(n_rowcount)`. One harness divided by the log of
the **wrong axis** — a length that did not vary with the sweep — which makes its `log2n` column
an un-normalized ratio in disguise: flat by construction, and larger by exactly that factor.

**Status:** the choice of `log2(n)` is unchanged and was never at risk. What changes is that
§5.3's open question is closed, and `n=1` can now be compared against `log2(n)` on a single
construction rather than across two that were never measuring the same thing.

## Construction

Stated before any number, per the repo's review standard.

| | |
|---|---|
| kernel | correct softmax, accumulating entirely in float32 |
| inputs | `x64` = N(0,1) float64 draws; `x32 = x64.astype(float32)` is what the kernel sees |
| reference | float64 softmax of **`x32`** — a recompute of the same input the kernel got |
| ratio | `‖cand − ref‖_inf / (‖ref‖_inf · eps32 · norm(n))`, `eps32` fixed |
| shape | `(4, n)`; the reduction is over the last axis |
| statistic | median of 31–41 trials per `n`; drift = `max(median)/min(median)` across `n` |

## What was ruled out

**Reference construction** — the candidate `tolerance.py`'s own docstring names. Recomputing the
reference from the original float64 draws instead of the float32 input does add an
input-quantization term, but it does not flip which normalization is flattest:

| reference | `1` | `log2n` | `sqrtn` | `n` |
|---|---|---|---|---|
| from `x32` (same input) | **1.3×** | 4.1× | 40× | 1820× |
| from `x64` (original draws) | **2.2×** | 2.4× | 23× | 1041× |

`1` is flattest under both. The docstring's claim that this choice "flips which normalization
looks flattest" is not reproduced.

**Residual norm** — the other candidate §5.3 names. Swapping the inf-norm for a 2-norm, a mean
or a sum moves `log2n`'s drift from 4.1× to ~2.5×, but never makes it flat and never makes it
flattest:

| residual | `log2n` drift | `1` drift |
|---|---|---|
| inf (max) | 4.1× | 1.2× |
| 2-norm | 2.8× | 1.7× |
| mean\|·\| | 2.5× | 1.9× |
| sum\|·\| | 2.5× | 1.9× |

## What reproduces it

Taking `n` from the row count (fixed at 4) rather than the reduction length. `log2(4) = 2` is
then a constant divisor, so the "log2n" column is the un-normalized ratio halved:

| n | `n` = reduction length (correct) | `n` = row count (wrong axis) | multiplier |
|---|---|---|---|
| 64 | 0.0839 | 0.2517 | 3.00× |
| 256 | 0.0646 | 0.2586 | 4.00× |
| 1024 | 0.0529 | 0.2647 | 5.00× |
| 4096 | 0.0360 | 0.2161 | 6.00× |
| 16384 | 0.0359 | 0.2515 | 7.00× |
| **drift** | **2.33×** | **1.23×** | |

Two things identify this as the cause rather than merely a construction that happens to fit.

**The multiplier is exact.** 3, 4, 5, 6, 7 is precisely `log2(n)/log2(4)` at each point — not an
approximate fit but the analytic ratio of the two divisors.

**Its mean over the swept range is 5**, which is §5.3's "consistent 5×" to the digit. The gap
was described as consistent because it was averaged over a sweep across which it in fact varies
from 3 to 7.

And the signature matches: §5.3 reports harness A seeing `log2n` "flat", 0.408 → 0.368. The
wrong-axis construction gives 0.252 → 0.252, flat at 1.23×. The residual ~1.6× magnitude
difference is accounted for by a different row count (`log2(rows)` is the divisor).

## Caveat, stated plainly

Harness A no longer exists, so this is not a confession recovered from it — it is a construction
that reproduces its reported signature, with an analytically exact multiplier and a mean
matching the reported gap. That is strong evidence, not proof.

## Consequences

**`log2(n)` stands.** It was chosen because it is the textbook bound for pairwise summation,
which is what these backends do, rather than as a fit to either dataset. Nothing here disturbs
that, and the correct construction still shows linear `n` drifting 1820× — the over-normalization
that would make the reference arm lose to the `allclose` baseline it exists to beat.

**`n=1` deserves the revisit §5.3 flagged.** On the correct construction `1` is flatter than
`log2n` (1.2–1.3× against 2.3–4.1×) under every reference and residual variant measured. §5.3
explicitly said the question "would matter if `n=1` were revisited", and the reason it was not
revisited was that the two harnesses disagreed. They no longer do.

That is a real open question, not a recommendation: flatness is one criterion and the
detection floor is another, and §5.3's own table shows `1` is *stricter* than `log2n` — which
raises the false-positive risk that the layernorm/`allclose` measurement showed is not
hypothetical. Deciding it needs the mutation corpus, so it belongs to Phase 2b.

**No code change.** `residual_ratio`'s `n=` contract is already correct and already tested
(`test_explicit_n_overrides_the_last_axis`, `test_default_n_still_depends_on_memory_layout`),
and `CLAUDE.md` already carries the contract that made this diagnosable: *"the default is the
last-axis length, which is wrong for an already-reduced array."* The defect was in a
measurement harness, not in the library.

## Threats

- One task (softmax), one dtype (float32), one kernel, `(4, n)` shapes. The drift figures are
  specific to this corpus.
- Medians of 31–41 trials. Enough to separate 1.2× from 4.1×, not enough to split 1.2× from 1.3×.
- The `n=1` comparison here measures flatness only. It says nothing about detection power, which
  is what would actually decide the question.
