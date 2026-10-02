# Twenty mutants, three corpora, zero arm differentiation

**Date:** 2026-08-18
**Instance:** Lambda A10, sm_86, torch 2.7.0, triton 3.3.0
**Corpora:** 5 contaminated + 5 blinded + 15 blinded (three per class, five independent agents)

**Headline:** across every mutant that has ever passed the gate — 20 of them, from three
independently authored corpora — **the four oracle arms have never once disagreed.** `allclose`,
the strengthened reference ratio, the declarative property set and the hybrid composition return
identical detection rates on every single kernel.

---

## The result

Eleven of fifteen blinded mutants passed the ladder gate. Every one:

| mutant | allclose | reference | declarative | hybrid | tolerance-free | arms differ? |
|---|---|---|---|---|---|---|
| operator_1 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | no |
| operator_3 | 0.333 | 0.333 | 0.333 | 0.333 | 0.000 | no |
| dtype_1 | 0.778 | 0.778 | 0.778 | 0.778 | 0.000 | no |
| indexing_1 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | no |
| indexing_2 | 0.778 | 0.778 | 0.778 | 0.778 | 0.000 | no |
| indexing_3 | 0.667 | 0.667 | 0.667 | 0.667 | 0.000 | no |
| predicate_1 | 0.333 | 0.333 | 0.333 | 0.333 | 0.000 | no |
| predicate_2 | 0.333 | 0.333 | 0.333 | 0.333 | 0.000 | no |
| predicate_3 | 0.333 | 0.333 | 0.333 | 0.333 | 0.000 | no |
| special_1 | 0.222 | 0.222 | 0.222 | 0.222 | 0.000 | no |
| special_2 | 0.222 | 0.222 | 0.222 | 0.222 | 0.222 | no |

Combined with the two earlier corpora: **20 mutants, 0 disagreements.**

The prediction that failed is worth recording. `indexing_3` reads a row window shifted by one
element into its neighbour — its author noted the output "still sums to 1 per row, just subtly
wrong". It was expected to *pass* the declarative arm's elementwise properties (it is a valid
probability distribution) and be caught only by the reference. It scored 0.667 on all four arms:
`shift_invariance` catches it, because a window straddling two rows sees two different per-row
constants. The metamorphic property did the work — but the reference arm caught it too, so the
arms still tied.

## What this does and does not say

**It does not say the declarative arm is useless.** Tolerance-free detection is real and
non-zero on 3 of 11 mutants — `operator_1`, `indexing_1` and `special_2` are caught with no
numerical tolerance anywhere in the decision.

**It says the declarative arm has no detection advantage.** Every bug it catches, a reference
recompute also catches. On this task, this shape space, and these fault classes, the property set
buys nothing a `torch.allclose` against an eager implementation would not.

The mechanism is not subtle: softmax has a cheap, reliable, trusted reference. When one exists,
a reference comparison is close to an oracle upper bound, and no property set can beat it — it
can only tie.

**Where the declarative arm should still win is where no reference exists.** That is the
translation workstream and NKI, which parent design §5.3 already argues: there is no second
Trainium implementation to differentially test against. The claim these numbers support is not
"properties catch more bugs" but "properties catch the same bugs *without needing a reference*",
which is a different and narrower contribution — and one this corpus cannot test, because it has
a reference by construction.

## The corpora have complementary blind spots

Gate outcomes across both tasks:

| | ladder | at-scale |
|---|---|---|
| admitted | 11/15 | 4/15 |
| rejection rate | 0.27 | 0.73 |

The ladder rejects **shape-specialized** defects: its widest row is 129, so `BLOCK` never exceeds
256 and any `BLOCK >= 1024` branch is dead.

The at-scale task rejects **padding and masking** defects for the opposite reason: every one of
its row widths (1024, 512, 4096, 256, 8192) is already a power of two, so `BLOCK == n_cols`
exactly and there are no partial tiles. `operator_3`, all three `predicate_*` and `special_1`
are dead there.

Neither shape set alone covers the space. A corpus needs **both** wide rows and non-power-of-two
widths, and currently no single task has both.

## The dtype blind spot

Three of the four ladder rejections are dtype-gated. `operator_2`, `dtype_3` and `special_3`
round-trip through `x_ptr.dtype.element_ty`, which is a no-op at float32, and `dtype_2` does not
compile at all. Their authors were writing mixed-precision bugs; every task in the corpus is
float32-only.

`domain.py` already declares `SUPPORTED_DTYPES = ("float16", "float32", "float64")` and no task
uses anything but float32. `data_type_semantics` is **58 of 301 bugs**, the second-largest
subcategory, and the corpus can currently express almost none of it.

## Threats

- **One task.** Everything here is softmax. relu and layernorm are unmeasured, and layernorm is
  where `allclose` is already known to false-positive 5/9 — the one place the arms *are* known to
  differ, though on false positives rather than detection.
- **Three mutants per agent** are less independent than three from three agents.
- **`intended_class` is not a partition.** `operator_1` and `indexing_1` are character-identical
  despite different authoring prompts, so per-class rates double-count mechanisms.
- **No correct-but-different kernels were run**, so no false-positive rate accompanies these.
- The ladder deflation applies: absolute rates understate by the known constant.
