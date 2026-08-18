# The hybrid reduction tree, and what it settles

**Date:** 2026-08-17
**Instance:** Lambda A10, sm_86, CUDA 12.8, torch 2.7.0, triton 3.3.0
**Closes:** the `n=1` question left open by design §5.3 and reopened by
`2026-08-17-normalization-discrepancy-resolved.md`

**Headline:** `n=1` looked flatter than `log2(n)` only because every measurement so far used a
**single-tile** reduction. In the multi-tile regime — which is what real kernels do — the
ordering **reverses**, and `log2(n)` wins. The choice made on textbook grounds survives on
empirical ones too.

Separately: the tightest bound measured is `log2(tile) + n_tiles`, which is flatter than
`log2(n)` by 1.7×, and needs the tile width — a launch property the telemetry schema already
records.

---

## Why a new kernel was needed

Every kernel measured before this held a whole row in one tile. That reduces as a **single
balanced tree** of depth `log2(n)` — precisely the pairwise bound the reference arm's
normalization assumes. The tolerance result was therefore scoped to the case that most flatters
it, which the at-scale write-up records as a limitation.

`_softmax_wide_kernel` loops over tiles: a balanced tree of depth `log2(TILE)` *within* each
tile, then **sequential** accumulation across `n/TILE` tiles. Sequential accumulation is the
regime whose error bound is linear, so a large tile count should make error outgrow `log2(n)`.

Three passes (max, sum, write) rather than a streaming formulation, deliberately: rescaling a
running sum as the max moves is a different algorithm with different error behaviour, and mixing
it in would confound the tree-shape measurement.

## Construction

Stated before the numbers, per the repo standard.

| | |
|---|---|
| kernel | correct float32 Triton softmax, three passes over tiles |
| reference | float64 softmax of the **same float32 input** the kernel received |
| ratio | `‖cand − ref‖_inf / (‖ref‖_inf · eps32 · norm)` |
| shapes | 4 rows × {4096, 16384, 65536, 262144} columns |
| tiles | 1024 and 4096, so tile count spans 1 → 256 and is separable from `n` |

## Result

| n_cols | tile | tiles | `log2(n)` | `1` | `n_tiles` | `log2(tile)+n_tiles` |
|---|---|---|---|---|---|---|
| 4096 | 1024 | 4 | 0.0425 | 0.5099 | 0.1275 | 0.0364 |
| 16384 | 1024 | 16 | 0.0334 | 0.4674 | 0.0292 | 0.0180 |
| 65536 | 1024 | 64 | 0.0582 | 0.9311 | 0.0145 | 0.0126 |
| 262144 | 1024 | 256 | 0.1985 | 3.5722 | 0.0140 | 0.0134 |
| 4096 | 4096 | 1 | 0.0425 | 0.5099 | 0.5099 | 0.0392 |
| 16384 | 4096 | 4 | 0.0501 | 0.7011 | 0.1753 | 0.0438 |
| 65536 | 4096 | 16 | 0.0582 | 0.9311 | 0.0582 | 0.0333 |
| 262144 | 4096 | 64 | 0.0794 | 1.4289 | 0.0223 | 0.0188 |
| **drift** | | | **5.9×** | **7.6×** | **36.5×** | **3.5×** |

## 1. The `n=1` question is settled, against `n=1`

Design §5.3 flagged `n=1` as deserving a revisit, and both the CPU re-measurement and the first
GPU run found it flatter than `log2(n)` — 1.2–1.3× and 2.1× respectively, against 4.1× and 5.9×.
Two independent substrates agreeing made it look robust.

**Both were single-tile measurements.** Here, with the tile count varying from 1 to 256, `n=1`
drifts **7.6×** and `log2(n)` **5.9×**. The ordering reverses, and the `n=1` column is starkly
U-shaped — 0.51, 0.47, 0.93, **3.57** — climbing steeply once sequential accumulation dominates.
That climb is the linear term, and no constant normalization can absorb it.

So the apparent flatness of `n=1` was an artifact of a regime where the reduction never left one
tile. `log2(n)` was chosen because it is the textbook bound for pairwise summation rather than a
fit to any dataset; that reasoning is now supported rather than merely defended.

## 2. The tightest bound is tile-aware

`log2(tile) + n_tiles` drifts **3.5×**, against `log2(n)`'s 5.9× — a 1.7× improvement, and it is
the only normalization tested that is flat in *both* `n` and tile width simultaneously. That is
what the hybrid tree predicts analytically: a balanced tree inside the tile, a linear term across
tiles.

**Adopting it would make the tolerance depend on a launch property, not just the task.** The
reference arm currently takes `n` alone. The tile width is a backend and launch-config fact —
and it is one the telemetry schema already records, in `constexprs`. The schema built to serve
tier-2 fault classes turns out to be what a tile-aware tolerance would need, which is a second
justification it was not designed for.

Not adopted here. It is a change to the reference arm's definition, it should not be made on one
kernel, one GPU and one dtype, and every number recorded so far uses `log2(n)`.

## 3. `log2(n)` remains safe in practice

At the worst measured point — 256 tiles, `n = 262144` — the correct-kernel ratio under `log2(n)`
is **0.1985** against a threshold of 30, still ~150× of headroom. `log2(n)` *under*-normalizes
relative to the hybrid bound, which makes it **stricter**, not blind: the failure direction is
false positives on enormous reductions, not missed bugs. Extrapolating the trend, the threshold
would not be reached until tile counts far beyond anything realistic.

So there is no urgency. The finding is that a better bound exists and what it costs, not that the
current one is broken.

## Threats

- One kernel shape (three-pass softmax), one GPU, one dtype, one seed.
- Tile counts 1–256. The linear term is visible but not dominant; a regime with thousands of
  tiles would test the hybrid bound far harder.
- The three-pass formulation is not what a performance-tuned kernel would use — streaming softmax
  is — and streaming has different error behaviour that is untested here.
- `log2(tile) + n_tiles` is fitted to this data in the sense that it was *predicted* first and
  then measured, but on one dataset. It should be re-derived on another kernel before being
  trusted as a definition.
