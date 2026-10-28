# Measurements

Dated, append-only records of every measurement this project has made. A record states its
construction before its numbers, says which environment it ran in, and is never edited to change a
result. Corrections are made in a later record that supersedes it. A pre-registration
(`*-prereg.md`) is written before its experiment runs, and results are appended below its line.
The paper quotes these records through [`paper/numbers.tex`](../../paper/numbers.tex), and
[`../findings.md`](../findings.md) ranks what they add up to.

Experiment directories moved under `experiments/{generation,verification}/` on 2026-10-11, and
the paths inside the records were updated to match. Nothing else in a record was changed.

## Verification layer: property-based oracles

| Record | Headline |
|---|---|
| [2026-08-16-allclose-layernorm-false-positives](./2026-08-16-allclose-layernorm-false-positives.md) | `allclose` flags a correct layernorm on 5/9 groups; the other arms 0/9 |
| [2026-08-16-layernorm-authoring-cost](./2026-08-16-layernorm-authoring-cost.md) | Authoring cost, `n = 1`, pre-registered |
| [2026-08-17-normalization-discrepancy-resolved](./2026-08-17-normalization-discrepancy-resolved.md) | Two harnesses disagreed because one took log of the wrong axis |
| [2026-08-17-hybrid-reduction-tree](./2026-08-17-hybrid-reduction-tree.md) | Multi-tile: `log2 n` beats `n = 1`; tile-aware bound is tightest |
| [2026-08-17-provisional-corpus-pilot](./2026-08-17-provisional-corpus-pilot.md) | Null result; blinding violated, so no number is reportable |
| [2026-08-17-blinded-corpus-pilot](./2026-08-17-blinded-corpus-pilot.md) | Blinding changed the corpus and exposed two coverage holes |
| [2026-08-18-arm-differentiation-null-result](./2026-08-18-arm-differentiation-null-result.md) | 20 mutants, 3 corpora, 0 arm disagreements |
| [2026-08-19-false-positive-rate](./2026-08-19-false-positive-rate.md) | `allclose` 10/54 on correct kernels; the other arms 0/54 |

## Translation

| Record | Headline |
|---|---|
| [2026-10-12-te1-triton-to-cuda-prereg](./2026-10-12-te1-triton-to-cuda-prereg.md) | Pre-registered: the known Triton ladder mapped to CUDA on one device; 8 predictions |

## Cross-backend transfer and hardware (evidence for translation)

| Record | Headline |
|---|---|
| [2026-08-17-first-hardware-run](./2026-08-17-first-hardware-run.md) | Tier-1 laws transfer NumPy → Triton unchanged (52 rows, A10) |
| [2026-08-17-at-scale-hardware-run](./2026-08-17-at-scale-hardware-run.md) | 29.4 M elements at 78% of peak bandwidth; a tile-tail defect caught |
| [2026-08-17-sanitizer-viability-probe](./2026-08-17-sanitizer-viability-probe.md) | compute-sanitizer is blind under PyTorch's caching allocator |

## Generation

| Record | Headline |
|---|---|
| [2026-08-18-autoresearch-loop-pbt-gate](./2026-08-18-autoresearch-loop-pbt-gate.md) | Property-gated loop: −21.2% val_bpb, 0 gate rejections, 6 saboteurs caught |
| [2026-10-01-softmax-temperature-blind-spot](./2026-10-01-softmax-temperature-blind-spot.md) | The decisive arm admits softmax(βx), 1/n and softmax(−x) |
| [2026-10-14-log-ratio-headroom-prereg](./2026-10-14-log-ratio-headroom-prereg.md) | One log-ratio threshold (30 eps) separates every correct softmax, A10 runs included, from every wrong one; device headroom shrinks with row range |
| [2026-10-14-e2-log-ratio-law-prereg](./2026-10-14-e2-log-ratio-law-prereg.md) | E2, softmax: the log-ratio law rejects every family the gate admitted, with no new false positive; fast-exp admitted only on the gate's domain |

## Formal verification

| Record | Headline |
|---|---|
| [2026-10-01-softmax-bundle-mechanized](./2026-10-01-softmax-bundle-mechanized.md) | Proposition 1(a,b) in Lean; natural `exp` axioms are temperature-blind |
| [2026-10-02-why3-real-theories-symmetry](./2026-10-02-why3-real-theories-symmetry.md) | Why3's `ExpLog` admits exp(βx); `Square` is categorical |
| [2026-10-02-prover-in-the-toolkit-prereg](./2026-10-02-prover-in-the-toolkit-prereg.md) | Pre-registration; ran, results not yet appended |
