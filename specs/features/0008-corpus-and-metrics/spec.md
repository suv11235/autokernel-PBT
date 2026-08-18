# Feature 0008: The mutation corpus and the metrics

## Problem

Every number this project has produced describes the instrument, not the result. Detection power
— whether a declarative property set catches bugs a reference comparison misses, and at what
false-positive cost — has never been measured, because there is no corpus of broken kernels to
measure it against.

## Scope

1. **A blinded authoring protocol** — the experimental method, version-controlled.
2. **An agent-authored mutation corpus**, one mutant per (task, CPU-reachable fault subcategory),
   in NumPy and Triton variants.
3. **An admission gate** — a candidate enters the corpus only if it is genuinely broken and
   genuinely judgeable; rejections are recorded with their reason.
4. **Correct-but-different kernels** for the false-positive denominator.
5. **The three metrics** — detection rate keyed by case group and split by `tolerance_free`,
   false-positive rate, cases-to-first-failure — computed from the recorded tables alone.
6. **A report** stating the ladder deflation and the intended-class caveat beside the numbers.

## Non-goals

- Metric 4, downstream kernel quality — needs the full agentic loop
- Automatic fault-class verification — the class is intended-by-construction
- Shrinking algorithms; tier-2 properties and `compute-sanitizer` (Phase 3b)
- NKI and Apple Silicon backends
- Adopting `log2(tile) + n_tiles` as the reference arm's normalization

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
