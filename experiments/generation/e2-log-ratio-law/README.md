# e2-log-ratio-law

**Question.** E2 (paper §5) asks whether the completing laws close every family the current
bundle admits, with no new false positive. This directory runs the softmax half: the log-ratio law
(`rows_have_constant_log_ratio`, feature 0014), and the completed softmax bundle built from it.
Layernorm's affine-tie law is not built yet.

**Record:** [`docs/measurements/2026-10-14-e2-log-ratio-law-prereg.md`](../../../docs/measurements/2026-10-14-e2-log-ratio-law-prereg.md),
pre-registered before this script existed, with the results appended below its line.
**Spec:** [`specs/features/0014-log-ratio-law/spec.md`](../../../specs/features/0014-log-ratio-law/spec.md).

**Status: [M] for softmax.**
- **Rejected:** the completed gate rejects softmax(βx) at β = 2, 0 and −1, the 1/n fill, rows rolled
  and columns reversed, each through the law.
- **No new false positive:** the 26 loop candidates, the 21 recorded A10 groups, and 464 CPU cases
  far wider and more masked than the gate draws (worst 1.38 against a budget of 30).
- **Fast-exp:** admitted on the gate's domain (17.2), and rejected on rows that underflow or carry −∞
  masks.

## Run

From this directory, in the project's pinned environment:

```bash
../../../.venv/bin/python e2.py
```

It needs no GPU and takes a few seconds. It reads the gate and the saboteur matrix from
`../autoresearch-pbt/`, the correct and wrong kernels from `../log-ratio-headroom/headroom.py`,
and the recorded runs from `runs/gpu-softmax`, `runs/gpu-tolerance_sweep` and `runs/gpu-scale`.

The completed gate's verdicts on the saboteur matrix are pinned, under both bundles, by
`../autoresearch-pbt/test_gate_saboteurs.py`.

## Files

```
e2.py   kernel sets A-E of the pre-registration, and the tables the record quotes
```
