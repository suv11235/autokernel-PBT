# e2-affine-tie-law

**Question.** E2 (paper §5) asks whether the completing laws close every family the current bundle
admits, with no new false positive. This directory runs the layernorm half: the affine-tie law
(`rows_have_affine_tie`, feature 0015), and the completed layernorm bundle built from it. Softmax's
half is [`../e2-log-ratio-law/`](../e2-log-ratio-law/).

**Record:** [`docs/measurements/2026-11-02-e2-affine-tie-law-prereg.md`](../../../docs/measurements/2026-11-02-e2-affine-tie-law-prereg.md),
pre-registered before this script existed, with the results appended below its line.
**Spec:** [`specs/features/0015-affine-tie-law/spec.md`](../../../specs/features/0015-affine-tie-law/spec.md).

**Status: [M] for layernorm.**
- **Rejected:** the completed gate rejects −LN, reversed columns, sorted rows and rolled rows, each
  through the law.
- **No new false positive:** the 26 loop candidates, the 27 recorded device groups, and 580 CPU cases far
  wider, narrower and more offset than the gate draws (worst 0.853 against a budget of 30).
- **Distributed division:** admitted on the gate's domain (4.72), and rejected at offsets of 1e3 and up,
  where the reference arm rejects it too on all but one input.

## Run

From this directory, in the project's pinned environment:

```bash
../../../.venv/bin/python e2_layernorm.py
```

It needs no GPU and takes a few seconds. It reads the gate and the saboteur matrix from
`../autoresearch-pbt/`, the correct and wrong kernels from `../affine-tie-headroom/tie_headroom.py`, and
the recorded runs from `runs/gpu-layernorm`, `runs/te1-triton-layernorm` and `runs/te1-cuda-layernorm`.

The completed gate's verdicts on the saboteur matrix are pinned, under both bundles, by
`../autoresearch-pbt/test_gate_saboteurs.py`.

## Files

```
e2_layernorm.py   kernel sets A-D of the pre-registration, and the tables the record quotes
```
