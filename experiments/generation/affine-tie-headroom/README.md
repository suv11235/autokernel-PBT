# affine-tie-headroom

**Question.** The affine-tie law (in each row, y is an increasing affine function of x) is the one
law Proposition 1 says completes the layernorm bundle. Can a single float threshold on its
statistic, with a positive slope, accept every correct layernorm we hold, the recorded device runs
included, and reject every wrong family the gate admits today? This is design input for E2
(complete the bundle). Nothing here changes a bundle, an arm or the gate.

**Record:** [`docs/measurements/2026-11-02-affine-tie-headroom-prereg.md`](../../../docs/measurements/2026-11-02-affine-tie-headroom-prereg.md),
pre-registered before the script existed, with the results appended below its line.

**Status: [M].** At the registered threshold of 30 every correct case is accepted, CPU at most 0.673
and devices at most 0.708. Every wrong output is rejected: −LN by its slope alone, and the others
by a ratio of at least 6.28e6. No range term is needed on the rows recorded.

## Run

From this directory, in the project's pinned environment:

```bash
../../../.venv/bin/python tie_headroom.py
```

It reads the gate's cases through `../autoresearch-pbt/pbt_gate.py`, the loop's kernels from
`../autoresearch-pbt/`, and the recorded runs from `runs/gpu-layernorm`,
`runs/te1-triton-layernorm` and `runs/te1-cuda-layernorm`. It needs no GPU and takes a few seconds.

## Files

```
tie_headroom.py   the statistic, the kernel list, and the tables the record quotes
```
