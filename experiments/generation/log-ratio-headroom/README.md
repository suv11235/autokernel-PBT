# log-ratio-headroom

**Question.** The log-ratio law (log y − x constant along each row) is the one law Proposition 1
says completes the softmax bundle. Can a single float threshold on its statistic accept every
correct softmax we hold, the recorded A10 runs included, and reject every wrong family the gate
admits today? This is design input for E2 (complete the bundle). Nothing here changes a bundle,
an arm or the gate.

**Record:** [`docs/measurements/2026-10-14-log-ratio-headroom-prereg.md`](../../../docs/measurements/2026-10-14-log-ratio-headroom-prereg.md),
pre-registered before the script existed, with the results appended below its line.

**Status: [M].** At the registered threshold of 30 eps every correct case is accepted, CPU at most
7.29 and A10 at most 12.3, and every wrong output is rejected, the weakest at 2.67e7. The device
spread grows with each row's input range (r = 0.92), so a range-aware bound is the float form E2
should test first.

## Run

From this directory, in the project's pinned environment:

```bash
../../../.venv/bin/python headroom.py
```

It reads the gate's cases through `../autoresearch-pbt/pbt_gate.py`, the loop's kernels from
`../autoresearch-pbt/`, and the recorded runs from `runs/gpu-softmax`, `runs/gpu-tolerance_sweep`
and `runs/gpu-scale`. It needs no GPU and takes a few seconds.

## Files

```
headroom.py   the statistic, the kernel list, and the tables the record quotes
```
