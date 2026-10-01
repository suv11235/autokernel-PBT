# autoresearch-pbt

An autoresearch-style optimization loop (karpathy/autoresearch), transplanted to
CPU/NumPy, with this repository's property oracles as the correctness gate.

```
prepare.py        fixed: data, byte tokenizer, fixed eval windows, val_bpb, budget
train.py          the agent edits this: NumPy GPT, AdamW, training loop
pbt_gate.py       the four oracle arms over train.py's softmax / layernorm
runner.py         the outer loop: gate -> measure -> keep/revert, ledger in results.tsv
rigor_sweep.py    multi-seed re-measurement of named candidates
program.md        agent instructions
best/             the current best train.py
candidates/       every measured candidate, verbatim (do not reformat: results.tsv points at them)
results.tsv       the ledger
rigor.json        the recorded multi-seed sweep (rigor_sweep.py refuses to overwrite it)
requirements.txt  this loop's own environment
```

## Environment

The loop runs in its **own** venv, not the project's. The project pins `numpy<2`
for torch interop, and NumPy 1.x wheels link OpenBLAS on macOS rather than
Accelerate. Under a fixed wall-clock budget that is not a detail: measured on the
winning `train.py` (M4 Pro, seed 0), NumPy 2.5.3 / Accelerate reached 1633 steps
and val_bpb 2.221 — matching the ledger — while NumPy 1.26.4 / OpenBLAS reached
265 steps and 3.293. The gate's verdicts are identical under both.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

`autokernel_pbt` is imported from `../../src` by `pbt_gate.py`, not installed, so
the project's `numpy<2` pin never enters this environment. `runner.py` and
`rigor_sweep.py` use `.venv/bin/python` here and warn if they fall back to the
project's.

## Loop

```bash
.venv/bin/python runner.py baseline
# an agent edits train.py, then:
.venv/bin/python runner.py evaluate "note about the change"
.venv/bin/python runner.py rigor 3
```

The gate runs **before** the metric, and its decisive arm is the **declarative**
one, not `allclose` — `allclose` false-positives on the baseline's own correct
layernorm (5 of 9 groups), so a loop gated on it cannot accept its own starting
point.
