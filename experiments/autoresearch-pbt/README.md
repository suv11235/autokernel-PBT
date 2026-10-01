# autoresearch-pbt

An autoresearch-style optimization loop (karpathy/autoresearch), transplanted to
CPU/NumPy, with this repository's property oracles as the correctness gate.

```
prepare.py   fixed: data, byte tokenizer, fixed eval windows, val_bpb, budget
train.py     the agent edits this: NumPy GPT, AdamW, training loop
pbt_gate.py  the four oracle arms over train.py's softmax / layernorm
runner.py    the outer loop: gate -> measure -> keep/revert, ledger in results.tsv
program.md   agent instructions
best/        the current best train.py
results.tsv  the ledger
saboteurs.py the gate's saboteur matrix: kernels, and exactly which arms catch each
test_gate_saboteurs.py   pins that matrix, one assertion per gate defect
```

Loop:

```bash
../../.venv/bin/python runner.py baseline
# an agent edits train.py, then:
../../.venv/bin/python runner.py evaluate "note about the change"
../../.venv/bin/python runner.py rigor 3
```

The gate runs **before** the metric, and its decisive arm is the **declarative**
one, not `allclose` — `allclose` false-positives on the baseline's own correct
layernorm (5 of 9 groups), so a loop gated on it cannot accept its own starting
point.

That choice has a measured cost: the declarative softmax bundle holds for
softmax(beta * x) at every beta, so the gate **admits** softmax(2x), the uniform 1/n
and softmax(-x), all of which the other three arms reject on every group they can
judge (`docs/measurements/2026-10-01-softmax-temperature-blind-spot.md`). The matrix
records it as a baseline — the decisive arm and the bundle are unchanged on purpose:

```bash
../../.venv/bin/python saboteurs.py                                # print the matrix
cd ../.. && .venv/bin/python -m pytest experiments/autoresearch-pbt -q   # pin it
```

The tests are outside the main suite's `testpaths`, so `pytest -m "not gpu"` does not
run them; run them explicitly.
