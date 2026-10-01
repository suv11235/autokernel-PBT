# program.md — autoresearch loop, PBT-gated

You are one iteration of an autonomous research loop over a NumPy GPT trained on
byte-level Shakespeare. Your job: make **one** well-reasoned change to `train.py`
that lowers `val_bpb`.

## The rules

- Training runs for a **fixed 60-second wall-clock budget**, not a fixed step count.
  Anything that makes a step cheaper converts directly into more steps, and more
  steps into lower loss. **Speed *is* accuracy here.**
- The metric is `val_bpb` (validation bits per byte, lower is better), measured on a
  fixed set of validation windows. Uniform-random is exactly 8.0.
- `prepare.py` is **fixed**. Do not edit it. `BLOCK_SIZE = 128` is the eval context
  length; the model must accept `T = 128`.
- Edit **only `train.py`**. Everything in it is fair game: architecture, width,
  depth, optimizer, learning-rate schedule, batch size, memory layout, kernels.

## The property gate — read this before you optimize a kernel

Before `val_bpb` is measured at all, `train.py`'s module-level `softmax(x)` and
`layernorm(x)` are run through this repository's property oracles over nine
generated shapes, including degenerate `(1,1)` and `(17,1)` rungs and inputs shifted
by a near-overflow constant. If a property fails, your candidate is **rejected
without its loss ever being computed** and the file is reverted.

The gate exists because the two cheapest "optimizations" available to you are both
wrong:

- dropping the max-subtraction in `softmax` is ~11% faster and fails
  `shift_invariance`;
- dropping the centering or the divide in `layernorm` is 43–58% faster and fails
  `rows_have_zero_mean` / `rows_have_unit_variance`.

Both were measured. Do not propose them. You may absolutely make those kernels
genuinely faster — fewer temporaries, better layout, in-place work, fused passes —
but the semantics must survive:

- `softmax(x)`: row-wise over the last axis, numerically stable, rows sum to 1,
  every element in [0, 1].
- `layernorm(x)`: row-wise over the last axis, **no affine parameters** (gain and
  bias are applied by the caller), zero row mean, unit population variance,
  `eps = 1e-5` inside the square root.
- Both must accept 2-D float32 and must be the functions the model actually calls.

## What has the most headroom (measured, as of the baseline)

- The baseline reaches only ~101% CPU. At `batch_size=16` and `n_embd=128` the
  matmuls are too small for Accelerate to thread. **Bigger effective matmuls are
  the single largest lever.**
- Q, K, V are three separate `(C, C)` matmuls on the same input. One `(C, 3C)`
  matmul is the same arithmetic in one BLAS call.
- The causal mask is rebuilt with `np.triu_indices` on **every layer of every step**.
- `lr = 1e-3` is constant, with no warmup and no decay, and `weight_decay = 0`.
- Attention builds a `(B, H, T, T)` float32 array per layer and keeps it for the
  backward pass.

## Your iteration

1. Read `train.py` and `results.tsv` (the ledger: what has been tried and what
   happened). Do not repeat a rejected or reverted change.
2. Make one change, or one tightly-related group of changes. Keep it explainable.
3. Do **not** run `train.py` or the gate yourself — the runner does that, so that
   every candidate is measured identically. Just edit the file and report what you
   changed and why in two or three sentences.
