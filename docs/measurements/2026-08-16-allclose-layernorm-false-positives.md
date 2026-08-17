# The allclose arm false-positives on correct layernorm kernels

**Metric:** parent design §7 metric 2 — false-positive rate, how often a correct kernel is
flagged.

**Status:** measured, reproduced through the driver, pinned by
`tests/integration/test_four_arms.py::test_allclose_false_positives_on_a_correct_layernorm_kernel`.

**This is a result, not a defect.** `AllcloseOracle` is carried precisely because it is the
untuned default the kernel literature uses. Tuning `rtol`/`atol` to make this go away would
delete the comparison the arm exists to provide.

## The measurement

Seed 42, the full nine-rung ladder, a correct layernorm kernel that accumulates in the input's
own float32 rather than widening to float64.

| Arm | Groups flagged | Rate |
|---|---|---|
| **allclose** | **5 / 9** | **0.556** |
| reference | 0 / 9 | 0.000 |
| declarative | 0 / 9 | 0.000 |
| hybrid | 0 / 9 | 0.000 |

Per case rather than per group: 6 / 18 = 0.333.

## Mechanism

`atol = 1e-8` is **~12× below float32 eps** (1.19e-7). A layernorm output is centered on zero by
construction, so near-zero elements are guaranteed in every row, and for those elements the
budget collapses to `atol`.

Worst element on the (8,8) rung:

| quantity | value |
|---|---|
| reference value at that element | 3.60e-04 |
| deviation of a correct float32 kernel | 4.21e-08 |
| allclose budget `rtol·|ref| + atol` | 1.36e-08 |

The deviation exceeds the budget by ~3×, on an element whose value is itself far above the
noise floor. Per-rung:

| shape | allclose | worst deviation | budget there |
|---|---|---|---|
| (8, 8) | **FAIL** | 4.21e-08 | 1.36e-08 |
| (4, 16) | pass | 1.63e-08 | 7.74e-08 |
| (16, 32) | pass | 1.19e-08 | 3.78e-08 |
| (3, 7) | pass | 7.45e-09 | 5.25e-07 |
| (5, 33) | pass | 1.49e-08 | 1.19e-07 |
| (7, 129) | pass | 3.96e-09 | 4.76e-08 |
| (1, 1) | pass | 0 | 1.00e-08 |
| (1, 64) | pass | 3.73e-09 | 4.54e-07 |
| (17, 1) | pass | 0 | 1.00e-08 |

## Why softmax does not show it

A softmax output is a probability distribution: every element is in [0, 1] and its absolute
error scales with its own magnitude, so `rtol·|ref|` dominates `atol` almost everywhere. The
two tolerance-bearing arms therefore agree on softmax and disagree on layernorm.

**Consequence for the corpus.** A task ladder that stopped at softmax would have reported the
field's default oracle as having a 0% false-positive rate, and the strengthened reference arm's
advantage would have been invisible. It took the normalization rung to expose it. Any
false-positive number this project reports must therefore say which tasks it was measured over,
because the answer is not task-independent.

## What this supports, and what it does not

**Supports:** the reference arm's `log2(n)`-normalized test ratio is scale-invariant, so it is
unmoved by near-zero elements. On this corpus it is strictly better calibrated than the field
default — it flags nothing on a correct kernel while `allclose` flags 56% of groups. That is
direct evidence for parent design §5.3's claim that the reference arm is a *strong* baseline
rather than a strawman, and it is evidence of the kind that could have come out the other way.

**Does not support:** any claim about detection power. This measurement is over correct kernels
only. `allclose` remains competitive at *catching* bugs; what it does badly here is not flagging
correct code.

## Threats

- One kernel, one seed, one dtype (float32), one task. The rate 0.556 is specific to this
  corpus and should not be quoted as "the" allclose false-positive rate.
- The correct kernel was written to accumulate in float32 deliberately, so that the
  tolerance-bearing arms consult their thresholds rather than being handed a residual of
  exactly zero. A kernel that widened to float64 internally would show a lower rate — arguably
  an unrealistically low one, since real kernels do not widen.
- `LAYERNORM_EPS = 1e-5` is PyTorch's default. A kernel targeting a different `eps` would sit
  further from the reference and inflate every tolerance-bearing arm's rate, this one included.
