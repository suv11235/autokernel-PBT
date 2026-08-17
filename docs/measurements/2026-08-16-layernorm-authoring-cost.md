# Authoring cost: onboarding layernorm

**Metric:** parent design §7 metric 3 — effort to onboard a new task under each oracle strategy.

**Protocol.** This file was created and committed *before any layernorm code existed*, and
filled as the work happened. `n = 1`, which is weak, and any report of it must say so. It is
pre-registered and honest, which reconstructing effort afterwards is not.

The two arms are costed separately because that *is* the comparison: the reference arm needs a
trusted implementation, the declarative arm needs a property set, and the open question is
which is cheaper to author for a new op.

| Measure | Reference arm | Declarative arm |
|---|---|---|
| Wall-clock minutes | ~1 | *not measurable — see below* |
| Lines authored (total diff) | 77 | 176 (129 properties + 47 contract) |
| Lines authored, excluding comments | 36 | 93 + contract |
| Comment/docstring lines | 31 | 19 + 40 |
| Defects found before it was correct | 0 | **1, and it was serious** |
| Design changes forced *elsewhere* | none | **1: the task's input distribution** |

**Wall-clock for the declarative arm is deliberately blank.** Authoring it spanned a blocking
design question that went to a human and back, so the elapsed time measures a conversation
rather than the work. A remembered number would be worse than no number. The line counts and
the defect count are unaffected and are reported.

## The headline result

**The declarative arm cost roughly 2.6× the lines and produced the only defect.** More
importantly, it was the only arm whose authoring forced a change to something *outside itself*.

The reference implementation was written once and was correct on the first run. Its cost was
almost entirely deciding two things that are documented rather than discovered: population
versus sample variance, and `eps` inside versus outside the square root. Both had a canonical
answer available (PyTorch's `LayerNorm`).

The declarative property set was not correct on the first run, and the failure was not a typo.

## The defect, in full, because it is the interesting part

`RowsHaveUnitVariance` **failed the correct reference** on the (8,8) rung.

The tolerance had been derived from float rounding: `THRESH · eps · n`. That is the wrong error
mechanism. The dominant deviation of a correct layernorm output's variance from 1 is the `eps`
regularization inside the square root, and `var/(var + eps) − 1` reproduced every observed value
to two significant figures:

```
observed deviation : 3.2e-05  2.1e-05  2.1e-05  2.7e-05  2.6e-05 ...
var/(var + eps) - 1: 3.2e-05  2.1e-05  2.1e-05  2.7e-05  2.6e-05 ...
```

That term scales as `eps/var`, so it *grows as the row variance shrinks* and has nothing to do
with `n`. Widening the bound by `n` would have fixed the failing rung by accident while leaving
the property wrong in principle.

**A second measurement made the first one worse.** Under the N(0,1) inputs the task originally
inherited from softmax, a kernel that centers but never divides — the exact defect this property
exists to catch — produced row variances of `[0.31, 0.47, 0.48, 0.37, 0.38, 1.04, 1.04, 0.90]`.
Three of eight rows sit within 4% of the target. The property was both false-alarming on correct
kernels *and* barely discriminating the bug it was written for.

**Resolution: the task's input distribution changed**, from N(0,1) to uniform(−10, 10). An
undivided row then has variance ≈ 33, which the property separates from 1 decisively, and the
regularization term falls to ~3e-7 — about 46× below the rounding budget. Verified across all
nine ladder rungs with zero false alarms, and the margin is pinned by a test so narrowing the
distribution fails loudly.

This is the vacuous-property failure mode the project exists to study, observed again in our own
suite. It is the same shape as the phase-1 finding that a unit-scale shift relation caught the
unstable softmax 0% of the time, and it is the second instance of the general rule: **a property
must be scaled to the failure mechanism, and the mechanism is rarely the one you first assume.**

## What the pair buys, measured

Neither property alone pins the normalized family. On the ladder:

| Injected defect | `rows_have_zero_mean` | `rows_have_unit_variance` |
|---|---|---|
| centers, never divides | 0/18 (correctly passes) | **14/18** |
| divides, never centers | **18/18** | 0/18 (correctly passes) |

14/18 rather than 18/18 because the four degenerate single-column cases abstain. A contract
carrying only one of the pair would be blind to exactly one of the two defects — which is the
declarative arm's whole thesis, and now a measurement rather than an assertion.

End to end through the driver, all four arms agree on layernorm: 0/9 on a correct kernel, 7/9 on
`centers_never_divides` — the same 0.778 the ladder's two degenerate rungs impose on softmax.

## Notes

- The declarative arm's line count is inflated relative to a mature library: both properties are
  new *classes*, whereas a third normalization op would reuse them and cost only a contract file
  (47 lines, of which 40 are comment).
- The reference arm's real cost is not visible in either number. It needs a trusted
  implementation to exist; here PyTorch's definition supplied one. For a kernel with no
  canonical reference — the situation the parent design argues is typical — that cost is
  unbounded, and this measurement says nothing about it.

## Threats to this measurement

Written before the result was known, and left unedited afterwards.

- `n = 1`: one op, one author, one session.
- The author had already read and internalized the softmax property set, so the declarative arm
  benefits from transfer a genuinely new task would not get. This biases *toward* the
  declarative arm.
- Wall-clock includes review rounds, and this repo's adversarial review standard makes those
  unusually heavy compared with ordinary practice. Both arms pay it, but not necessarily
  equally: a property's failure modes are subtler than a reference's, so review cost may not
  scale with authoring cost.
- layernorm is the *second* normalization-flavoured op after softmax, so its properties are
  partly analogous to ones already written. A first-of-its-kind op would likely cost more on
  the declarative side.
- The reference implementation had a known-correct definition available (PyTorch's
  `LayerNorm`). An op without a canonical reference would shift cost toward the reference arm,
  which is precisely the situation the parent design argues kernels are usually in.

**Post-hoc addition, flagged as such:** the "design changes forced elsewhere" row was not in the
pre-registered table. It was added because the measurement produced one and there was nowhere
honest to record it. Treat it as an observation, not a pre-registered metric.
