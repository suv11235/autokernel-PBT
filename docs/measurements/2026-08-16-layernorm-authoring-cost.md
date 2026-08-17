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
| Wall-clock minutes | | |
| Lines of code authored | | |
| Lines auto-drafted and kept unchanged | | |
| Lines auto-drafted then corrected | | |
| Token cost, if agent-authored | | |
| Defects found in review | | |

**Leave a cell blank rather than estimate it.** A blank cell is data; a remembered number is
not.

## Notes

Record anything that made one arm harder than the other — a property that was hard to state, a
reference that was hard to trust, a relation that turned out vacuous, a tolerance that had to
be derived rather than guessed.

## Threats to this measurement

Written now, while there is no result yet to be tempted to protect.

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
