# Feature 0006: Four arms and the normalization rung

## Problem

Three gaps remain between the instrument phase 1.5 delivered and the experiment it exists to
run.

The reference arm was deliberately strengthened into a LAPACK-style normalized test ratio so
it would not be a strawman, and phase 1 measured it as catching everything
`allclose(rtol=1e-5)` catches. That measurement is currently something a reader must take on
trust, because plain `allclose` is not among the arms.

The declarative and hybrid arms — the two the project's claim actually rests on — have no
acceptance criteria, and `HybridOracle` is not driven at all.

The ladder stops at softmax, so the normalization rung named in the parent design is missing,
along with the property class that comes with it.

## Scope

1. **A fourth arm** — plain `allclose`, numpy's defaults, unmodified.
2. **All four arms driven**, in randomized order so `elapsed_s` is not systematically biased
   toward whichever arm ran second.
3. **Acceptance criteria** for the declarative, hybrid and allclose arms.
4. **layernorm** — reference, property set, contract, with its authoring cost measured before
   it is written.
5. **A reducible case spec** — a group is regenerable from `(seed, task_id, group_index,
   shape, transforms)`. No shrinking algorithm.

## Non-goals

- The mutation corpus and the four metrics (phase 2b)
- Any shrinking algorithm — only the representation
- Repeated timing for a fair cost-per-bug denominator (phase 2b)
- Deciding what partial abstention means
- CUDA / Triton / NKI backends and tier-2 telemetry (phase 3)
- Any change to phase 1.5's storage, driver join, or score table

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
