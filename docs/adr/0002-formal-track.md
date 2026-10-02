# ADR 0002: A formal track, narrowed to agent-written proofs about kernel code

## Status

Accepted (2026-10-01)

## Context

The parent design (`docs/superpowers/specs/2026-08-14-kernel-property-oracle-layer-design.md`,
§10) lists formal verification as a non-goal, and `reference/PBT-property-based-testing/NOTES.md`
§5.4 cedes equivalence checking of statically analyzable GPU kernels to VOLTA.

Two things changed. The property gate was measured admitting wrong kernels its laws cannot see
(`docs/measurements/2026-10-01-softmax-temperature-blind-spot.md`) — a limit of testing, not of any
one tolerance. And the project's direction is now that the kernel-writing agent also writes formal
proofs, in every formal language, cooperating with property-based testing rather than competing
with it.

## Decision

1. The non-goal is narrowed, not removed. In scope: agent-written proofs, in any formal language,
   that a given kernel adheres to a spec the agent also writes — and the measurement of how such
   proofs fail. Out of scope: equivalence checking of statically analyzable kernels, which VOLTA
   covers; human-written proofs; verifying the Triton compiler or the device's IEEE behaviour.
2. The formal track issues certificates. It is not an oracle arm and is not scored as one.
3. Proofs are post-hoc certificates on kernels PBT already accepted, never the optimization loop's
   filter.
4. The design is `docs/superpowers/specs/2026-10-01-agent-formal-adherence-design.md`.

## Consequences

- Features 0009 (the language-neutral core), 0010 (pilot adapters) and 0011 (the pilot run), each
  spec-first under ADR 0001.
- Formal toolchains are optional, like the GPU stack: their tests are gated behind a marker and
  skipped when the toolchain is absent, so CI stays CPU-only and toolchain-free.
- Every formal artifact's claims are executed by the PBT machinery — the same cases, backends and
  rounding budget — so the two tracks check each other rather than running side by side.
