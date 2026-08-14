# ADR 0001: Spec-driven and test-driven development

## Status

Accepted

## Context

autokernel-PBT combines agent-style kernel development with property-based validation. Requirements evolve quickly; we need traceability from spec → test → harness behavior.

## Decision

1. Every feature gets `specs/features/NNNN-*/spec.md` and `acceptance.yaml` before implementation.
2. Tests in `tests/spec/` are marked `@pytest.mark.spec` and map to acceptance criteria.
3. Harness output is JSON Schema validated (`specs/schemas/`).
4. `harness/bench.py` is the stable agent-facing evaluation contract.

## Consequences

- CI runs unit + spec tests without GPU.
- GPU benchmarks gated behind `@pytest.mark.gpu` and optional deps.
- Breaking harness JSON requires schema version bump.
