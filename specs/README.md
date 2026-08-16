# Spec-driven development (SDD)

Every feature starts as a **spec** before code. Tests are written from **acceptance criteria** before implementation (TDD).

## Workflow

1. **Spec** — Add `specs/features/NNNN-short-name/spec.md` (problem, scope, non-goals).
2. **Acceptance** — Add `acceptance.yaml` with checkable criteria (schemas, thresholds, CLI contracts).
3. **Tests** — Add `tests/spec/test_NNNN_*.py` marked `@pytest.mark.spec` — should fail (red).
4. **Implement** — Fill `src/autokernel_pbt/` and `harness/` until green.
5. **Harness** — Wire `harness/bench.py` if the feature affects agent evaluation.

## Feature index

| ID | Feature | Status |
|----|---------|--------|
| [0001](./features/0001-harness-eval/spec.md) | Harness evaluation pipeline | skeleton |
| [0002](./features/0002-correctness-harness/spec.md) | Multi-stage correctness | skeleton |
| [0004](./features/0004-property-oracle-layer/spec.md) | Property/oracle layer (phase 1) | implemented |
| [0005](./features/0005-measurable-runs/spec.md) | Measurable runs | in progress |

## Schemas

Shared JSON Schema definitions live in [`schemas/`](./schemas/). Harness results and kernel candidates must validate against these before recording a run.

## Architecture

High-level design notes: [`architecture/overview.md`](./architecture/overview.md)
