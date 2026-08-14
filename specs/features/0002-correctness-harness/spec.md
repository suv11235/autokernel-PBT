# Feature 0002: Multi-stage correctness harness

## Problem

Kernel candidates must pass staged correctness checks before any speedup is recorded (AutoKernel five-stage pattern).

## Scope

Stages (configurable order):

1. **smoke** — single small shape
2. **shape_sweep** — multiple M,N,K or seq lengths
3. **numerical_stress** — adversarial inputs (extreme magnitude, near-zero variance)
4. **determinism** — repeatability where applicable
5. **edge_cases** — boundary sizes (remainder tiles, odd heads)

Each stage reports `{name, passed, message}` in `HarnessResult.correctness.stages`.

## Non-goals

- FP8 tolerance tables (document in config later)
- Cross-backend numerical equivalence

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
