# Feature 0003: Population-based kernel search (PBT)

## Problem

Single-trajectory agent search can stall in local optima. PBT maintains a population: **exploit** top candidates, **explore** mutations.

## Scope

- `Population` dataclass: members with fitness, lineage, generation
- `PBTScheduler.step()` — select exploiters/explorers, apply mutation hook
- Fitness from harness `speedup_vs_eager` when `passed=true`, else `-inf`
- Config: population size, exploit fraction, mutation rate (YAML)

## Non-goals

- Hyperparameter PBT for LLM weights (kernel config/metadata only in v0.1)
- Crossover of source code (mutation hook only; crossover spec'd for later)

## References

- DeepMind PBT ([1711.09846](../reference/L4-agentic-search/evolutionary-pbt/1711.09846.pdf))
- cuPilot strategy-level evolution ([2512.16465](../reference/L4-agentic-search/evolutionary-pbt/2512.16465.pdf))

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
