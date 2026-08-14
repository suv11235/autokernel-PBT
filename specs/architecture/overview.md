# Architecture overview

## Components

```mermaid
flowchart LR
  subgraph spec["Spec layer"]
    SPEC[spec.md]
    ACC[acceptance.yaml]
    SCH[JSON schemas]
  end

  subgraph test["Test layer"]
    UT[unit tests]
    ST[spec tests]
    IT[integration]
  end

  subgraph runtime["Runtime"]
    HAR[harness/bench.py]
    PBT[PBT scheduler]
    KRN[kernels/]
  end

  SPEC --> ST
  ACC --> ST
  SCH --> HAR
  ST --> PBT
  HAR --> PBT
  PBT --> KRN
```

## Data flow (one PBT generation)

1. **Population** holds N kernel candidates (source + metadata).
2. **Harness** compiles, runs correctness stages, benchmarks vs baseline.
3. **Fitness** derived from `HarnessResult` (correctness gate + speedup).
4. **PBT step** copies weights/config from top performers, mutates explorers.
5. **Ledger** appends validated JSON lines to `.runs/<run_id>/results.jsonl`.

## Contracts

See [`contracts/`](../contracts/) for typed boundaries between modules.

## Non-goals (v0.1 skeleton)

- Full KernelBench integration (stub hooks only)
- Multi-GPU distributed PBT
- LLM agent orchestration (interface reserved in specs)
