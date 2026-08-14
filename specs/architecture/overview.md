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
    KRN[kernels/]
  end

  SPEC --> ST
  ACC --> ST
  SCH --> HAR
  ST --> HAR
  HAR --> KRN
```

## Data flow (one evaluation)

1. **Harness** compiles a candidate kernel, runs correctness stages, benchmarks vs baseline.
2. **Correctness** must pass before any speedup is recorded.
3. **Ledger** appends validated JSON lines to `.runs/<run_id>/results.jsonl`.

## Contracts

See [`contracts/`](../contracts/) for typed boundaries between modules.

## Non-goals (v0.1 skeleton)

- Full KernelBench integration (stub hooks only)
- Distributed multi-device evaluation
- LLM agent orchestration (interface reserved in specs)
