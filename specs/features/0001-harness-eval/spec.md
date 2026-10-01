# Feature 0001: Harness evaluation pipeline

## Problem

Agent and test loops need a **single fixed entrypoint** that returns structured, schema-valid results — analogous to AutoKernel's `bench.py`.

## Scope

- CLI: `harness/bench.py` (or the `akpbt bench` console script)
- Inputs: kernel module path, reference module path, config YAML
- Outputs: `HarnessResult` JSON matching `specs/schemas/harness_result.schema.json`
- Stages: compile → smoke → benchmark (correctness stages expanded in 0002)

## Non-goals

- KernelBench dataset download
- NCU profiling integration

## Dependencies

- `specs/schemas/harness_result.schema.json`
- `specs/schemas/benchmark_config.schema.json`

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
