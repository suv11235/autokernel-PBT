# Contracts

Stable interfaces between subsystems. Implementation types live in `src/autokernel_pbt/`; specs define behavior via acceptance tests.

| Contract | Module | Description |
|----------|--------|-------------|
| `Backend` | `autokernel_pbt.props.backends.base` | `run(kernel, case) -> ExecutionResult` |
| `Oracle` | `autokernel_pbt.props.oracle` | `evaluate(rows) -> list[PropertyResult]`; one per arm |
| `ExecutionTable` | `autokernel_pbt.props.table` | Recorded executions: `rows.parquet` + `tensors/` |
| `ScoreTable` | `autokernel_pbt.props.scores` | Per-arm property results: `scores.parquet` |
| `run_task` / `read_run` | `autokernel_pbt.props.driver` | Record + score a run; read a fingerprint-matched pair |
| `run_harness` | `autokernel_pbt.harness.runner` | Feature 0001 skeleton → `HarnessResult` dict |
| `run_stages` / `should_run_benchmark` | `autokernel_pbt.harness.correctness` | Feature 0002 skeleton stages |
| `run_benchmark` | `autokernel_pbt.harness.benchmark` | Placeholder timing |

## Versioning

Bump `HarnessResult.version` when breaking JSON output shape.
