# Contracts

Stable interfaces between subsystems. Implementation types live in `src/autokernel_pbt/`; specs define behavior via acceptance tests.

| Contract | Module | Description |
|----------|--------|-------------|
| `HarnessRunner` | `autokernel_pbt.harness.runner` | Run full eval → `HarnessResult` dict |
| `CorrectnessPipeline` | `autokernel_pbt.harness.correctness` | Stage runner |
| `BenchmarkRunner` | `autokernel_pbt.harness.benchmark` | Timing vs baseline |
| `Population` | `autokernel_pbt.pbt.population` | Candidate set |
| `PBTScheduler` | `autokernel_pbt.pbt.scheduler` | One generation step |
| `MutationHook` | `autokernel_pbt.pbt.mutation` | Explore new candidates |

## Versioning

Bump `HarnessResult.version` when breaking JSON output shape.
