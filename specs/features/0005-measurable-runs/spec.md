# Feature 0005: Measurable runs

## Problem

Phase 1 built every layer but recorded too little to measure. Four of the five research metrics
cannot be computed from the persisted artifacts:

- **Bug-catching power** and **false-positive rate** need to join a row to *which kernel produced
  it* and *whether that kernel was known-broken*. The execution table recorded neither.
- **Cost-per-bug** needs oracle evaluation timed. Only kernel wall time was recorded.
- The **tolerance-free split** was correct in memory but `PropertyResult` was never persisted.

Separately, the pipeline was composed only in test code, twice, and the two copies had already
diverged.

## Scope

1. `ExecutionResult` and the Parquet schema carry `kernel_id` and `kernel_is_broken`.
2. A score table persists `list[PropertyResult]` beside the execution table, joinable by
   `case_id`/`group_id`, and records which oracle arm produced each result and how long that arm took.
3. A `run_task` driver in `src/` composes generate → execute → persist → score for one task and
   one kernel, and is the only such composition.

## Non-goals

- Metric *computation* — this feature makes the numbers derivable, not derived.
- The mutation corpus (Phase 2).
- Device backends and tier-2 telemetry (Phase 3).

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
