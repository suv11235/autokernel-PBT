# Harness

CLI over the feature 0001/0002 skeleton. **Every stage passes and the benchmark numbers are
placeholders** — it does not load the kernel or reference. The real pipeline is
`autokernel_pbt.props.driver.run_task` (see `scripts/gpu_record.py`).

## Usage

```bash
python harness/bench.py \
  --kernel path/to/candidate.py \
  --reference path/to/reference.py \
  --config harness/configs/default.yaml \
  --dry-run --json
```

## Outputs

JSON matching `specs/schemas/harness_result.schema.json`. Nothing is persisted.

## Specs

- [0001 Harness eval](../specs/features/0001-harness-eval/spec.md)
- [0002 Correctness](../specs/features/0002-correctness-harness/spec.md)
