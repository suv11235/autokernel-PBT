# Harness

Fixed evaluation entrypoint for agent and PBT loops.

## Usage

```bash
python harness/bench.py \
  --kernel kernels/triton/candidate.py \
  --reference kernels/triton/reference_relu.py \
  --config harness/configs/default.yaml \
  --dry-run --json
```

## Outputs

JSON matching `specs/schemas/harness_result.schema.json`. Runs should be logged under `.runs/<run_id>/`.

## Specs

- [0001 Harness eval](../specs/features/0001-harness-eval/spec.md)
- [0002 Correctness](../specs/features/0002-correctness-harness/spec.md)
