# Feature 0004: Property/oracle layer (phase 1, CPU)

## Problem

A kernel is not self-describing. Something must decide whether an execution was correct. That
decision procedure is the **oracle**. The project compares three oracle strategies, so oracle
choice must be a variable that can be changed without changing anything else — including the
inputs the kernel saw.

## Scope

Batch-first record/replay, CPU only:

1. **Generate** — deterministic, seeded case sets described by a serializable `InputDomain`.
   Related inputs are emitted as **case groups** so metamorphic properties are expressible.
2. **Execute** — one backend pass over the whole batch, persisting inputs, outputs and telemetry.
3. **Check** — oracles evaluated offline over the recorded table, never influencing generation.

Three oracle arms: reference (one strong property), declarative (many weak properties), hybrid.

Every property verdict is recorded individually and tagged with its tier and whether it required
a numerical tolerance.

## Non-goals

- Mutation corpus, metrics, and shrinking (phase 2)
- CUDA / Triton / NKI backends and tier-2 telemetry (phase 3)
- bfloat16 (no native NumPy dtype; arrives with the device backends)
- Attention and GEMM tasks; KernelBench integration

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
