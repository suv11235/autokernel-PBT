# Feature 0007: Triton backend and the first hardware run

## Problem

Every measurement this project has produced is from one backend. The tier-1 properties claim to
be portable — they are the cross-backend equivalence contract the translation workstream is
built on — and that claim has never been tested against a second backend. The reference arm's
`log2(n)` tolerance is the pairwise-summation bound, and no backend other than NumPy has been
checked against it.

Recording on hardware needs neither the mutation corpus nor the metrics layer, because scoring
is offline over the recorded table.

## Scope

1. **Telemetry schema** — everything free at compile time, captured on every execution, with a
   `schema_version` and defensive extraction that records missing fields as missing.
2. **`TritonKernel`** — a callable adapter satisfying the existing `Backend` protocol while
   exposing the compiled artifact.
3. **`TritonBackend`** — device execution, status mapping including `COMPILE_ERROR`, and an
   input-mutation check that lives where the device buffers do.
4. **Triton ports** of relu, softmax and layernorm against their existing contracts.
5. **A tolerance-sweep task** whose reduction lengths reach 16384, because the ladder spans only
   `log2(n)` 0..7.
6. **A runbook** for the Lambda session.

## Non-goals

- Tier-2 properties, `compute-sanitizer`, `ncu` (Phase 3b)
- The mutation corpus and the four metrics (Phase 2b)
- NKI / Trainium
- GEMM, attention, and any claim about tensor-core round-toward-zero
- Autotuning or any search over launch configurations

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
