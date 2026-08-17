# Design: Phase 3a — the Triton backend and the first hardware run

**Date:** 2026-08-17
**Status:** Approved (design); implementation plan pending
**Scope:** A Triton/CUDA backend, the telemetry it must capture, Triton ports of the existing
ladder, and one recorded GPU run scored offline. Tier-2 properties, sanitizer integration, the
mutation corpus and NKI are explicitly out of scope.

**Parent design:** `docs/superpowers/specs/2026-08-14-kernel-property-oracle-layer-design.md`

---

## 1. Why this can jump the queue

The parent design phases device work last, after the mutation corpus and metrics. That ordering
is not a dependency — it is a convenience, and the architecture was built to let it be reordered.

Scoring is **offline, over the recorded table, with no backend involved** (§3.1). A hardware run
is therefore a reusable dataset: record now, score whenever the metrics layer exists. That is
the property record/replay was designed to buy, and taking it early is using the architecture as
intended rather than cutting a corner.

Concretely, Phase 3a needs none of Phase 2b. It needs a backend, telemetry, and kernels.

## 2. What the first run produces

Two results, neither of which requires a mutation corpus:

**Tier-1 property transfer** — parent design open question 2: *which declarative properties fail
to transfer across backends, and why*. The same contracts, the same property set, a different
backend. The measurement is the false-positive rate of the tier-1 set on a **correct** Triton
kernel: any property that passes on NumPy and fails on Triton for a correct kernel has failed to
transfer, and the reason is a finding.

**Tolerance validation under a real reduction tree.** The reference arm's `log2(n)` normalization
is the pairwise-summation bound. NumPy reduces pairwise; Triton reduces in a block-level tree
whose shape depends on `BLOCK_SIZE` and `num_warps`. Whether `log2(n)` holds against Triton's
actual tree is measurable here and has never been checked.

### A correction to carry into the plan

Parent design §5.3 warns that NVIDIA tensor cores accumulate in round-toward-zero, "a systematic
non-cancelling bias that breaks any √n-calibrated tolerance." **That does not bite on this
ladder.** Tensor cores engage for matrix multiply; relu, softmax and layernorm are elementwise
and reduction kernels that run on the regular FP32 pipeline. The concern is real and stands for
GEMM and attention, and it should not be claimed as validated by this run.

What this run does validate is reduction-order and block-size sensitivity, which is a different
and smaller claim. Saying otherwise would overstate the result.

## 3. Telemetry: the one irreversible decision

A missing counter costs another hardware run. Everything else is re-derivable offline for free,
so the schema is over-specified deliberately.

**Always on, because it is free at compile time.** All of it comes off the JIT artifact and the
device handle:

| Group | Fields |
|---|---|
| Device | name, compute capability, total/free memory, multiprocessor count |
| Toolchain | driver version, CUDA runtime version, torch version, triton version |
| Launch | grid, `num_warps`, `num_stages`, and every `BLOCK_*` constexpr the kernel was specialized on |
| Compiled artifact | registers per thread, spill stores/loads, shared-memory bytes, local-memory bytes, PTX hash, SASS hash where available |
| Occupancy | theoretical occupancy, max active blocks per multiprocessor |
| Timing | wall ms, and CUDA-event device ms measured separately |

The launch and compiled-artifact groups are not incidental. "Tile mapping and launch" and the
register/spill signals are precisely what the ISSTA taxonomy's device-only fault classes are
about, and none of it can be recovered from `(inputs, outputs)` afterwards.

**Opt-in, because it does not compose.** `compute-sanitizer`'s four subtools cannot be combined —
each is a separate process and a separate slowdown, so always-on memcheck would buy one subtool
at most of the price of all four. `ncu` counters likewise. Both are per-run flags, off by
default, and both belong to Phase 3b.

**Schema shape.** Telemetry is already a free-form JSON column on the execution row, so no
storage change is needed. It gains a `schema_version` integer so a later reader can tell a run
recorded before a field existed from a run where the field was genuinely absent — the difference
between "not captured" and "captured as zero" is exactly the kind of thing that silently
corrupts an aggregate.

## 4. The kernel interface

Parent design §5.1 already fixes the shape: kernel *source* stays backend-native, and "the
harness treats a kernel as an artifact plus a compile/launch adapter."

`Backend.run` takes `Callable[..., np.ndarray]`. A Triton kernel is not that: it needs a launch
grid, constexpr block sizes, and torch device tensors. So the thing handed to `run` is a
**callable adapter object** — it satisfies the existing protocol, and it additionally exposes the
compiled artifact the backend reads telemetry from.

```
TritonKernel                      # callable; satisfies Callable[..., np.ndarray]
  jit_fn                          # the @triton.jit function
  launch(**numpy_inputs)          # host numpy -> device torch -> launch -> host numpy
  compiled                        # populated after first call; the telemetry source
  constexprs                      # BLOCK_SIZE etc., recorded into launch telemetry
```

The protocol is unchanged. `NumpyBackend` keeps taking plain functions; the Triton backend
requires an adapter, and says so with a clear error rather than a `TypeError` from inside a
launch.

## 5. Three device realities the CPU backend never faced

**Compilation is a distinct failure mode, and it is lazy.** Triton compiles on first call, so a
compile error surfaces during execution rather than before it. `Status.COMPILE_ERROR` already
exists in the vocabulary and is currently unused; this is what it is for. Distinguishing it from
`LAUNCH_ERROR` matters because a kernel that never compiled is not evidence about numerics, and
both must remain `INCONCLUSIVE` in every arm.

**The read-only-inputs guarantee weakens.** `CLAUDE.md` records that kernel inputs are read-only
during execution and that `readonly_inputs` turns silent corruption into a loud `launch_error`.
On device that guard protects the *host* array, which the kernel never touches — a Triton kernel
that writes to its input parameter mutates the device copy, invisibly. The recorded input stays
correct (it is written from `case.tensors` before the copy), so replay fairness is not
threatened. But "the kernel did not modify its inputs" stops being enforced. The backend
therefore hashes the device input buffers before and after launch and reports a mismatch as
`LAUNCH_ERROR`, restoring the guarantee at the only place it can now be checked.

**Execution is not bitwise reproducible.** Atomics and non-deterministic reduction order mean
re-running a Triton kernel need not reproduce the recorded output. This does not threaten
anything — it is an argument *for* record/replay, since the recorded execution is the one the
arms score and re-execution is never required. It does mean a "re-run and compare" test is not
available on device, and the plan must not assume one.

## 6. CI and local development

This is developed on a machine with no CUDA, so the backend is written blind and cannot be run
until hardware. Two mitigations:

- **`gpu`-marked tests** that skip cleanly when `torch.cuda.is_available()` is false. The marker
  is already declared in `pyproject.toml` and is currently used by nothing.
- **A CPU-runnable contract test** over the backend's *structure* — that `TritonKernel` satisfies
  the callable protocol, that the telemetry dict has every declared key, that the status mapping
  is total. This catches the majority of wiring defects before any hardware is paid for, which
  is the point.

CI stays CPU-only and green. GPU tests are run by hand on the rented instance.

## 7. Sequence

1. `TritonKernel` adapter and the telemetry schema, with CPU contract tests.
2. `TritonBackend`, `gpu`-marked.
3. Triton ports of relu, softmax, layernorm, each matching its existing `acceptance.yaml`.
4. **A short smoke session on rented hardware** — compile, launch, record one small run, and
   confirm the telemetry schema survives contact with reality *before* the real run. The
   irreversible decision deserves one cheap rehearsal.
5. The real run: full ladder, correct kernels, recorded and brought home.
6. Score offline on CPU with the existing four arms. Report tier-1 transfer and the tolerance
   result.

## 8. Out of scope

- Tier-2 properties and the `compute-sanitizer` / `ncu` integration (Phase 3b)
- The mutation corpus and the four metrics (Phase 2b)
- NKI / Trainium, and the numerical-characterization probe that must precede it
- GEMM and attention — and therefore any claim about tensor-core round-toward-zero
- Autotuning, and any search over launch configurations

## 9. Open questions

1. How many launch configurations per task? One fixed config is simplest and makes the run a
   clean tier-1 transfer measurement; sweeping `BLOCK_SIZE` would additionally probe the
   block-size sensitivity §2 mentions, at a multiple of the hardware time.
2. Which GPU? Compute capability changes register limits, occupancy and available SASS
   introspection, and the telemetry should record it rather than assume it.
3. Does `log2(n)` need a Triton-specific reduction length? Triton reduces within a block and then
   across blocks, so the effective tree depth may not be `log2(row_length)`. If the first run
   shows a systematic drift in the correct-kernel ratio, that is the likely cause and it is a
   finding, not a defect.
