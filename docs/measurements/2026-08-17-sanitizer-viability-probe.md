# compute-sanitizer viability probe

**Date:** 2026-08-17
**Instance:** Lambda A10, sm_86, CUDA 12.8, triton 3.3.0, compute-sanitizer 2025.1.0.0
**Purpose:** test Phase 3b's load-bearing assumption — that `compute-sanitizer` can see defects in
Triton-generated kernels — while a validated instance was still live.

**Headline:** it can, **but PyTorch's caching allocator hides them completely.** Without
`PYTORCH_NO_CUDA_MEMORY_CACHING=1`, memcheck reports a clean run on a kernel reading a megabyte
past its buffer.

---

## The measurement

One Triton kernel, one deliberate indexing defect: the mask is computed against `n_cols + OVER`
instead of `n_cols`, so every program reads and writes past the end of its row. Buffer is 1 KB.

| overrun | caching allocator | memcheck result |
|---|---|---|
| 256 bytes | default (on) | `ERROR SUMMARY: 0 errors` |
| 16 KB | default (on) | `ERROR SUMMARY: 0 errors` |
| 1 MB | default (on) | `ERROR SUMMARY: 0 errors` |
| **16 KB** | **`PYTORCH_NO_CUDA_MEMORY_CACHING=1`** | **`Invalid __global__ read of size 16 bytes … is out of bounds`** |

Same kernel, same overrun, opposite verdicts. A correct kernel reports 0 errors under both, so
the check is not simply always-on.

## Why

PyTorch's CUDA caching allocator issues a small number of large `cudaMalloc` segments and
sub-allocates tensors inside them. `compute-sanitizer` tracks *actual* allocations, so an overrun
that stays inside a segment is not an out-of-bounds access at the CUDA level — there is nothing
for memcheck to report, and it is correct not to report it.

Because the allocator's segments are large and reused, **essentially every realistic tensor
overrun is intra-segment.** The failure is not marginal; it is the default.

This is the concrete, PyTorch-specific instance of the limitation
`reference/PBT-property-based-testing/NOTES.md` §5.1 records from *Chasing Elusive Memory Bugs in
GPU Programs*: intra-allocation OOBs that allocation-granularity checkers structurally cannot
see. The paper describes the class; this is what it looks like in our stack, with a mitigation.

## What it means for Phase 3b

**The assumption survives, with a mandatory flag.** `compute-sanitizer` does work on
Triton-generated PTX — the tool is not confused by the JIT, the kernel names resolve, and the
error report is precise. The Phase 3b design does not need rethinking.

**But a memcheck integration written without that flag would be a false-negative machine.** It
would report zero errors on genuinely out-of-bounds kernels and make *indexing and stride* — 35
of 301 bugs, the largest CPU-unreachable subcategory in the ISSTA taxonomy — read as
undetectable. The tier-2 detection rate would have come out near zero for a reason having nothing
to do with the oracle.

`PYTORCH_NO_CUDA_MEMORY_CACHING=1` must therefore be set for any sanitizer run, and the Phase 3b
backend should set it itself rather than documenting it, since a forgotten env var is
indistinguishable from a clean result.

**Correctness is unaffected by the flag.** The full `gpu`-marked suite passes identically with
caching disabled (7 passed), so it changes allocation behaviour and nothing observable about the
kernels.

## Cost

| configuration | wall time | vs baseline |
|---|---|---|
| baseline | 2.2 s | 1.0× |
| memcheck | 2.6 s | 1.2× |
| no-caching only | 2.1 s | 1.0× |
| memcheck + no-caching | 2.7 s | 1.2× |

**This 1.2× is a floor, not a general cost, and should not be quoted as one.** The measured
workload is the `gpu` test suite, which is dominated by process startup and Triton JIT
compilation; the kernels themselves run in microseconds. Sanitizer overhead applies to *kernel
execution*, which is a rounding error here. A workload with substantial device time will see far
more.

What it does establish is that memcheck is affordable on *this* corpus, where the ladder is nine
trivial kernels — so Phase 3b need not treat it as prohibitively expensive at this scale.

Separately, the four subtools still do not compose: memcheck, racecheck, initcheck and synccheck
are four separate passes. That multiplier is unchanged by this probe and is a property of the
tool, not the workload.

## Threats

- One GPU (A10, sm_86), one CUDA version, one PyTorch version. Allocator behaviour is a PyTorch
  implementation detail and could change.
- Only memcheck was exercised. racecheck, initcheck and synccheck are untested here, and
  racecheck in particular is documented to detect only shared-memory races.
- The OOB was a *masking* defect, which is the most natural Triton indexing bug. Other shapes —
  a wrong stride, a bad broadcast — may interact with the allocator differently.
- The cost figures are from a workload with negligible device time; see above.
