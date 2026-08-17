# The at-scale hardware run

**Date:** 2026-08-17
**Instance:** Lambda A10, sm_86, 72 SMs, CUDA 12.8, torch 2.7.0, triton 3.3.0
**Supersedes in part:** `2026-08-17-first-hardware-run.md` §3 and §4

**Why this exists.** The first run pushed **430 KB** through an A10 and used at most 17 of its 72
SMs. The device was effectively idle, four of five compiled-telemetry fields were *constant*
across every row, and the tolerance result — reported as validating "Triton's actual reduction
tree" — had in fact only exercised the single-block tree. The corpus is calibrated for detecting
semantic bugs at boundary shapes, and that is a deliberate and defensible choice; but it meant
nothing device-specific was being exercised at all.

---

## 1. What changed

**The tile width is derived from the shape** rather than fixed at 16384. A fixed tile was wrong
in both directions:

- **Too large** and every shape compiles to the same artifact, so `n_regs`, `n_spills`,
  `shared_bytes`, `num_warps` and `num_stages` are constant by construction — the telemetry
  schema carries no information.
- **Too small and the kernel is silently wrong.** `tl.arange(0, BLOCK)` never reaches past the
  tile, so a row wider than it loses its tail with no error. Measured on device with `n_cols =
  4096`: `BLOCK = 2048` makes softmax rows sum to **1.5089**, `BLOCK = 1024` to **2.0311**. The
  ladder never exceeded 16384, so this was unreachable by the existing corpus.

  Worth noting the declarative arm catches it — `rows_sum_to_one` fails outright — so this is a
  defect the oracle layer would have found had any shape reached it. The launcher now refuses the
  configuration rather than returning a plausible wrong answer.

**A `softmax_at_scale` task** with five shapes from `(4096, 1024)` to `(1024, 8192)`: grid depth
far past the SM count, tile widths from 256 to 8192.

## 2. The device is now actually working

| | first run | at scale |
|---|---|---|
| elements through the GPU | 110,207 | **29,470,335** |
| payload | 430 KB | **112 MB** |
| max grid | 17 blocks | **16,384 blocks** (72 SMs) |

Measured kernel time, 50 iterations after warm-up, reading input and writing output:

| shape | tile | kernel time | effective bandwidth |
|---|---|---|---|
| (4096, 1024) | 1024 | 71.2 µs | 471.5 GB/s |
| (8192, 512) | 512 | 71.6 µs | 468.7 GB/s |
| (2048, 4096) | 4096 | 141.2 µs | 475.4 GB/s |
| (16384, 256) | 256 | 72.3 µs | 464.2 GB/s |
| (1024, 8192) | 8192 | 141.0 µs | 476.1 GB/s |

The A10's peak memory bandwidth is ~600 GB/s, so these run at roughly **78% of roofline** —
memory-bound, which is what softmax should be. The kernels are doing real work, not being
dominated by launch overhead.

## 3. The telemetry now carries signal

| field | first run | at scale |
|---|---|---|
| `n_regs` | 2 distinct | **10 distinct** (14 → 168) |
| `shared_bytes` | 1 (constant `16`) | **3 distinct** (0, 8, 16) |
| `n_spills` | 1 (constant `0`) | 1 (constant `0`) |
| `num_warps` | 1 (constant `4`) | 1 (constant `4`) |
| `num_stages` | 1 (constant `3`) | 1 (constant `3`) |

Register pressure scales cleanly and monotonically with tile width:

| tile | 256 | 512 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|
| `n_regs` | 19 | 22 | 25 | 48 | 91 | 168 |

That is the signal the schema was built to carry, and it was invisible before. `n_regs` is now a
usable independent variable for the tier-2 work.

**`n_spills` is still constant zero**, and that is an honest limit rather than a success: at 168
registers on a 255-register budget we never reached spilling, so the spill field remains
unexercised. Provoking it needs either a wider tile than `MAX_BLOCK` allows or a deliberately
register-hungry kernel, and that belongs with the Phase 3b mutation work.

**`num_warps` and `num_stages` are still constant** because Triton's defaults (4 and 3) were
never overridden. They will only vary under an explicit sweep, which remains Phase 3b's.

## 4. Correctness at scale

All four arms on 5 groups, 29.4M elements, correct kernels:

| arm | FAIL | INCONCLUSIVE | PASS |
|---|---|---|---|
| allclose | 0 | 0 | 5 |
| reference | 0 | 0 | 5 |
| declarative | 0 | 0 | 5 |
| hybrid | 0 | 0 | 5 |

Tier-1 property transfer holds at 100× the data volume and full device occupancy, not just on
boundary-shaped toys. `allclose` passes here because the task is softmax — its false positives
are a layernorm phenomenon, driven by outputs centred on zero, and are unrelated to scale.

## 5. What is still not tested

- **The multi-block reduction tree.** Every row still fits in one tile, so a two-stage reduction
  across blocks is untested, and the `log2(n)` result remains scoped to the single-block tree.
  Reaching it needs a genuinely different kernel.
- **Register spilling**, per §3.
- **`num_warps` / `num_stages` sensitivity**, which needs an explicit sweep.
- **Kernel *performance* as a metric.** Nothing here is a benchmark of kernel quality; the
  bandwidth numbers exist to show the device was loaded, not to rank implementations. Parent
  design §7 metric 4 defers downstream kernel speed to a later phase and that is unchanged.
- **Tensor cores**, still untouched — these are elementwise and reduction kernels.

## 6. Threats

- One GPU (A10, sm_86), one dtype, one seed.
- Bandwidth figures are single-run medians over 50 iterations on an otherwise idle device; they
  are indicative, not a careful benchmark.
- `softmax_at_scale` reuses `softmax_reference`, so its numbers stay comparable with every other
  softmax measurement — but it is therefore not an independent test of anything but scale.
