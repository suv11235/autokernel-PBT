# The first hardware run

**Date:** 2026-08-17
**Instance:** Lambda Cloud, NVIDIA A10, compute capability 8.6, 72 SMs, 23.7 GB
**Toolchain:** driver/CUDA 12.8, torch 2.7.0, triton 3.3.0 (Lambda Stack)
**Recorded:** 52 rows across four tasks, seed 42, every execution `ok`
**Scored:** offline on CPU, no device involved — the property the record/replay architecture exists to buy

---

## 1. Tier-1 property transfer: clean

Parent design open question 2 — *which declarative properties fail to transfer across backends*
— answered for this corpus: **none of them do.**

| task | allclose | reference | declarative | hybrid | groups |
|---|---|---|---|---|---|
| relu | 0 | 0 | 0 | 0 | 9 |
| softmax | 0 | 0 | 0 | 0 | 9 |
| layernorm | **5** | 0 | 0 | 0 | 9 |
| tolerance_sweep | 0 | 0 | 0 | 0 | 7 |

FAIL counts on **correct** Triton kernels, so every non-zero is a false positive.

The reference, declarative and hybrid arms flagged nothing anywhere. Properties written against
NumPy semantics — finiteness, unit interval, row sums, zero mean, unit variance, shift invariance
— held against a completely different execution substrate with no adjustment. That is the
cross-backend equivalence claim the translation workstream rests on, and it now has evidence
rather than an assumption behind it.

Layernorm's declarative arm returned INCONCLUSIVE on 2 of 9 groups, which is the designed
abstention on the `(1, 1)` and `(17, 1)` rungs where a constant input normalizes to zeros. Not a
transfer failure.

**Scope.** Four tasks, one GPU, one dtype, elementwise and reduction kernels only. It says
nothing about GEMM, attention, or fused kernels, and nothing about tier-2 properties.

## 2. The allclose false-positive rate replicates exactly

On CPU, `allclose` false-positived on **5 of 9** layernorm groups. On the A10, running actual
Triton kernels: **5 of 9**.

The mechanism is backend-independent, as predicted: `atol=1e-8` sits ~12× below float32 eps, and
a layernorm output is centered on zero by construction, so near-zero elements are guaranteed in
every row and their error budget collapses to `atol`. Nothing about the backend changes that.

This is worth more than the CPU measurement alone. A reviewer can reasonably ask whether a
CPU-only false-positive result is an artifact of NumPy; the answer is that the same arm fails the
same 5 of 9 groups on real hardware, and the strengthened reference arm passes all nine on both.

## 3. Tolerance: `log2(n)` holds comfortably, and `n=1` is flatter — again

Measured on the tolerance-sweep task, reduction lengths 8 → 16384 (`log2(n)` 3 → 14), correct
Triton softmax against the float64 reference:

| n | log2(n) | ratio ÷ log2(n) | ratio ÷ 1 | ratio ÷ √n |
|---|---|---|---|---|
| 8 | 3.0 | 0.3275 | 0.9824 | 0.34734 |
| 64 | 6.0 | 0.0997 | 0.5981 | 0.07477 |
| 129 | 7.0 | 0.0926 | 0.6492 | 0.05716 |
| 512 | 9.0 | 0.0568 | 0.5114 | 0.02260 |
| 4095 | 12.0 | 0.0913 | 1.0953 | 0.01712 |
| 4096 | 12.0 | 0.0757 | 0.9086 | 0.01420 |
| 16384 | 14.0 | 0.0552 | 0.7731 | 0.00604 |
| **drift** | | **5.9×** | **2.1×** | **57.5×** |

**`log2(n)` is safe.** The largest correct-kernel ratio is 0.327 against a threshold of 30 — about
92× of headroom. No correct Triton kernel comes close to being flagged, which is what the
normalization has to guarantee first.

**`√n` is decisively wrong**, drifting 57.5×. It over-normalizes hard, exactly as the CPU
measurements said.

**`n=1` is flatter than `log2(n)` on device too** — 2.1× against 5.9×, the same direction and
similar magnitude as the CPU result (1.2–1.3× against 2.3–4.1×) in
`2026-08-17-normalization-discrepancy-resolved.md`. Two independent substrates now agree, which
removes the last reason that comparison was being deferred.

This is **not** a recommendation to switch. Flatness is one criterion; `n=1` is also *stricter*,
and §2 above is a live demonstration that stricter is not free. Deciding it needs the mutation
corpus, so it stays a Phase 2b question — but it is now a question with consistent evidence on
both sides of the backend boundary.

Note the 4095/4096 pair: a non-power-of-two length sits slightly *above* its power-of-two
neighbour (0.0913 vs 0.0757) rather than wildly off, so Triton's masked tail handling is not
introducing a systematic error. That pair was put in the sweep to make tail handling visible, and
it is visible and benign.

## 4. Telemetry: every declared field populated

The one irreversible decision, validated. Zero fields came back empty:

```
telemetry_schema_version 1          n_regs 168        n_spills 0
shared_bytes             16         num_warps 4       num_stages 3
device_name              NVIDIA A10 compute_capability 8.6
multi_processor_count    72         total_memory_bytes 23684841472
driver_version           12.8       runtime_version 12.8
torch_version            2.7.0      triton_version 3.3.0
grid                     [8]        constexprs {'BLOCK': 16384}
ptx_hash                 d464a6e9b34886f6              input_mutated False
```

The probe locations chosen blind were all correct for Triton 3.3.0: `n_regs` and `n_spills` on
the `CompiledKernel`, `shared` falling through to `metadata.shared`. Both `n_spills` and
`shared_bytes` legitimately read 0 — the falsy-not-absent case the schema was explicitly
designed to distinguish, and it distinguished it.

## 5. Three defects the run found that no local test could

All three passed every CPU check beforehand.

**The numpy ABI break.** `pip install -e ".[dev]"` upgraded numpy to 2.x against Lambda Stack's
torch 2.7.0, which is built for 1.x. Symptom: `Failed to initialize NumPy: _ARRAY_API not found`
— torch silently lost its numpy interop, and the Triton backend does `torch.as_tensor(ndarray)`
on every launch. It would have failed on the first kernel. Caused by `numpy>=1.24` with no upper
bound in `pyproject.toml`; fixed on the instance by pinning `numpy<2`.

**The compiled artifact was never captured.** `TritonKernel._record_compiled` existed, was
tested, and was called by nothing — the launcher discarded Triton's return value. Every compiled
telemetry field read `MISSING` while the schema looked complete. This is precisely the failure
the schema's "declared keys are always present" rule was designed to make *visible* rather than
silent, and it worked: the field was present and empty, not absent.

**`pythonpath` is a pytest setting.** The device tests imported `kernels.triton.ladder` fine
because pytest injects the repo root; the standalone recording script died with
`ModuleNotFoundError`. Tests green, script broken.

The smoke session existed to find exactly this class of thing while it was cheap. It found three.

## 6. Threats

- One GPU model (A10, sm_86). Register counts, occupancy and SASS introspection are
  capability-specific; a second capability would be a separate measurement.
- One launch configuration per kernel (`BLOCK = 16384`, one program per row). Block-size
  sensitivity is deliberately unexplored and is Phase 3b's.
- One seed (42) and one dtype (float32).
- The tolerance sweep uses 4 rows per shape; the quantity under study is the reduction length,
  so row count buys variance reduction rather than range.
- These are *correct* kernels only. Nothing here measures detection power — that needs the
  mutation corpus.
- Tensor cores were not exercised: relu, softmax and layernorm run on the regular FP32 pipeline.
  The round-toward-zero concern in parent design §5.3 remains untested and must not be claimed as
  validated by this run.
