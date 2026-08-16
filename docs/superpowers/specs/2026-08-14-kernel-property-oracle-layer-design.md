# Design: Kernel property/oracle layer

**Date:** 2026-08-14
**Status:** Approved (design); implementation plan pending
**Scope:** First slice of the autokernel-PBT project — the property and oracle layer. Kernel
generation, translation, and any search loop are downstream consumers and are out of scope here.

---

## 1. Problem

A compute kernel is not self-describing. Before any generated or translated kernel can be judged,
something must decide whether an execution was correct. That decision procedure is the **oracle**.

Today the field's default oracle is `torch.allclose(candidate, reference)` against a PyTorch eager
implementation. That choice is rarely examined, and it has two known failure modes:

1. It encodes the reference's reduction order and an arbitrary per-dtype tolerance as if they were
   specification.
2. In low-precision, long-reduction regimes it is close to informationless — the classical
   `γ_n ≈ n·u` error bound is vacuous for bf16 at K≥1024.

The alternative — many individually-weak declarative properties (algebraic laws, metamorphic
relations) — is cheap to state but of unknown power. Nobody has measured the trade for kernels.

## 2. Research framing

> **Which oracle strategy, at what cost, catches which fault classes — measured over identical
> replayed executions, across three backends, with tolerance-free and tolerance-dependent
> detection reported separately.**

This framing was chosen deliberately over the more obvious "properties beat reference outputs,"
which is now settled literature (see `reference/PBT-property-based-testing/NOTES.md` §7.1).

Three oracle strategies form a spectrum from one strong property to many weak ones:

| Arm | Properties | Character |
|-----|-----------|-----------|
| **Reference** | One: `output ≈ reference(input)` | Maximally strong, maximally brittle |
| **Declarative** | Many algebraic/metamorphic laws | Individually weak, jointly constraining |
| **Hybrid** | Composition with precedence | Laws as filter, reference where trustworthy |

**Prior art that must be cited rather than claimed** (all established as of mid-2026): reference
oracle inadequacy; declarative kernel contracts as a concept; LLM-authored properties; and
"hybrid beats either" on non-numeric code. What remains open: oracle strategy as a controlled
independent variable scored by fault detection; byte-identical replayed executions; cross-backend
portability including NKI; and cost-per-bug.

**A published result points the other way and must be addressed head-on.** Hughes'
*How to Specify It!* found the model-based oracle (our reference arm) caught all 8 planted bugs
roughly 10× faster than metamorphic properties. His stated escape clauses — use metamorphic when
the model is expensive or replicates the implementation's bugs — are exactly the kernel situation,
and that is the argument this project must make empirically rather than assert.

## 3. Architecture

### 3.1 Batch-first record/replay

The property-based testing loop (generate → execute → check) is normally one tight function. Here
it is **split into three stages**:

```
Phase A  Generate    seeded case set, deterministic, serializable
Phase B  Execute     whole batch on one backend, one job, persist every row
Phase C  Check       oracles evaluated OFFLINE over the recorded table
Phase D  Shrink      on failure only, re-executes reduced inputs
```

Two properties follow, and they are the reason for the design:

- **Fair comparison by construction.** One hardware run replays through all three oracle
  strategies over identical inputs. If oracle choice influenced generation, the comparison would
  confound generation with checking and the headline claim would be unfalsifiable.
- **Hardware runs become reusable datasets.** A fourth oracle can be scored months later without
  touching a device. This matters because hardware is non-persistent cloud, not local.

The cost is losing adaptive generation and free shrinking. Shrinking becomes an explicit Phase D.

### 3.2 The execution row

```
ExecutionRow
  case_id
  group_id          ← case-group identity (§3.3)
  task_id, backend, dtype, shape
  inputs            ← tensor payload reference
  outputs           ← tensor payload reference
  telemetry         ← §3.4
  status            ← ok | compile_error | launch_error | timeout
```

**Storage is split.** Tensor payloads go to **safetensors** (zero-copy, dtype-faithful,
framework-neutral); execution-row metadata goes to **Parquet** (columnar, and the analysis is
column-oriented aggregation over many rows). The existing JSONL run ledger is retained for
run-level records only. Tensor data is far too large for the JSON ledger the current skeleton
uses.

### 3.3 Case groups

Metamorphic properties need a *second* execution of the same kernel on a transformed input —
softmax shift invariance needs both `X` and `X + c`. Therefore:

- The generator emits **case groups**: related inputs sharing a `group_id` and a relation tag.
- Groups also cover execution-parameter variation (split-K, dtype ladders), not just input
  transforms.
- The batch and the table must preserve group identity end to end.

If rows were modelled as flat and independent, roughly half the declarative arm would be
unimplementable. This is the single most important structural requirement.

### 3.4 Property tiers and telemetry

**Tier 1 — portable/semantic.** Hold for any correct implementation on any backend. Pure
functions of `(inputs, outputs)`. These are the cross-backend equivalence contract used by the
translation workstream.

**Tier 2 — backend-specific.** Encode stack idiosyncrasies: CUDA OOB/race/launch-bound/register
spill; Triton block-size and mask constraints; NKI tile, SBUF/PSUM capacity and layout limits.

**Most tier-2 properties are not functions of `(inputs, outputs)`.** They need side-channel data —
sanitizer output, compiler diagnostics, occupancy, register counts. That must be captured *during*
Phase B, on the device, in the same job. It cannot be recovered offline, and re-running a Trainium
job to add a missing counter is precisely the cost this architecture exists to avoid.

Known constraint: `compute-sanitizer`'s `racecheck` detects only shared-memory races; no subtool
detects global-memory races; the four tools do not compose (4× budget); output is XML only.

### 3.5 Per-property attribution and the tolerance-free tag

Every property verdict is recorded individually, not collapsed into pass/fail. Each property
carries:

- `tier` — 1 or 2
- `tolerance_free` — whether the check needs a numerical tolerance argument at all

Attribution answers whether ten properties are pulling their weight or one is doing all the work.
The tolerance-free tag supports what may be the sharpest defensible claim: **not "more bugs," but
"bugs found without a tolerance argument."** About a third of the catalogue qualifies — causal-mask
locality, batch-axis independence, order preservation, exact zeros, convex-hull bounds.

### 3.6 Three-valued verdicts

Oracles return **pass / fail / inconclusive**. The third value is mandatory: "the reference itself
overflowed" or "no tolerance is defined for this fp8 case" must not count as a caught bug, or the
false-positive metric is meaningless.

## 4. Component design

| Component | Responsibility | Depends on |
|-----------|---------------|------------|
| `InputDomain` | Serializable, seeded description of a task's input space | — |
| `Generator` | `InputDomain` → deterministic case set with groups | `InputDomain` |
| `Backend` | Compile + launch on one target; emit outputs and telemetry | — |
| `ExecutionTable` | Persist/load rows; split tensor and metadata storage | — |
| `Property` | One predicate over rows/groups; tagged tier + tolerance_free | `ExecutionTable` |
| `Oracle` | A property set plus composition policy; three-valued verdict | `Property` |
| `Shrinker` | Domain-specific reduction of a failing case | `Backend`, `Oracle` |
| `MutationCorpus` | Deliberately-broken kernels with ground truth | `Backend` |
| `ExperimentRunner` | Vary oracle, hold executions fixed; emit metrics | all |

Each is independently testable on CPU. `Backend` is the only component that needs a device, and it
is behind an interface with a CPU/NumPy implementation as a first-class member.

## 5. Key decisions

### 5.1 Language: Python 3.11+

Decided by ecosystem, not preference: Triton and NKI are both Python DSLs, PyTorch references are
Python, KernelBench is Python. Kernel *source* stays backend-native (CUDA C++ via
`torch.utils.cpp_extension`); the harness treats a kernel as an artifact plus a compile/launch
adapter.

Declarative specs are a **Python eDSL, not YAML** — properties must compute (permutations,
shape-dependent anchors, tolerances). Their law list serializes into the ledger for attribution.

### 5.2 Hypothesis: strategy library, not driver

**Use `hypothesis.extra.numpy` / `extra.array_api` as generators; write a thin custom batch
driver; do not use `@given` as the driver.**

Reasons:
- `@given` interleaves generation with evaluation, so two oracle strategies would see different
  inputs — the exact confound the architecture exists to remove.
- Hypothesis's shrinker reduces the internal choice sequence and **re-runs the generator and the
  property**, which for us means re-executing on hardware. It cannot work offline.
- Reproducibility is weaker than needed: `derandomize` holds only until Hypothesis or Python is
  upgraded; `@reproduce_failure` makes no cross-version guarantee; undocumented internals may
  break in patch releases. A Hypothesis corpus is reproducible-by-artifact, not by seed.

But its float edge-case tuning (subnormals, ±0.0, exponent boundaries) is a decade of work and is
where bugs live. So: **harvest a corpus once** via the supported `@given(phases=[Phase.generate])`
route, materialize and hash it. Keep a live Hypothesis loop for Phase D minimization only, where
`hypothesis.target()` steering on relative error is a cheap win.

Case groups are something Hypothesis has no notion of and we must own regardless, which makes the
marginal cost of owning the driver small.

### 5.3 The reference arm uses test ratios, not `allclose`

LAPACK-style normalized residuals — e.g. `‖b−Ax‖/(‖A‖‖x‖ε)` — with a single dimensionless
threshold across every routine, size and precision (LAWN 41 §7.1.1 uses `THRESH = 30.0`).

This makes the reference arm a *strong* baseline rather than a strawman, which the comparison's
credibility depends on. Nobody in the kernel literature currently does this.

Tolerance derivation, not guessing: the classical `γ_n ≈ n·u` bound is vacuous for bf16 at large
K; Higham & Mary give `√(n log n)·u`, sharpening toward `O(u)` for zero-mean data — so **the
correct tolerance depends on the generator's distribution, not just the dtype**. Separately,
NVIDIA tensor cores accumulate in round-toward-zero, a systematic non-cancelling bias that breaks
any √n-calibrated tolerance.

**Measured during Phase 1 implementation (2026-08-15).** Two results worth carrying forward:

*Metamorphic relations must be scaled to the failure mechanism, or they are vacuous.* A softmax
without max-subtraction is mathematically shift invariant; it fails only when `exp` overflows
(`x > 88.7` in float32). Shifts drawn from `N(0, 1)` reach at most ~4.7 over 100k draws, so a
unit-scale shift relation caught the unstable kernel **0% of the time** across 400 trials.
Scaling to `0.5·log(finfo(dtype).max)` gives an 11.8–18.0% catch rate with **0% false alarms**.
This is a concrete instance of the vacuous-property failure mode the project exists to study,
found in our own suite — evidence that "the property passed" is uninformative without a
demonstrated ability to fail.

*The normalization exponent decides whether the reference arm is a strong baseline or a
strawman.* The test ratio divides the residual by `eps` and by a function of the reduction
length. Which function matters enormously, measured on float32 softmax/layernorm against float64
references:

| normalization | correct-kernel ratio, n=64 → 16384 | detection floor at n=4096 | vs `allclose(rtol=1e-5)` |
|---|---|---|---|
| `n` (LAPACK's literal convention) | 0.038 → 0.0003 (127× drift) | 1.5e-2 | ~1500× blinder |
| `√n` | 0.306 → 0.040 (7.6× drift) | 2.3e-4 | ~23× blinder |
| **`log₂n`** | **0.408 → 0.368 (flat)** | **4.3e-5** | **~4× blinder** |
| `1` | 2.45 → 5.15 (2.1× drift) | 3.6e-6 | stricter |

Linear `n` — the literal LAPACK convention — would have made the reference arm miss bugs that
three lines of `np.allclose` catch: a denominator error of 0.3%, a float16 accumulator, a
dropped element. That would invert the paper's argument, since the reference arm exists to be a
*strong* baseline. `O(eps·log₂n)` is the textbook bound for **pairwise** summation, which is what
these backends actually do; linear `n` is the bound for sequential accumulation, a different
algorithm. A good normalization is identifiable by a correct-kernel ratio that is flat in n.

**Open question this raises.** The ratio and `allclose` do not measure the same thing — the ratio
is an infinity-norm-scaled backward-style measure, deliberately blind to relative error on tiny
entries; `allclose` is elementwise forward relative error. A small constant-factor gap is a
defensible design difference; a large one invites the objection that the reference arm is weaker
than standard practice.

**Measured outcome under `log₂n`.** On float32 softmax at n=4096 the reference arm now catches
every injected bug `allclose(rtol=1e-5)` catches — a dropped element (ratio 44.8), a float16
accumulator (123.7), a 1e-4 denominator error (70.0), a 0.3% denominator error (2090.9) — while a
correct kernel sits at 0.04, leaving 213×–779× headroom across n=8..16384. The baseline is no
longer weaker than standard practice on this corpus.

**Unresolved, recorded rather than reconciled.** Two independent measurement harnesses disagree on
which normalization is *flattest*: one finds `log₂n` flat with `n=1` drifting 2.1×, the other finds
`n=1` flattest (1.3×) and `log₂n` over-correcting (3.7×, in the conservative direction). Absolute
magnitudes differ by a consistent 5×, so a setup difference — reference construction or residual
scaling — remains unidentified. It does not change the choice: `log₂n` is the textbook pairwise
bound rather than a fit to either dataset, and is safe under both. It would matter if `n=1` were
revisited.

Whether to additionally report plain `allclose` as a fourth arm — pre-empting the "your baseline is
weak" objection entirely, at low cost — is undecided.

*Shift invariance is only approximate in reduced precision, and the tolerance dominates.* In
float16 a wide shift produces genuine deviations of 0.5–2 ulp (median 4.9e-4, p95 9.9e-4 against
`eps = 9.77e-4`). False-alarm rate is entirely tolerance-driven: 100% at `atol=1e-5`, ~2–4.5% at
`1e-3`, and 0% at `1e-2`. Repeating with fp16 inputs but exact float64 arithmetic leaves the curve
unchanged, isolating the cause as `x + c` destroying the row's mantissa detail rather than fp16
exp or accumulation. **Implication:** reduced-precision arms need an eps-scaled tolerance, not a
narrower relation — which is a specific, testable instance of the general tolerance argument
above.

### 5.4 Generator defaults are shape/dtype/backend, not adversarial values

Evidence: boundary shape sampling gives 78% recall at 0% false positives; adversarial NaN/Inf
value injection gives 99% recall at 94% FP. Corroborated by an ISSTA 2026 study of 301 real
Triton/TileLang bugs, which are tightly coupled to shapes, dtypes and backend targets.

Adversarial value families remain available but are opt-in per task, never the default.

## 5.5 Implementation phasing

This design is deliberately larger than one implementation plan. It decomposes into three
sub-projects, each of which gets its own plan and can be built and validated independently:

**Phase 1 — Core loop on CPU.** `InputDomain`, `Generator` with case groups, `ExecutionTable`,
CPU/NumPy `Backend`, `Property`, `Oracle` with three-valued verdicts, tier-1 properties for the
elementwise→reduction ladder. Fully testable with no hardware. Delivers: a working record/replay
pipeline and the reference and declarative arms on CPU.

**Phase 2 — Measurement.** `MutationCorpus`, `ExperimentRunner`, the four metrics, the shrinker.
This is the phase that produces the paper's numbers. Still CPU-only.

**Phase 3 — Device backends and tier 2.** CUDA/Triton backend, telemetry capture, tier-2
properties, then NKI. Introduces the batch-job boundary and the only components requiring hardware.

Phase 1 is the subject of the first implementation plan. Phases 2 and 3 are re-brainstormed
against what Phase 1 actually reveals.

## 6. Corpus

**Development ladder** (build and validate the instrument):
elementwise (relu, gelu) → reduction (rowsum, softmax) → normalization (layernorm) → fused.
Each rung introduces one new property class: pointwise → reduction order/associativity →
numerical stability.

**Then attention and GEMM** — where kernel work matters and where cross-backend translation is
interesting.

**KernelBench subset for reporting**, so headline numbers are comparable to published baselines.
Note its PyTorch-reference framing biases toward the reference arm; that bias must be stated.

## 6.5 A methodological finding from building the harness

The project's own fairness criterion — `REPLAY_FAIRNESS`, which certifies that competing oracle
arms score byte-identical executions — passed **vacuously through three successive fixes**. Each
fix was correct and each exposed the next layer:

1. The fingerprint covered `case.tensors` (the field named "input") but not `row.outputs`, which
   is what the arms actually score. A "repairing" arm flipped 7 of 9 detections while the
   criterion stayed green.
2. Fixed. The record-fidelity witness then covered inputs only, so a read path that repaired
   outputs identically on every call destroyed 7 of 7 detections — an identically-corrupting read
   is a fixed point.
3. Fixed. `row_fingerprint` still omitted `case.metadata()`. An arm that merely relabelled
   `case.relation` cost 14 of 14 detections, because `ShiftInvariance` finds its base and partner
   *by* that field — with every tensor byte untouched.

Separately, the "which assertion catches this?" defect appeared three times and **relocated rather
than recurred**: closing the gap on one arm's assertion made the other arm's deletable with zero
test failures, because the surviving assertion subsumed it.

**The invariant that actually works is not "every assertion has a saboteur" but "every assertion
is the *unique* catcher for at least one saboteur."** Operationally: parametrize saboteurs, pair
each with the exact message expected to catch it, and verify by deleting each assertion in turn
that precisely its own cases fail. Without the message pairing, a saboteur caught by an earlier
assertion silently certifies a later one it never reached.

Two things follow for the paper. First, this is direct evidence for the motivating claim that
"the property passed" is uninformative without a demonstrated ability to fail — observed in our
own instrumentation, not only in the kernels under test. Second, any detection-rate number
produced by this harness should be accompanied by the saboteur matrix that establishes the
harness could have failed.

## 7. Evaluation

Metrics, in the order they become measurable:

1. **Bug-catching power** — against a mutation corpus of deliberately-broken kernels. Scored as
   faults detected and cases-to-first-failure. Reported split by tolerance-free vs
   tolerance-dependent detection.
2. **False-positive rate** — how often a correct kernel is flagged. Includes the case where an
   authored property is simply not true of the op; this makes spec-authoring quality an observable
   in the same experiment rather than a separate study.
3. **Authoring cost** — effort to onboard a new task under each strategy: lines, time, token cost,
   how much can be auto-drafted.
4. **Downstream kernel quality** — end-to-end effect on generated kernel speed. Requires the full
   loop; deferred to a later phase.

Cost-per-bug is reported alongside detection, since an oracle's latency caps the search rate of
any agentic inner loop.

## 8. Execution environment

CPU-only development and CI; hardware acquired on demand and run in batches. This is why the
CPU/NumPy backend is first-class and why device execution sits behind an async batch-job boundary.

## 9. LLM role

**Deterministic arithmetic at the core, LLM at the edges.** Every oracle is mechanical and
reproducible — an LLM judging float correctness would be non-reproducible, which is disqualifying
when the oracle's own false-positive rate is a headline metric.

LLMs are used for:
- **Spec authoring** (compile-time): drafting declarative property sets for human review, after
  which they are frozen, versioned code. This is where the authoring-cost metric lives.
- **Kernel generation and translation** (downstream, out of scope for this slice).

## 10. Non-goals

- Kernel generation and translation loops (consumers of this layer)
- Any search or optimization loop
- Formal verification — the literature already wins where it applies; we do not compete there
- Multi-device distributed execution
- FP8 tolerance tables (documented in config later)

## 11. Consequences for the existing skeleton

- `harness/correctness.py`'s fixed five-stage pipeline is **replaced** by the property layer.
- `HarnessResult` schema and the run ledger **survive in modified form** — extended with
  per-property attribution, tier, tolerance_free, and telemetry.
- The candidate-search subsystem has already been removed (PR #2).

## 12. Open questions

1. Does the tolerance-free subset actually dominate in low-precision long-reduction regimes? This
   is the most interesting hypothesis available and is directly testable here.
2. Which declarative properties fail to transfer across CUDA / Triton / NKI, and why?
3. What is the right numerical characterization of Trainium? No published account of its rounding
   mode, accumulation order, or accumulator width exists; the tensor-core probe methodology gives
   a template. This is a prerequisite for any defensible NKI tolerance and is self-contained work.
4. Can shrinking be made to *preserve or maximize the violation ratio* rather than only minimize
   size? No prior art found.

## 13. Risks

| Risk | Mitigation |
|------|-----------|
| Reference arm wins outright, as in Hughes' study | Frame around fault-class and tolerance-free split, not a single count; the result is publishable either way |
| Competing preprints land first | Area moves ~1/month; re-run prior-art search before submission |
| Telemetry gaps found after an expensive run | Treat Phase B schema as the highest-risk interface; over-capture rather than under-capture |
| Custom generator/shrinker is the main engineering cost | Accepted deliberately; harvest Hypothesis corpora to avoid rebuilding float edge-case tuning |
