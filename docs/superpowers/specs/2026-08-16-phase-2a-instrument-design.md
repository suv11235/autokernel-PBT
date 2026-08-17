# Design: Phase 2a — completing the measurement instrument

**Date:** 2026-08-16
**Status:** Partly superseded — see §0. The decisions in §2 stand; the storage architecture in
§3.1–3.2 does not.
**Scope:** The instrument that Phase 2b's measurement runs on. Corpus, metrics, and the
experiment itself are explicitly out of scope and get their own design.

---

## 0. Status amendment (2026-08-16, after phase 1.5)

Phase 1.5 "measurable runs" landed independently and in parallel with this design, and built
much of §3 by a different route. This section records what survived and what did not, rather
than editing the document to look prescient.

**Superseded.** §3.1's append-only, one-file-per-writer layout and §3.2's argument that it
retires `REPLAY_FAIRNESS` as an assertion. Phase 1.5 chose the alternative the design
considered and rejected — kernel identity as a *column* on the execution row — and added
`corpus_fingerprint`, a per-write identity that makes a join between two different runs fail
loudly instead of silently reporting a rate about neither.

The consequence, verified empirically rather than read off the source: `run_dir` holds **one
kernel**. A second `write()` replaces the first kernel's rows, and two kernels in a single
write are rejected on the `case_id` payload collision. Multi-kernel comparison happens across
run directories, which is what `corpus_fingerprint` exists to make safe.

Replay fairness therefore remains a *checked* property rather than a structural one. That is
the known cost, and the argument in §3.2 for why checking it has repeatedly failed still
stands as a risk — it is now a risk the project carries knowingly rather than one it has
retired.

**Superseded.** The `ExperimentRunner`, kernel identity, and verdict persistence components of
§3.3, all delivered as `props/driver.py`, `props/table.py` and `props/scores.py`. Open
obligations 1, 2 and 4 are discharged.

**Stands.** Every decision in §2 — the fourth arm, the taxonomy-derived corpus and its 66/34
CPU-reproducible split, the shrinker interface without an algorithm, layernorm as the
normalization rung, and instrumented authoring cost. None of it was built by phase 1.5.

**Better than what this document claimed.** §3.3 assumed per-property verdicts could be
aggregated however the metrics phase liked. Phase 1.5 measured that they cannot: the same 14
detections give a rate of **0.778 or 0.222** depending on whether results or case groups are
the unit. Scores are keyed by case group for that reason, and this design's silence on the
question was a gap, not a simplification.

**Feature numbering.** This design reserved feature id 0005; phase 1.5 used it for "measurable
runs". The follow-on work is feature **0006**.

**Predecessor:** `docs/superpowers/specs/2026-08-14-kernel-property-oracle-layer-design.md`
(§5.5 Phase 2), `docs/superpowers/plans/2026-08-14-property-layer-phase-1.md`

---

## 1. Why this is a separate phase

The parent design names Phase 2 as "MutationCorpus, ExperimentRunner, the four metrics, the
shrinker" and says to re-brainstorm it against what Phase 1 revealed. Phase 1 revealed that
Phase 2 as scoped is two projects, not one, and that the second cannot start until the first
lands.

`CLAUDE.md` records five open obligations from the final Phase 1 review. Four of them are
prerequisites for measuring anything:

1. **No driver.** Nothing in `src/` composes generate → execute → persist → score. It is
   assembled twice in test files and the two already differ.
2. **Metrics are not computable from what is recorded.** The execution table has no
   kernel-identity column, and `PropertyResult` has no persistence path.
3. **The declarative arm, hybrid arm, and contract loader have no acceptance criteria.**
4. **`contract.py` is imported only by its tests** — the spec-as-oracle path is not exercised
   by production code.

Obligation 2 carries a deadline the others do not: it is cheap now and expensive after the
first paid hardware run, because a recorded run missing a column cannot be repaired offline.
That is the same argument §3.4 of the parent design makes for telemetry, and it applies here
for the same reason.

So Phase 2 splits:

| Phase | Delivers | Done when |
|---|---|---|
| **2a** (this design) | The instrument | Any kernel can be scored through four arms, every verdict persisted, byte-identical replay guaranteed by construction |
| **2b** | The measurement | Taxonomy-derived corpus, the four metrics, the numbers the paper reports |

## 2. Decisions taken

### 2.1 A fourth arm: plain `allclose`

Parent design §5.3 left this undecided. It is now decided: **yes**, reported alongside
reference, declarative, and hybrid.

The reference arm was deliberately strengthened into a LAPACK-style normalized test ratio so it
would not be a strawman, and Phase 1 measured it as catching everything `allclose(rtol=1e-5)`
catches at n=4096. That measurement is currently something a reader must take on trust.
Carrying `allclose` as a fourth arm makes it auditable, and pre-empts the "your baseline is
weaker than standard practice" objection entirely rather than answering it in prose. The cost
is one column per results table.

The arm is deliberately the field's naive default, unmodified: `rtol=1e-5`, `atol=1e-8`,
`tier=1`, `tolerance_free=False`. Improving it would defeat its purpose.

### 2.2 The corpus derives from a real-bug taxonomy

Phase 2b's mutants come from Rathnasuriya et al., *Characterizing Real-World Bugs in Tile
Programs for Automated Bug Detection* (ISSTA 2026, arXiv:2605.19652), already in the reference
library. This is a Phase 2b decision recorded here because it constrains 2a: the instrument
must be able to express the fault classes the taxonomy names.

The alternative — hand-authoring mutants from Phase 1's findings — was rejected on provenance.
The same intuition authored the properties and would author the bugs, and a reviewer is
entitled to discount any detection rate produced that way.

**Table 2 of that paper, with a CPU-reproducibility assessment added:**

| Category | Bugs | Share | Reproducible on the NumPy backend |
|---|---|---|---|
| Type and Operator | 147 | 48.8% | **Yes** — operator logic (80), dtype semantics (58), special-value handling (9) |
| Memory | 58 | 19.3% | **Partly** — indexing and stride (35) yes; resource allocation and cache ordering (23) no |
| IR Construction and Transformation | 49 | 16.3% | No — compiler-internal |
| Tile Mapping and Launch | 19 | 6.3% | No — device |
| Control Flow and Scheduling | 16 | 5.3% | **Yes**, as boundary-predicate and mask errors |
| Device-Specific | 12 | 4.0% | No |

Roughly **198 of 301 (66%) have a CPU-expressible analogue.** This is the honest scope
statement for a CPU-only measurement, and the complement is the concrete justification for
Phase 3: the remaining 34% is not reachable without device backends and tier-2 telemetry.

Two findings from that paper bear directly on this design and are recorded because they
corroborate choices previously made on weaker evidence:

- **F8/I8** — a single output mismatch stems from diverse root causes, so "no single oracle
  suffices for all bug types," requiring "diverse oracles — differential, metamorphic, canary,
  and algebraic." This is peer-reviewed support for the hybrid arm, which until now rested on
  assertion. It should be cited rather than re-argued.
- **F6/I6** — bugs concentrate at tile boundaries and non-divisible extents, and
  "shape-dependent bugs are missed by uniform random manual testing." This corroborates the
  shape-first generator of parent design §5.4, which was resting on a solo-author unrefereed
  preprint.

### 2.3 The shrinker: interface now, algorithm deferred

`reference/PBT-property-based-testing/NOTES.md` §5.3 records the spirv-fuzz result: if
metamorphic transformations are small and independent, plain delta debugging over the
*transformation sequence* gives reduction for free — and this is an architectural decision
that must be made up front, not retrofitted.

Phase 2a therefore ships the reducible representation and **no shrinking algorithm**. A `Case`
becomes regenerable from a `CaseSpec`, and shrinking later means delta-debugging a list. This
honors the "up front" constraint at near-zero cost while keeping 2a focused.

Shrinking the *tensor* is explicitly rejected. Parent design open question 4 — whether
shrinking can preserve or maximize the violation ratio rather than only minimize size — stays
open and is not pursued here.

### 2.4 The ladder gains layernorm

Tasks become relu, softmax, **layernorm** — the normalization rung named in parent design §6.
It introduces a property class the current library has no instance of (mean-zero,
unit-variance, scale-shift equivariance) and a second numerical-stability story independent of
softmax's.

Fused kernels, attention, and GEMM remain out of scope, as does rowsum.

### 2.5 Authoring cost is instrumented, not reconstructed

Parent design §7 metric 3 is authoring cost. Onboarding layernorm is the only clean opportunity
to measure it, and only if the protocol is set up *before* the task is authored — measuring it
afterwards means reconstructing effort from memory, which is worthless.

Before layernorm is written, record separately for its reference implementation and its
declarative property set: wall-clock time, lines of spec, token cost, and what fraction was
auto-drafted versus hand-corrected.

`n=1` is weak and the paper must say so. It is nonetheless pre-registered and honest, and it
costs almost nothing when done in the right order. Phase 2b may extend it.

## 3. Architecture

### 3.1 Append-only storage, one file per writer

The current `ExecutionTable` writes one `rows.parquet` plus a tensor file per case, for a
single implicit kernel. Phase 2 needs many kernels scored over identical inputs, and needs
per-property verdicts to survive.

```
run_dir/
  cases.parquet                        written once
                                       case_id, group_id, relation, task_id, dtype,
                                       shape, case_spec
  inputs/<case_id>.safetensors         written once — ONE copy, shared by every kernel
  executions/<kernel_id>.parquet       one file per kernel: case_id, status, error, telemetry
  outputs/<kernel_id>/<case_id>.safetensors
  verdicts/<kernel_id>.<arm>.parquet   one file per (kernel, arm): property_name, tier,
                                       tolerance_free, verdict, detail, case_id | group_id
```

Every file is written exactly once and swapped atomically. Nothing is ever appended to or
read-modify-written. Three consequences follow, and they are the reason for the layout:

**Phase 1's atomicity contract survives unchanged.** `CLAUDE.md` records that the execution
table is never observed torn — index and payloads swap atomically, a crash may lose the table
but must never mix runs. A layout that appended each kernel's rows into one shared Parquet
file would have broken that. One file per writer preserves it trivially.

**Adding a kernel or an oracle is a pure append.** Scoring a fourth oracle strategy months
later without touching a device is the property parent design §3.1 exists to buy; under this
layout it writes new `verdicts/` files and touches nothing that exists.

**Replay fairness stops being a claim.** See below.

### 3.2 Why this retires REPLAY_FAIRNESS as an assertion

`REPLAY_FAIRNESS` certifies that competing arms score byte-identical executions. Parent design
§6.5 records that it **passed vacuously through three successive fixes** — the fingerprint
covered `case.tensors` but not `outputs`; then the record-fidelity witness covered inputs only,
so an identically-corrupting read path was a fixed point; then `row_fingerprint` still omitted
`case.metadata()`, and an arm that merely relabelled `case.relation` cost 14 of 14 detections
with every tensor byte untouched.

Each fix was correct. The pattern is that the criterion is structurally hard to make true by
checking, because every check has a surface it does not cover.

Under this layout there is **one copy of each input tensor on disk**. Two arms cannot see
different inputs because there are not two inputs to differ. The guarantee moves from a
fingerprint comparison into the shape of the filesystem.

The criterion is retained, but its check changes to something much harder to satisfy
vacuously: **no code path writes under `inputs/` after the generation stage.** That is a
statement about the program, not about a pair of values that happened to match.

### 3.3 Components

| Component | Responsibility | New or changed |
|---|---|---|
| `CaseSpec` | `(seed, task_id, shape, transform_sequence)`; a `Case` is regenerable from it alone. Persisted in `cases.parquet`. | New |
| `KernelVariant` | `kernel_id` plus a content hash of the kernel source, so two runs cannot disagree about what `mutant_07` was. | New |
| `ExecutionTable` | Rewritten for the layout in §3.1. Reads and writes cases, executions, and verdicts. | Rewritten |
| `AllcloseOracle` | The fourth arm. | New |
| `VerdictTable` | Persistence for `PropertyResult`, closing obligation 2. | New |
| `ExperimentRunner` | The driver: generate → execute → persist → score → persist. Closes obligation 1. | New |
| `layernorm` task | Reference, contract, property set. Authoring instrumented per §2.5. | New |

### 3.4 Data flow

```
InputDomain + seed
  → Generator → CaseSpec[] → Case[]                        Phase A, once per (task, seed)
  → cases.parquet + inputs/*.safetensors

  for each KernelVariant:
      Backend.run over every case                          Phase B
    → executions/<kernel_id>.parquet + outputs/<kernel_id>/*

  for each (KernelVariant, arm):
      Oracle.evaluate over the recorded rows               Phase C, offline, no backend
    → verdicts/<kernel_id>.<arm>.parquet
```

Phase C never touches a backend. That is what makes a recorded run a reusable dataset.

### 3.5 Error handling

Unchanged from Phase 1, and the existing module contracts are load-bearing rather than
incidental:

- Bad **data** → `INCONCLUSIVE`; bad **call** → raise. The line is whether a re-run costs
  hardware time.
- A failed execution persists with its status and no outputs. All four arms must score it
  `INCONCLUSIVE`, never `FAIL` — booking a launch error as a caught bug is precisely what makes
  a false-positive rate meaningless.
- `ExactDtypeError` stays caught narrowly and mapped to `INCONCLUSIVE`.
- Every `PropertyResult` carries exactly one of `case_id` / `group_id`. The verdict schema must
  preserve that, because `HybridOracle` concatenates arms and the split point is not
  recoverable from a flat list.

## 4. Testing

Beyond the round-trip and unit coverage Phase 1 established:

1. **Bitwise round-trip across the three-file layout.** The single most important test in
   Phase 1 was that rows survive persistence byte-identically; it remains so with more files.
2. **Input immutability.** No write path touches `inputs/` after generation. This replaces the
   fingerprint check and is the real content of `REPLAY_FAIRNESS` under the new layout.
3. **Saboteur matrix.** Per `CLAUDE.md`, every new assertion must be the *unique* catcher for
   at least one saboteur — parametrized, each paired with the exact expected message, verified
   by deleting each assertion in turn and confirming precisely its own cases fail. "Every
   saboteur is caught" is too weak; that defect appeared three times in Phase 1 and relocated
   each time it was fixed.
4. **Acceptance criteria for the declarative arm, hybrid arm, and contract loader**, closing
   obligation 3. The nine existing criteria cover infrastructure only.
5. **A production code path that loads a contract**, closing obligation 4. `ExperimentRunner`
   building its declarative arm from `acceptance.yaml` is that path.

## 5. Costs and risks

| Risk | Assessment |
|---|---|
| Rewriting `ExecutionTable` invalidates a recorded corpus | **Verified not to apply.** Every `ExecutionTable` construction in the tree is in a test, under `tmp_path`; no `rows.parquet` or `.safetensors` exists on disk outside the reference library. A rewrite is safe; no migration path is needed. |
| `test_table.py` must be rewritten | Real, accepted. ~500 lines of well-targeted tests, many of which port directly. The saboteur discipline must be reapplied to the new assertions rather than assumed to carry over. |
| More files per run raises per-run overhead | Accepted. Runs are batch jobs measured in minutes; file count is not the bottleneck, and the atomicity and fairness properties are worth more. |
| The degenerate ladder rungs still deflate absolute detection | Known and asserted, not newly introduced. `(1, 1)` and `(17, 1)` make softmax identically 1.0, so ~22% of groups score any kernel clean. It deflates every arm equally, so arm-vs-arm stays unbiased; the paper must state the constant. Layernorm inherits the same ladder and needs the same analysis. |
| `n=1` authoring-cost measurement is weak | Accepted and stated. Pre-registered and honest beats reconstructed. |

## 6. Out of scope for Phase 2a

- The mutation corpus itself, and any detection or false-positive number (Phase 2b)
- The four metrics and their reporting (Phase 2b)
- Any shrinking *algorithm* — only the reducible representation ships
- CUDA / Triton / NKI backends and tier-2 telemetry (Phase 3)
- Attention, GEMM, fused kernels, rowsum; KernelBench integration
- Retiring `harness/correctness.py`. Parent design §11 says the property layer replaces its
  five-stage pipeline, but it is load-bearing for features 0001 and 0002 and their acceptance
  criteria. Retiring it means retiring a feature, which is a scope decision of its own and is
  not smuggled into this phase.

## 7. Open questions

1. Does `layernorm` need a metamorphic relation the current `RELATIONS` registry lacks? Scale
   and shift equivariance suggest yes, and a new relation must be scaled to its failure
   mechanism or it will be vacuous — the Phase 1 finding that a unit-scale shift caught the
   unstable softmax 0% of the time is the cautionary case.
2. Should `CaseSpec` capture the *distribution* as well as the transform sequence? Regenerating
   a case from its spec requires it; storing it duplicates `InputDomain`. Resolve during
   planning.
3. Does the ladder need tile-multiple and non-divisible-extent shapes that F6/I6 specifically
   names, beyond the powers-of-two and odd-remainder rungs already present? Probably yes for
   Phase 3, possibly already covered on CPU where there is no tile.
