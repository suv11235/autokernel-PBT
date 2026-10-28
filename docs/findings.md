# Findings: where property-based testing and formal verification paid off

A ranked summary of what this repository has established, for readers who want the results
without reading every record. Each row links to the dated record under
[`docs/measurements/`](./measurements/) it comes from, and the paper draft quotes the same numbers
through [`paper/numbers.tex`](../paper/numbers.tex). When this page and a record disagree, the
record is right.

**Status tags:** **[M]** measured and recorded · **[B]** built and tested, not yet measured ·
**[P]** probe reproduced by a script but not yet recorded · **[planned]** pre-registered, not run.

**Segments:** **G** autokernel generation (an agent writes or optimizes a kernel) · **T**
autokernel translation (an agent ports a kernel between backends) · **G+T** both. See
[`pipelines/`](./pipelines/) for how each result feeds a pipeline.

**Scope of every device number:** one NVIDIA A10 (sm_86), Triton 3.3.0, PyTorch 2.7.0, float32,
seed 42, except result 9, which ran on one NVIDIA RTX A6000 (also sm_86) with the same versions. The optimization loop ran on CPU (NumPy kernels on an Apple M4 Pro). Rates are per
*case group*. A kernel that is correct on the ladder's two single-column shapes can fail at most
7 of 9 groups, so absolute detection rates are understated. Comparisons between arms are not
affected.

---

## At a glance

| # | Gain | Method | Seg. | Headline number | Status |
|---|---|---|---|---|---|
| 1 | A gate calibrated well enough to run a loop at all | PBT | G+T | `allclose` rejects **10/54** correct-kernel groups (0.185); the reference, declarative and hybrid arms reject **0/54** | [M] |
| 2 | A cheap gate made an optimization loop safe | PBT | G | **−21.2% val_bpb** at **+27.7% throughput** over 26 candidates; 6 wrong optimizations caught, the 3 timed ones 1.11–2.40× faster than correct | [M] |
| 3 | Laws ported across backends with no second implementation | PBT | T | Tier-1 laws moved NumPy → Triton **unchanged**: no false failure from 52 executions up to 29.4 M elements at 78% of peak bandwidth | [M] |
| 4 | A gap in the laws became an exact characterization with a one-law fix | PBT + FV | G+T | The gate admits wrong kernels: **4** recorded softmax families [M], including a constant fill **52–54× faster** than softmax, and **6** more from a symmetry probe [P]. Proposition 1 characterizes the admitted set, and one added law per task closes it; the core is machine-checked in Lean (6 theorems) | [M], [P] |
| 5 | Verifier axioms carry the same blind spot, and an audit finds it | FV | G+T | All **10** axioms of Why3's `real.ExpLog` hold for `exp(βx)`, so a Why3 softmax certificate holds at *some* temperature (9 theorems). A sampled audit finds the same symmetry in seconds per theory | [M] |
| 6 | A certificate core that catches spec gaming by execution | FV × PBT | G (T next) | **6** checks, each the only catcher of one cheat; 20 conditions deleted one at a time, and each deletion was caught | [B] |
| 7 | The reference arm's tolerance, derived rather than guessed | PBT | G+T | LAPACK's linear-`n` ratio drifts **127×** and is **1500×** blinder than `allclose`; `log2 n` catches everything `allclose` does, with **213–779×** headroom | [M] |
| 8 | A metamorphic partner caught a "correct" kernel that was wrong | PBT | G | `E[x²]−E[x]²` layernorm passes plain inputs (ratio < 0.2) and fails the shifted partner (**48.1** against a threshold of 30) | [M] |
| 9 | A port judged against the kernel it came from, with no false alarm | PBT | T | Triton ladder → CUDA on one device: the port differs bitwise on **55–75%** of reduction elements, yet T1 (`reference_source`) and the Tier-1 laws reject **0** groups, with T1's largest ratio **0.986** against 30. T0 (`allclose_source`) rejects **2/9** layernorm groups | [M] |

The one-line version: **PBT's gain is calibration, not detection. FV's gain is completeness. Each
checks the other.** The law bundle and the normalized reference catch the same bugs as `allclose`
without its 18.5% false-alarm rate, and the laws need no reference at all, except where a bundle
is incomplete (result 4). Proofs say exactly which wrong kernels a law set admits, and the
certificate checks run the formal artifacts' claims on PBT's cases.

---

## Property-based testing

### 1. Calibration: the gate can accept correct kernels  [M] · G+T

The test population was six correct kernels. Three are rewrites an optimizing author would write
(`exp2`, reciprocal-multiply, `rsqrt`), and each was verified correct by the admission gate
rejecting it as "not broken":

| arm | false positives |
|---|---|
| `allclose` (rtol 1e-5, atol 1e-8: the field default) | **10/54 = 0.185**, all on layernorm, **5/9** on each of two independent kernels |
| reference (normalized test ratio) | 0/54 |
| declarative (law bundle) | 0/54 |
| hybrid | 0/54 |

The cause does not depend on the backend. `atol` sits about 12× below float32 eps, and a layernorm
row is centered on zero, so some of its elements get only `atol` as an error budget. The same
failure shows up in the optimization loop (result 2), measured separately on NumPy:
`allclose` rejects the loop's own correct baseline layernorm on 5 of 9 groups, so an
`allclose`-gated loop never completes iteration 0. The choice of oracle bounds the frontier before
search starts. Translation is likely to make this worse, because two backends that round
differently produce exactly the "correct but different" outputs `allclose` misjudges.
Records: [`2026-08-19-false-positive-rate.md`](./measurements/2026-08-19-false-positive-rate.md),
[`2026-08-16-allclose-layernorm-false-positives.md`](./measurements/2026-08-16-allclose-layernorm-false-positives.md),
[`2026-08-18-autoresearch-loop-pbt-gate.md`](./measurements/2026-08-18-autoresearch-loop-pbt-gate.md) §4.

### 2. A property-gated optimization loop  [M] · G

karpathy/autoresearch's keep/revert loop was transplanted to a NumPy byte-level GPT with a fixed
60 s budget. The model's `softmax` and `layernorm` are gated by all four arms before `val_bpb` is
computed, and the gate decides on the declarative arm. Over **26 candidates in 6 rounds**,
`val_bpb` fell **2.829 → 2.228 (−21.2%, 3 seeds each)** at **+27.7% throughput**. The gate is one
small module that runs on every candidate before training. **Six saboteurs**, each a plausible
optimization, were caught by every arm: no max-subtraction, a float16 accumulator and a dropped
column on softmax; no centering, no divide, and sample instead of population variance on layernorm.
Three were timed, at 1.11×, 1.76× and 2.40× faster than the correct kernel, which shows why a
speed objective needs the gate.
Record: [`2026-08-18-autoresearch-loop-pbt-gate.md`](./measurements/2026-08-18-autoresearch-loop-pbt-gate.md) ·
code: [`experiments/generation/autoresearch-pbt/`](../experiments/generation/autoresearch-pbt/).

### 3. Cross-backend portability of the laws  [M] · T

This is the premise the translation segment rests on, and the first evidence for it. Tier-1 laws,
which are pure functions of (inputs, outputs), transferred from the NumPy backend to Triton on an
A10 **with no changes**. No law failed on a correct device kernel, from the first run's 52
executions up to 29.4 M elements at 78% of the A10's peak bandwidth. At that scale, deriving the
tile width turned up a launch defect: a tile narrower than its row silently dropped the row's
tail, and rows summed to 1.51. The record notes that the oracle layer would have caught it had
any shape reached it, and the launcher now refuses that configuration. Every run was scored
offline on a CPU, so new oracles can be scored months later without a device.
Records: [`2026-08-17-first-hardware-run.md`](./measurements/2026-08-17-first-hardware-run.md),
[`2026-08-17-at-scale-hardware-run.md`](./measurements/2026-08-17-at-scale-hardware-run.md).

### 7. A reference tolerance with a derivation  [M] · G+T

The reference arm's test ratio is `‖y − ref‖∞ / (‖ref‖∞ · eps · max(log2 n, 1))`, a
pairwise-reduction bound, rather than LAPACK's literal linear `n`. Linear `n` drifts 127× over
n = 64–16384 on correct kernels and is 1500× blinder than `allclose`. `log2 n` catches every bug
`allclose` catches at n = 4096, with 213–779× headroom on correct kernels. On a multi-tile kernel
the tile-aware bound `log2(tile) + n_tiles` drifts 3.5×, against 5.9× for `log2 n`. It is not
adopted yet: one kernel, one GPU. A cross-backend translation tolerance will need the same kind
of derivation for the target's accumulation order.
Records: design §5.3, [`2026-08-17-hybrid-reduction-tree.md`](./measurements/2026-08-17-hybrid-reduction-tree.md),
[`2026-08-17-normalization-discrepancy-resolved.md`](./measurements/2026-08-17-normalization-discrepancy-resolved.md).

### 8. Metamorphic case groups  [M] · G

`layernorm_sumsq`, which computes the variance as `E[x²] − E[x]²`, was written as a *correct*
variant. Its test ratios on plain inputs stay below 0.2. On the **shifted** partner, where the
row mean is large and the two terms nearly cancel, they reach 34.3, 48.1 and `inf` against a
threshold of 30. The catch belongs to the transform, not to any one arm: all four arms see it
because the case group put the shifted input in front of them.
Record: [`2026-08-19-false-positive-rate.md`](./measurements/2026-08-19-false-positive-rate.md) §3.

### 9. Translation: a correct port, judged against the known kernel  [M] · T

TE1 mapped the known Triton ladder to CUDA C++ and recorded both setups on byte-identical inputs
in one session, on one RTX A6000. The port is correct, and it is not the same computation: only
25–45% of the output elements of each reduction task are bitwise equal to the known kernel's,
because the two kernels reduce in different orders. Judged against the known kernel's recording,
the normalized ratio (T1) rejects 0 groups. Its largest ratio is 0.986, on layernorm, where each
kernel's own largest distance from float64 is 1.00. The Tier-1 laws also reject 0 groups, so the laws transfer to
a second backend unchanged, as they did NumPy → Triton (result 3). `allclose` against the known
kernel (T0) rejects 2 of 9 layernorm groups, the translation analogue of result 1. All eight
pre-registered predictions hold. One of them, P2, held only after a telemetry probe was fixed
before any row was recorded. Not established: detection of wrong ports (TE3) and any agent doing
the porting (TE4).
Record: [`2026-10-12-te1-triton-to-cuda-prereg.md`](./measurements/2026-10-12-te1-triton-to-cuda-prereg.md).

---

## Formal verification

### 4. From "the gate admits wrong kernels" to an exact characterization  [M], [P] · G+T

Every law in the softmax bundle (finite, in [0, 1], rows sum to one, shift invariance) holds for
`softmax(βx)` at **every** real β. The gap was found by reasoning from the bundle's invariances,
not by the optimization loop, none of whose 26 candidates was a temperature kernel. The gate
decides on the declarative arm, so it admits:

- **[M]** four recorded softmax kernels: softmax(2x), softmax(0x) = 1/n, a constant 1/n fill, and
  softmax(−x). `allclose`, the reference arm and the hybrid arm reject each on 7 of 9 groups, the
  most any arm can score on this ladder. The constant fill runs **52–54× faster** than the real
  softmax, so a loop optimizing kernel speed against this gate would converge to it.
- **[P]** six more from a symmetry probe that is reproduced by a script but not yet recorded:
  rolled rows and reversed columns on softmax, and −LN, reversed columns, sorted rows and rolled
  rows on layernorm.

**Proposition 1** characterizes the solution sets exactly. The softmax bundle admits precisely the
shift-invariant maps into the simplex. Adding the **log-ratio law** (`log y_ij − x_ij` constant
along a row) pins softmax, and the **affine-tie law** pins layernorm. Both completing laws tie
outputs to inputs, while every law in both bundles reads outputs alone. That gives a cheap
syntactic test for any drafted bundle. **Part (b), and part (a) for the temperature family, are
machine-checked in Lean 4 without Mathlib**: 6 theorems, 367 lines, resting on Lean's standard
axioms only. The permutation members of (a) and all of part (c) are proven on paper only.

Mechanizing it found two things the paper proof did not say. Completeness uses **no property of
`exp`**. And the natural axioms of `exp` (positive, additive, injective) are **scale-invariant**,
so a proof from them is temperature-blind in exactly the way the test bundle is. The tangent
axiom `1 + x ≤ exp x` restores rigidity. That became a design requirement: a rigidity audit of
every primitive an agent axiomatizes.
A float form of the log-ratio law is measured offline in
[`2026-10-14-log-ratio-headroom-prereg.md`](./measurements/2026-10-14-log-ratio-headroom-prereg.md): at 30 eps it accepts every correct softmax we
hold, the A10 runs included (at most 12.3), and rejects every wrong output (at least 2.67e7).
The device spread grows with each row's input range, so the headroom on wider inputs is open.
E2's softmax half adds it to the gate as `rows_have_constant_log_ratio`, in a range-aware float form
(feature 0014): [`2026-10-14-e2-log-ratio-law-prereg.md`](./measurements/2026-10-14-e2-log-ratio-law-prereg.md).
The completed gate rejects softmax(βx) at β = 2, 0 and −1, the 1/n fill, rows rolled and columns
reversed. It raises no new false positive on the 26 loop candidates, the 21 recorded A10 groups, or
464 CPU cases with row ranges in the hundreds, underflow and −∞ masks (worst 1.38 against 30).
Fast-exp is admitted on the gate's N(0, 1) domain (17.2), but the law rejects it on rows that
underflow or carry −∞ masks: its argument floor leaves 1.18e-38 where softmax underflows.
Records: [`2026-10-01-softmax-temperature-blind-spot.md`](./measurements/2026-10-01-softmax-temperature-blind-spot.md),
[`2026-10-01-softmax-bundle-mechanized.md`](./measurements/2026-10-01-softmax-bundle-mechanized.md) ·
proof: [`proofs/softmax-bundle/`](../proofs/softmax-bundle/).

### 5. A deployed verifier has the same symmetry  [M] · G+T

Why3's `real.ExpLog` axiomatizes `exp`, and Frama-C/WP reasons from it; SPARK/GNATprove was not
audited. **All ten of its axioms hold for `(exp(βx), log(y)/β)` for every β > 0.** A Why3
certificate that code computes softmax therefore certifies softmax at *some* positive
temperature. The theory excludes β ≤ 0, and with it the uniform and reversed rows, but keeps
exactly the temperatures the training loop cannot see either. `real.Square` is categorical on
x ≥ 0, so a layernorm certificate does exclude −LN. `Trigonometry` admits a discrete frequency
alias (cos 5x, sin 5x), which matters for rotary embeddings. The result is 9 theorems in core
Lean, plus a language-neutral sampled audit that finds the same symmetries in seconds per theory.
The sampler also showed that check C6 as designed would falsely reject true axioms on up to 2.4%
of samples near zero, so C6 needs a floored comparison. A language that *defines* `exp`, such as
Lean with Mathlib, does not have this symmetry; it comes from axiomatizing.
Record: [`2026-10-02-why3-real-theories-symmetry.md`](./measurements/2026-10-02-why3-real-theories-symmetry.md) ·
proof: [`proofs/why3-real-theories/`](../proofs/why3-real-theories/) ·
audit: [`experiments/verification/formal-rigidity/`](../experiments/verification/formal-rigidity/).

### 6. Certificates whose claims are executed  [B] · G, then T

The agent writes everything: the spec, a model of the kernel, the hypotheses and the proof. A
checked proof is therefore a *claim*, and the harness tests it. C1 is the language's own checker.
C2–C5 run on PBT's own recorded cases with harness-owned tolerances, and C6 samples the axioms:

| check | rejects |
|---|---|
| C1 proof | a proof with an escape hatch (`sorry`, `Admitted`, `assume`, …) |
| C2 spec soundness | a spec the reference fails, such as one that restates the kernel bit for bit |
| C3 spec completeness | a spec that admits a hidden wrong kernel, or the kernel in hand by name |
| C4 model fidelity | a model that does not compute what the real kernel computed |
| C5 non-vacuity | hypotheses that are false on the cases actually run |
| C6 axioms | assumptions that are false, or false under their intended reading (`exp(x) = 1 + x`) |

Each check is the only catcher of at least one cheat, and deleting any of the 20 conditions one at
a time is caught by the tests. Review found three ways to certify a wrong kernel, each now closed
and pinned by its own cheat test. Proofs are **post-hoc certificates on kernels PBT already
accepted, never the loop's filter**: a hard proof gate is a known performance trap (0.24×
best-provable against 1.03× best-tested on GEMM, as cited in the paper). Only the language-neutral
core is built, and it is tested against a fake adapter. Per-language adapters (0010) and the
7-language pilot (0011, E1) come next.
Design: [`2026-10-01-agent-formal-adherence-design.md`](./superpowers/specs/2026-10-01-agent-formal-adherence-design.md) ·
code: [`src/autokernel_pbt/formal/`](../src/autokernel_pbt/formal/) ·
spec: [`specs/features/0009-formal-core/`](../specs/features/0009-formal-core/).

---

## Where they did not help (reported, not buried)

- **Detection is a tie.** Across **20 admitted mutants** from three independently authored corpora,
  plus 6 saboteurs, the four arms **never disagree**. Where a cheap, trusted reference exists, laws
  can at best tie it. Tolerance-free detection, meaning a catch with no tolerance in the decision,
  is rare: 3 of 11 in the largest corpus. The laws' case is the same bugs *without a reference*,
  which is the translation setting.
  [`2026-08-18-arm-differentiation-null-result.md`](./measurements/2026-08-18-arm-differentiation-null-result.md)
- **The drafted law bundles are incomplete** (result 4). On those families the declarative arm
  misses what the reference catches, and until E2 lands the gate admits them. A **fast-exp
  softmax**, with 40× the exact kernel's relative error, passed every arm. It was stopped by being
  slower, not by the gate.
- **Zero in-loop rejections measures compliance, not gate strength.** The loop's instructions name
  the cheapest reward hacks, and no agent used them. Gate strength comes from saboteurs, or from E3.
- **No agent-written kernel proof has been scored yet.** The prover-in-the-toolkit experiment ran
  on 2026-10-02, but only 11 of 24 agents' self-reports survived and none has been evaluated
  ([`experiments/verification/oracle-research/`](../experiments/verification/oracle-research/)).
- **Threats to validity:** one GPU model, float32 only, seed 42; softmax-only corpora of at most 15
  mutants each; the authoring cost of layernorm is `n = 1`; the families in result 4 were found by
  authors who knew the bundles, so they show the gate *can* fail, not how often it does.

## What is next

Pre-registered in the paper (§5) and [`experiments/README.md`](../experiments/README.md): **E1**
agent-written adherence proofs across 7 languages, **E2** complete the bundles and re-gate, **E3**
red-team the gate with a speed objective and a hidden checker, **E4** return formal feedback to the
agent. Translation: **TE1** has run (result 9); **TE2–TE4** are proposed in
[`pipelines/translation.md`](./pipelines/translation.md).
