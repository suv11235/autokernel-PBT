# Design: Phase 2b — the mutation corpus and the metrics

**Date:** 2026-08-17
**Status:** Approved (design); implementation plan pending
**Scope:** An agent-authored mutation corpus derived from the ISSTA fault taxonomy, scored on
both backends, and the three metrics that are measurable without the full agentic loop.

**Parent design:** `docs/superpowers/specs/2026-08-14-kernel-property-oracle-layer-design.md` §7
**Depends on:** phases 1, 1.5, feature 0006 (four arms), feature 0007 (Triton backend)

---

## 1. What this phase produces

Parent design §7 metrics 1–3, reported **per fault class and per backend**:

1. **Detection rate** — the fraction of case groups on which an arm reaches FAIL against a broken
   kernel, split by `tolerance_free`. This is the headline claim: bugs found without a tolerance
   argument.
2. **False-positive rate** — the same measurement on *correct* kernels.
3. **Authoring cost** — extending the `n = 1` layernorm measurement in
   `docs/measurements/2026-08-16-layernorm-authoring-cost.md`.

Metric 4, downstream kernel speed, needs the full generate-and-improve loop and stays deferred.

Everything below exists to make those three numbers mean what they say.

## 2. The corpus is agent-authored, and that is a design decision

The obvious approach is to hand-write the broken kernels. That was the original plan and it is
wrong for this project, for two reasons.

**It is the wrong threat model.** This project is about *automated* kernel development. The bugs
that matter are the ones a code-generating model actually produces, not the ones a human imagines
it might. An agent-authored corpus samples the real distribution; a hand-authored one samples the
author's priors about it.

**It removes a provenance objection that would otherwise be fatal.** If the same person authors
the properties and the bugs, the declarative arm wins by construction — a reviewer is entitled to
discount any detection rate produced that way, and would be right to. Agent authorship decouples
them *provided the agent is blinded*, which §3 makes precise.

Scope: one mutant per (task, CPU-reachable fault subcategory) pair where the pairing is
meaningful, in both a NumPy and a Triton variant. Roughly 10–12 mutants, so ~20–24 broken kernels.

## 3. Blinding

The authoring agent receives **only**:

- the ISSTA subcategory description for the fault class it is asked to model, and
- the correct reference implementation for the task.

It does **not** receive `acceptance.yaml`, the property set, the reference arm's tolerance, or
any part of the oracle layer.

This is the control that makes a detection evidence rather than an artifact of shared authorship.
A mutant written with the property checklist in hand tells you only that the checklist matches
itself.

Each mutant records the taxonomy row it was asked to model, so fault-class reporting falls out of
the corpus rather than being reconstructed afterwards.

## 4. The validation gate

Agent-authored mutants cannot be taken at their word. Three failure modes, each corrupting a
different number, and each invisible without an explicit check:

| Failure | Consequence if unchecked |
|---|---|
| The mutant is actually **correct** | It enters the detection denominator as a bug nobody can catch. Every arm's rate drops, for free, and the corpus looks harder than it is. |
| It is broken in a **different class** than intended | Per-class rates are wrong while the total stays plausible — the hardest kind of error to notice. |
| It is **catastrophically** broken | Every arm returns INCONCLUSIVE, the driver refuses the run as an arm that established nothing, and no scores are written at all. |

So every candidate passes a gate before it is scored. To be admitted, a mutant must:

- differ from its reference beyond tolerance on **at least one** case group, and
- agree on **at least one** case group, or otherwise not fail everywhere for a trivial reason, and
- return `Status.OK` on enough cases to be judgeable at all.

Rejected candidates are **recorded with their rejection reason**, not silently dropped. The
rejection rate is itself a finding: it says what proportion of an agent's attempts at a named
fault class are not that fault, which is a fact about code-generating models and costs nothing
extra to collect.

**What the gate deliberately does not do.** It does not verify the fault *class*. Classifying a
defect automatically is a research problem of its own, and a weak classifier would silently
mislabel exactly the cases that matter. The class is recorded as **intended**, established by
construction from the prompt, and the paper must say so. The gate checks only that the mutant is
genuinely and judgeably broken.

## 5. The correct-kernel set

The false-positive denominator is currently one correct kernel per task, which is too thin to
carry metric 2.

It is widened with **correct-but-different** implementations: softmax with and without float64
widening, layernorm computing the variance by two algebraically equivalent routes. These are the
realistic false-positive risk — a kernel that is right but differs from the reference in the last
few ulps — and the shape of that risk is already measured: `allclose` flags 5 of 9 layernorm
groups for exactly this reason
(`docs/measurements/2026-08-16-allclose-layernorm-false-positives.md`).

A correct kernel that is bit-identical to the reference tests nothing, because every
tolerance-bearing arm is handed a residual of exactly zero.

## 6. Metric definitions

**The unit is the case group**, not the case. This is settled and measured: per-result rates
differ 0.778 against 0.222 for the same 14 detections, and `ScoreTable` already refuses a row
without a `group_id`.

- **Detection rate** = groups where `summarize(arm results) is FAIL` ÷ groups scored, per
  (arm, mutant, backend).
- **Tolerance-free detection** = the same, counting only groups where at least one FAIL came from
  a result with `tolerance_free = True`. This is the project's sharpest claim and needs its own
  numerator.
- **False-positive rate** = the same computation over the correct-kernel set.
- **Cases-to-first-failure** = the index of the first group, in generation order, on which the arm
  reaches FAIL. Generation order is deterministic from `(seed, index)`, so this is reproducible.

## 7. Reporting

A fault-class × arm × backend table, with three things stated **next to the numbers** rather than
in a footnote:

- **The ladder deflation.** Open obligation 3: `(1,1)` and `(17,1)` make softmax identically 1.0
  and make layernorm's variance property abstain, so every absolute rate is understated by a
  constant measured at 7/9 = 0.778. Arm-vs-arm stays unbiased; the absolute number does not.
- **The intended-class caveat** from §4.
- **The corpus size**, per class. With ~10–12 mutants a per-class rate rests on one or two
  kernels, and is indicative rather than statistical.

## 8. Out of scope

- Metric 4, downstream kernel quality — needs the full loop
- Shrinking; `CaseSpec` ships the representation and no algorithm
- Tier-2 properties and `compute-sanitizer` (Phase 3b)
- NKI, and any third backend including Apple Silicon — an M-series Mac is a genuinely useful third
  substrate for tier-1 transfer, but Triton does not target Metal, so it needs its own backend and
  exercises a different fault taxonomy
- Adopting `log2(tile) + n_tiles` as the reference arm's normalization; measured as tighter, not
  adopted on one kernel and one GPU

## 9. Open questions

1. **How many correct-but-different kernels per task?** Two is enough to have a denominator; more
   would make metric 2 statistical rather than indicative, at linear authoring cost.
2. **Does the Triton mutant have to be a faithful port of the NumPy one?** A fault class can be
   expressible in one and not the other — "dtype semantics" behaves differently when the compiler
   chooses the accumulator. If they diverge, per-backend detection is comparing different bugs,
   and the design must either constrain the port or report the divergence.
3. **What does the agent do when it cannot produce the requested fault?** Some subcategories may
   be inexpressible in an elementwise NumPy kernel. Recording the refusal is more honest than
   accepting a substitute, but it leaves a class with no mutant.
