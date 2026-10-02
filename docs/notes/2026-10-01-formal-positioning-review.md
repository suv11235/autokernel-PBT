# Review: positioning autokernel-PBT around formal verification in agentic kernel writing

**Date:** 2026-10-01
**Scope:** the formal-adherence design (`docs/superpowers/specs/2026-10-01-agent-formal-adherence-design.md`),
ADR 0002, feature 0009 as built, and the AAAI draft under `paper/`. Read against the question:
*how should this utility be positioned for formal verification inside agentic automation
workloads, kernel writing above all?*

## 1. What the current positioning gets right

1. **The verifier of the verifier.** The strongest idea in the stack is not "agents write
   proofs" (Vericoding, VERINA, CLEVER already do) but that when the agent writes the spec, the
   model and the proof, *a checked proof is a claim, and the harness must test the claim*. The six
   checks execute every formal artifact on PBT's own cases with harness-owned tolerances. That is
   the right division of labour for an agent that is simultaneously author and prover, and it is
   the piece no prior work has for numerical kernels.
2. **Certificates, not gates.** Refusing to make proof the loop's filter — citing the measured
   0.24× vs 1.03× GEMM cost of a hard proof gate — keeps the formal track from becoming the
   performance trap the field has already walked into. Post-hoc certificates on PBT-accepted
   kernels is the correct shape for an optimization loop.
3. **Gaming is measured, not forbidden.** Letting the agent write the spec on purpose turns the
   obvious objection ("the agent will weaken the spec") into the headline measurement (the gaming
   profile by check). That is a result reviewers will not have seen.
4. **The motivating blind spot is real and quantified.** softmax(βx) admitted at every β, a
   52–54× faster constant fill admitted, and a proposition that characterizes the solution set:
   the case that *testing cannot say its checks characterize the function* is made with numbers.

## 2. Where the positioning is weak

1. **The formal layer had no result.** Until today every formal claim was "built" or "planned".
   The paper's related-work section already states the Cedar division of labour — "proofs about
   the oracle and the plan, tests on the code" — but nothing in the repo proved anything about the
   oracle mechanically. `proofs/softmax-bundle/` now does
   (`docs/measurements/2026-10-01-softmax-bundle-mechanized.md`), and it is the first formal
   artifact in the project. The paper should lead the formal section with it.
2. **"Prove the oracle" and "prove the kernel" were conflated in the narrative.** They are
   different products with different costs and different consumers:
   - *proofs about the oracle* — that a property bundle characterizes the function — are cheap
     (367 lines, no Mathlib, one second), human- or agent-written once per task, and make the
     gate complete. Their consumer is the loop designer.
   - *proofs about the kernel* — that a given candidate adheres — are expensive, per candidate,
     agent-written, and are the pilot's subject. Their consumer is whoever ships the kernel.
   The utility should be positioned as supplying **both** and saying which is which. For
   agentic workloads the first is the one that changes the loop's reachable frontier (it fixes
   what the gate converges to); the second is what turns "admitted" into "certified".
3. **The conclusion overclaims.** "Formal proof … is what can turn 'passes the gate' into a
   guarantee" is not what the design delivers: C2–C6 are sampled and C3 is a lower bound. The
   honest claim is *a certificate whose every assumption has been executed against the cases
   the gate used*. Say that.
4. **The pilot's language story is thin where it matters most for agents.** Agents in
   automation workloads will reach for SMT-backed languages first (short proofs, fast
   feedback). The design's only remark on them is that `exp` exists only as axioms "so C6
   carries the weight". Today's mechanization makes that precise: the natural axioms of `exp`
   are scale-invariant, so a proof from them is temperature-blind, and one first-order axiom
   (the tangent line) restores rigidity. That turns a vague risk into a protocol item and a
   prediction, and it should be in the pilot's pre-registration.
5. **The loop never sees the formal track.** E4 (formal feedback for self-improvement) is the
   experiment the user's stated direction asks for — proofs and PBT counterexamples feeding the
   agent — and it is last in the program. For positioning in *agentic* workloads it should be
   the centrepiece, with E1 as its precondition rather than the headline.

## 3. Recommended positioning statement

> A correctness gate for kernel-writing agents in which **property-based tests decide, formal
> proofs certify, and each checks the other**: proofs about the oracle establish that the
> gate's laws characterize the function (so the loop cannot converge on a wrong kernel the laws
> admit), and proofs about a kernel are accepted only when every spec, model, hypothesis and
> axiom they rest on has been executed on the gate's own cases (so an agent that writes its own
> spec cannot certify the wrong thing). Both kinds of proof are written by agents; both kinds
> of failure are measured.

Three audiences, one sentence each:
- *Loop designers*: a complete, calibrated gate costs milliseconds; completeness is proven, not
  hoped.
- *Verification researchers*: the first measured gaming profile for agent-written specs of
  numerical kernels, across verifier families.
- *Kernel consumers*: a certificate that names what was proved, under which `exp`, and what was
  only tested.

## 4. Directions opened today, in priority order

1. **Rigidity audit of primitive axioms (C6′).** Per attempt, check whether the axioms the proof
   uses are invariant under argument rescaling; record it beside the gaming profile. One
   function call against the executable readings. Prediction: every SMT-language attempt that
   omits a tangent-type axiom is temperature-blind and is caught only by C3's execution.
2. **Mechanize 1(c)** (layernorm's affine-tie law) and the permutation members, then the
   float forms of both completion laws — E2 proper, spec-first because it edits the bundle.
3. **E4 first, not last.** Return failed checks, unsolved goals and refuting inputs to the agent
   and measure whether attempts improve beyond what the completed bundle alone gives. The
   harness needs only the existing certificate record and a prompt template.
4. **A Lean-without-Mathlib adapter for 0010.** Today's file shows propositions about the oracle
   need no library; the adapter's executable-reading route can instantiate `F := Float` through
   core Lean's IEEE model (v4.33+) for everything except `exp`, which stays an external binding.
