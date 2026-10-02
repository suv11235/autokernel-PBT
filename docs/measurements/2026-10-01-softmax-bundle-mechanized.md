# The softmax bundle's gap, mechanized: Proposition 1 in Lean 4, and what an axiomatized `exp` cannot see

**Date:** 2026-10-01
**Instance:** Apple M4 Pro, macOS 26.6.1
**Toolchain:** Lean 4 v4.33.0 (`~/.elan/toolchains/leanprover--lean4---v4.33.0/bin/lean`), **no
Mathlib** — the field and ordered-ring reasoning comes from core Lean's `grind` classes
(`Lean.Grind.Field`, `Lean.Grind.OrderedRing`)
**Artifact:** `proofs/softmax-bundle/SoftmaxBundle.lean`, 367 lines, six audited theorems; compiles
in about one second; pinned by `tests/proofs/test_softmax_bundle_lean.py` (marker `formal`,
skipped without the toolchain)
**Companions:** `2026-10-01-softmax-temperature-blind-spot.md` (the measurement this proves the
reason for) and the paper's Proposition 1 (`paper/sections/results.tex`)

**Headline.** The paper's Proposition 1(a) and (b) — the gate's softmax bundle admits softmax(βx)
at every β, and adding the log-ratio law pins softmax — are now machine-checked over an abstract
ordered field, on Lean's three standard axioms and nothing else. Mechanizing them exposed two
facts the paper proof did not state. **The completeness half uses no property of `exp` at all**:
row sums plus the log-ratio law pin the kernel to softmax *relative to whatever `exp` the spec
names*, by field algebra alone. And **the natural first-order axioms of `exp` — positive,
additive, injective — are scale-invariant**: they hold equally for x ↦ exp(βx), and softmax with
respect to that function *is* softmax(βx). So a proof that a kernel is "softmax with respect to
`exp`", carried out from those axioms, is temperature-blind in exactly the way the test bundle is.
One more first-order axiom, the tangent line 1 + x ≤ exp x, restores rigidity: if `exp` and its
rescaling both satisfy it, β = 1. That axiom is sampleable, which is what check C6 does.

---

## 1. Construction

| | |
|---|---|
| numbers | any `Lean.Grind.Field` `F`; for the order-dependent theorems also `LE`, `LT`, `LawfulOrderLT`, `IsLinearOrder`, `OrderedRing` — an abstract ordered field, which ℝ instantiates |
| rows | `List F`; a kernel's row output is a list the same length as its input row |
| `exp` | an opaque `HasExp.exp : F → F` with a separate class `ExpLaws` of exactly three axioms: `0 < exp x`, `exp (x + y) = exp x * exp y`, `exp x = exp y → x = y` |
| the bundle | ROW_SUMS as `ys.sum = 1`; UNIT_INTERVAL as `∀ y ∈ ys, 0 ≤ y ∧ y ≤ 1`; SHIFT_INVARIANCE as `softmaxRow (xs.map (· + c)) = softmaxRow xs`. FINITE_OUTPUT is vacuous over a field |
| the log-ratio law | in multiplicative form, `∃ c, ys = xs.map (fun x => c * exp x)` — "log y_j − x_j is constant along the row", stated without `log` |
| the family | `softmaxBeta β xs := softmaxRow (xs.map (β * ·))`, the construction of `saboteurs.py` |
| audit | `#print axioms` on every theorem; the test fails on any axiom beyond `propext`, `Classical.choice`, `Quot.sound`, and on a renamed or deleted theorem |
| reproduce | `~/.elan/toolchains/leanprover--lean4---v4.33.0/bin/lean proofs/softmax-bundle/SoftmaxBundle.lean` |

## 2. The result

| theorem | statement | assumes about `exp` | paper |
|---|---|---|---|
| `softmax_unique` | log-ratio law ∧ row sums ⇒ `ys = softmaxRow xs` | **none** | 1(b) |
| `softmaxBeta_satisfies_bundle` | ∀ β, on a non-empty row, softmax(βx) satisfies ROW_SUMS, UNIT_INTERVAL, SHIFT_INVARIANCE | `exp_pos`, `exp_add` | 1(a) membership |
| `bundle_incomplete` | ∀ β ≠ 1, `softmaxBeta β [0, 1] ≠ softmaxRow [0, 1]` | + `exp_inj` | 1(a) strictness |
| `expAxioms_scaled` | the three axioms hold for `x ↦ exp (β * x)` whenever β ≠ 0 | — | new |
| `softmaxRowWith_scaled` | softmax w.r.t. `x ↦ exp (β * x)` equals `softmaxBeta β` | — | new |
| `tangent_rigid` | `Tangent exp ∧ Tangent (x ↦ exp (β * x)) → β = 1`, where `Tangent e := ∀ x, 1 + x ≤ e x` | `exp_pos`, `exp_add` | new |

Every `#print axioms` line reads `[propext, Classical.choice, Quot.sound]` (one reads `[propext]`).
No `sorry`, no declared `axiom`.

**The proof is cheap.** 367 lines including comments, written and checked in one session with no
library beyond core Lean; the algebra is discharged by `grind`. The order-dependent half needed
hand-steered positivity (`OrderedRing.mul_pos`, `Field.IsOrdered.inv_pos_iff`), the rest is
induction over lists.

## 3. What the mechanization found that the paper proof did not say

**3.1 Completeness is `exp`-agnostic.** `softmax_unique` is stated in a section whose only
instances are `Field F` and `HasExp F`: `ExpLaws` is not in scope and the proof does not use it.
That is, {row sums, log-ratio law} pins y = softmax_e(x) for *any* function e — the completion
law makes the bundle complete relative to the `exp` the spec names, and says nothing about whether
that `exp` is the real one. In the executed gate this is harmless, because the executable reading
instantiates `exp` at `np.exp`, which is why the log-ratio spread separates fast-exp (48 eps) from
the exact kernel (4.8 eps) in the companion. In a formal certificate it is the whole question:
**which `exp` did the agent prove about?**

**3.2 The natural axioms do not say.** `expAxioms_scaled` + `softmaxRowWith_scaled`: positive,
additive and injective are all preserved by `exp ∘ (β ·)`, and softmax with respect to the
rescaled function is softmax(βx). Hence any theorem derivable from those three axioms about
"softmax w.r.t. `exp`" holds, under the rescaled interpretation, of softmax(βx). A certificate
whose proof rests only on them has established adherence to a spec that softmax(2x) also meets.
This is the test bundle's blind spot reproduced one level up, and it is not a defect of any
language: it is a property of the axiom set, so it afflicts every SMT-backed language that must
axiomatize `exp` (Dafny, Verus, F*, Why3) and every ITP proof that chooses to.

**3.3 One tangent-line axiom fixes it.** `tangent_rigid`: 1 + x ≤ exp x is not scale-invariant,
and the proof shows that together with `exp_add` it forces β = 1 (the argument: from the tangent
line and `exp(x)·exp(−x) = 1` one gets `exp x · (1 − x) ≤ 1`; applying the two bounds to `exp` and
to its rescaling at a chosen x yields (1 − β) ≤ β·x or (β − 1) ≤ β·x with x half as large as the
left side allows). The axiom is first-order, mentions `exp` alone, and holds on samples of ℝ — so it
is exactly the kind of statement the design's C6 admits and samples at 4096 points.

## 4. What this changes in the formal-track design

- **C6's primitive axioms need a rigidity audit, not only a truth audit.** The design checks that
  each axiom is about declared primitives and true on samples. All three of `ExpLaws` pass both,
  and together they certify nothing about temperature. The pilot protocol should record, per
  attempt, whether the axioms the proof actually uses are scale-rigid — the cheapest test is the
  one above: does `x ↦ exp(βx)` satisfy them for some β ≠ 1? A C3 probe with softmax(2x) in the
  hidden set would still reject such an attempt *by execution*; what this adds is attributing the
  rejection to the proof's axioms rather than to the agent's spec.
- **The "Verus cannot state the spec" prediction is refined.** A language with no real-number
  type can state and prove everything in this file: the development never names ℝ. What it cannot
  do is instantiate `exp` — and §3.2 shows that instantiation is where the temperature lives. So
  the honest prediction for SMT languages is not "cannot express" but "proves a temperature-blind
  theorem unless the tangent axiom is assumed".
- **E2 inherits its first step done.** Proposition 1(a) and (b) are mechanized; (c), the layernorm
  affine-tie law, is not. The floating-point forms remain open exactly as before.

## 5. Threats, and what is NOT claimed

- **Over an abstract ordered field, not over floats.** Nothing here says what the log-ratio law
  should do when an output underflows to 0, or what tolerance it carries. That is E2's design
  question and is untouched.
- **Rows as lists, one row at a time.** Shift invariance is stated per row with one constant;
  that matches `ShiftRows`, which draws one constant per row. Permutation members
  (P softmax(x) Q) of Proposition 1(a) are not mechanized — they need an index structure the
  list form does not give cheaply.
- **Layernorm, 1(c), is not mechanized.**
- **`tangent_rigid` is a sufficient axiom, not a characterization of rigidity.** It shows one
  axiom that pins β; it does not show the three `ExpLaws` plus tangent determine `exp` (they do
  not: over ℝ they are satisfied only by exp, but over a non-Archimedean field other models exist).
  The claim used is only the one proved: rescaling by β ≠ 1 breaks the tangent line.
- **Core Lean's `grind` algebra classes are young** (`Lean.Grind.Field` and friends arrived in
  2025–26). The theorems are checked by the kernel like any other, so trust is in Lean's kernel,
  not in `grind` — but a future toolchain may rename the classes and the file is pinned to
  v4.33.0 for that reason.

## 6. What the next experiments inherit

- A precise statement, in the artifact's own terms, of the symmetry an agent-written formal spec
  must break, and a one-line test for whether its axioms break it.
- A mechanized anchor for the completed bundle: when E2 adds the log-ratio law, `softmax_unique`
  is the statement that the new bundle's solution set over ℝ is {softmax}, modulo the `exp`
  question of §3.1.
- The Lean side of the planned pilot needs no Mathlib for propositions of this kind — the real
  analysis the design feared (§7, "`exp` is a definition; proofs are long") was not needed to say
  anything true here. What Mathlib would add is the instantiation `F := ℝ`, `exp := Real.exp`,
  which is also where §3.2's blindness ends.
