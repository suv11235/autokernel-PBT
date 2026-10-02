/-
Proposition 1 (a) and (b) of the paper, mechanized in core Lean 4 — no Mathlib.

The softmax property bundle the gate decides on is {finite, in [0, 1], rows sum to one,
shift invariance}. `docs/measurements/2026-10-01-softmax-temperature-blind-spot.md` measured
that the gate admits softmax(βx) at every β. This file proves why, and proves that adding
the log-ratio law closes the gap, over an abstract ordered field with an abstract `exp`.

Only three facts about `exp` are ever assumed (`ExpLaws`): it is positive, it turns sums into
products, and it is injective. The completeness theorem `softmax_unique` assumes none of them:
it is pure field algebra. `#print axioms` at the end of the file audits both claims.
-/
open Std Lean.Grind

/-- An exponential, as an opaque primitive: the one the informal spec names. -/
class HasExp (F : Type) where
  exp : F → F

/-- Everything about `exp` the incompleteness half uses. Each is first-order and about `exp`
alone, so a language with no real-number type can take them as axioms about an
uninterpreted function — and the certificate's check C6 can sample them. -/
class ExpLaws (F : Type) [Field F] [LT F] [HasExp F] where
  exp_pos : ∀ x : F, 0 < HasExp.exp x
  exp_add : ∀ x y : F, HasExp.exp (x + y) = HasExp.exp x * HasExp.exp y
  exp_inj : ∀ x y : F, HasExp.exp x = HasExp.exp y → x = y

namespace SoftmaxBundle

open HasExp

section Algebra

variable {F : Type} [Field F] [HasExp F]

/-- The row's partition function, Σ_j exp x_j. -/
def partition (xs : List F) : F := (xs.map exp).sum

/-- Row-wise softmax: a row of n inputs to a row of n outputs. -/
def softmaxRow (xs : List F) : List F := xs.map (fun x => exp x / partition xs)

/-- Law ROW_SUMS of `kernels/tasks/softmax/acceptance.yaml`. -/
def rowsSumToOne (ys : List F) : Prop := ys.sum = 1

/-- The log-ratio law of Proposition 1(b), in multiplicative form: log y_j − x_j is constant
along the row, i.e. y_j = c · exp x_j for one c per row. Stated without `log`, so it needs no
inverse of `exp`. -/
def logRatioLaw (xs ys : List F) : Prop := ∃ c : F, ys = xs.map (fun x => c * exp x)

omit [HasExp F] in
theorem sum_map_mul_left (c : F) (f : F → F) (xs : List F) :
    (xs.map (fun x => c * f x)).sum = c * (xs.map f).sum := by
  induction xs with
  | nil => simp; grind
  | cons a l ih => simp [ih]; grind

theorem softmaxRow_eq_map (xs : List F) :
    softmaxRow xs = xs.map (fun x => (partition xs)⁻¹ * exp x) := by
  unfold softmaxRow
  congr 1
  funext x
  grind

/-- **Proposition 1(b).** Row sums and the log-ratio law pin softmax: any row output satisfying
both is `softmaxRow xs`. No property of `exp` is used — the proof is field algebra alone, so
the law pins the kernel to softmax *relative to whatever `exp` the spec names*. -/
theorem softmax_unique (xs ys : List F) (hlaw : logRatioLaw xs ys) (hsum : rowsSumToOne ys) :
    ys = softmaxRow xs := by
  obtain ⟨c, rfl⟩ := hlaw
  unfold rowsSumToOne at hsum
  rw [sum_map_mul_left] at hsum
  have hc : c = (partition xs)⁻¹ := Field.eq_inv_of_mul_eq_one hsum
  rw [softmaxRow_eq_map, hc]

end Algebra

section Order

variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F]
  [HasExp F] [ExpLaws F]

open ExpLaws

theorem partition_pos (xs : List F) (h : xs ≠ []) : 0 < partition xs := by
  unfold partition
  induction xs with
  | nil => exact absurd rfl h
  | cons a l ih =>
    simp
    cases l with
    | nil =>
      have := exp_pos a
      grind
    | cons b m =>
      have := ih (by simp)
      have := exp_pos a
      grind

theorem partition_ne_zero (xs : List F) (h : xs ≠ []) : partition xs ≠ 0 :=
  Preorder.ne_of_gt (partition_pos xs h)

/-- Law ROW_SUMS holds for softmax on every non-empty row. -/
theorem softmaxRow_sum (xs : List F) (h : xs ≠ []) : rowsSumToOne (softmaxRow xs) := by
  unfold rowsSumToOne
  rw [softmaxRow_eq_map, sum_map_mul_left]
  exact Field.inv_mul_cancel (partition_ne_zero xs h)

omit [HasExp F] [ExpLaws F] in
theorem le_sum_of_mem (l : List F) (hpos : ∀ z ∈ l, 0 ≤ z) (y : F) (hy : y ∈ l) : y ≤ l.sum := by
  induction l with
  | nil => simp at hy
  | cons a m ih =>
    simp at hy
    have hm : ∀ z ∈ m, 0 ≤ z := fun z hz => hpos z (by simp [hz])
    have hsum : 0 ≤ m.sum := by
      clear ih hy hpos
      induction m with
      | nil => simp
      | cons b k ihk =>
        simp
        have hb := hm b (by simp)
        have hk := ihk (fun z hz => hm z (by simp [hz]))
        grind
    simp
    rcases hy with rfl | hy
    · grind
    · have := ih hm hy
      have := hpos a (by simp)
      grind

/-- Law UNIT_INTERVAL holds for softmax. -/
theorem softmaxRow_unit (xs : List F) (h : xs ≠ []) :
    ∀ y ∈ softmaxRow xs, 0 ≤ y ∧ y ≤ 1 := by
  intro y hy
  have hpos : ∀ z ∈ softmaxRow xs, 0 ≤ z := by
    intro z hz
    rw [softmaxRow_eq_map] at hz
    simp at hz
    obtain ⟨x, _, rfl⟩ := hz
    exact Preorder.le_of_lt
      (OrderedRing.mul_pos (Field.IsOrdered.inv_pos_iff.mpr (partition_pos xs h)) (exp_pos x))
  constructor
  · exact hpos y hy
  · have := le_sum_of_mem _ hpos y hy
    have := softmaxRow_sum xs h
    unfold rowsSumToOne at this
    grind

omit [LE F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] in
/-- Partition under a per-row shift: Σ exp(x_j + c) = exp c · Σ exp x_j. -/
theorem partition_shift (xs : List F) (c : F) :
    partition (xs.map (fun x => x + c)) = exp c * partition xs := by
  unfold partition
  rw [List.map_map, ← sum_map_mul_left]
  congr 1
  apply List.map_congr_left
  intro x _
  simp [Function.comp, exp_add]
  grind

/-- Law SHIFT_INVARIANCE holds for softmax. -/
theorem softmaxRow_shift (xs : List F) (h : xs ≠ []) (c : F) :
    softmaxRow (xs.map (fun x => x + c)) = softmaxRow xs := by
  unfold softmaxRow
  rw [partition_shift, List.map_map]
  congr 1
  funext x
  simp [exp_add]
  have := Preorder.ne_of_gt (exp_pos c)
  have := partition_ne_zero xs h
  grind

/-- softmax at temperature 1/β: the family of `saboteurs.py`. -/
def softmaxBeta (β : F) (xs : List F) : List F := softmaxRow (xs.map (fun x => β * x))

/-- **Proposition 1(a), membership.** For every β, softmax(βx) satisfies all of ROW_SUMS,
UNIT_INTERVAL and SHIFT_INVARIANCE (FINITE_OUTPUT is vacuous over a field). -/
theorem softmaxBeta_satisfies_bundle (β : F) (xs : List F) (h : xs ≠ []) :
    rowsSumToOne (softmaxBeta β xs)
    ∧ (∀ y ∈ softmaxBeta β xs, 0 ≤ y ∧ y ≤ 1)
    ∧ (∀ c : F, softmaxBeta β (xs.map (fun x => x + c)) = softmaxBeta β xs) := by
  have hne : xs.map (fun x => β * x) ≠ [] := by cases xs <;> simp_all
  refine ⟨softmaxRow_sum _ hne, softmaxRow_unit _ hne, ?_⟩
  intro c
  unfold softmaxBeta
  have : (xs.map (fun x => x + c)).map (fun x => β * x)
      = (xs.map (fun x => β * x)).map (fun x => x + β * c) := by
    rw [List.map_map, List.map_map]
    congr 1
    funext x
    simp
    grind
  rw [this, softmaxRow_shift _ hne]

/-- **Proposition 1(a), incompleteness.** At every β ≠ 1 the family member differs from softmax
on the row (0, 1): the bundle's solution set is strictly larger than {softmax}. -/
theorem bundle_incomplete (β : F) (hβ : β ≠ 1) : softmaxBeta β [0, 1] ≠ softmaxRow [0, 1] := by
  intro heq
  unfold softmaxBeta softmaxRow partition at heq
  simp at heq
  obtain ⟨_, h2⟩ := heq
  have e0 := exp_pos (0 : F)
  have e1 := exp_pos (1 : F)
  have eb := exp_pos β
  have h0 : exp (0 : F) = 1 := by
    have h := exp_add (0 : F) 0
    have := Preorder.ne_of_gt e0
    grind
  have : exp β = exp (1 : F) := by grind
  exact hβ (exp_inj _ _ this)

end Order

end SoftmaxBundle

#print axioms SoftmaxBundle.softmax_unique
#print axioms SoftmaxBundle.softmaxBeta_satisfies_bundle
#print axioms SoftmaxBundle.bundle_incomplete

/-! ## What an axiomatized `exp` can and cannot pin

The three `ExpLaws` axioms are scale-invariant: if `exp` satisfies them, so does
`x ↦ exp (β * x)` for every β ≠ 0, and softmax with respect to that function *is* softmax(βx).
So a theorem proved from `ExpLaws` alone about "softmax with respect to `exp`" is equally a
theorem about softmax(βx): the proof cannot tell the temperature, exactly as the test bundle
cannot. Only something outside the axioms can — the executable reading, which instantiates
`exp` at the real one, or an axiom that is not scale-invariant. The tangent line
`1 + x ≤ exp x` is such an axiom, and `tangent_rigid` shows it suffices. -/

namespace SoftmaxBundle

section Functional

variable {F : Type} [Field F] [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F]

/-- `ExpLaws`, as a predicate on an arbitrary function. -/
def ExpAxioms (e : F → F) : Prop :=
  (∀ x, 0 < e x) ∧ (∀ x y, e (x + y) = e x * e y) ∧ (∀ x y, e x = e y → x = y)

/-- The tangent-line axiom: `e` lies above its tangent at 0. -/
def Tangent (e : F → F) : Prop := ∀ x, 1 + x ≤ e x

/-- softmax with respect to an arbitrary exponential. -/
def softmaxRowWith (e : F → F) (xs : List F) : List F := xs.map (fun x => e x / (xs.map e).sum)

omit [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] in
/-- With the true `exp`, `softmaxRowWith` is `softmaxRow`. -/
theorem softmaxRowWith_exp [HasExp F] (xs : List F) :
    softmaxRowWith HasExp.exp xs = softmaxRow xs := rfl

omit [LE F] [LT F] [LawfulOrderLT F] [IsLinearOrder F] [OrderedRing F] in
/-- softmax with respect to the rescaled exponential is `softmaxBeta β`, the kernel the gate
admits. -/
theorem softmaxRowWith_scaled [HasExp F] (β : F) (xs : List F) :
    softmaxRowWith (fun x => HasExp.exp (β * x)) xs = softmaxBeta β xs := by
  unfold softmaxBeta softmaxRow partition softmaxRowWith
  rw [List.map_map, List.map_map]
  rfl

/-- Every `ExpLaws` axiom survives rescaling the argument by β ≠ 0. -/
theorem expAxioms_scaled (e : F → F) (he : ExpAxioms e) (β : F) (hβ : β ≠ 0) :
    ExpAxioms (fun x => e (β * x)) := by
  obtain ⟨hpos, hadd, hinj⟩ := he
  refine ⟨fun x => hpos _, fun x y => ?_, fun x y h => ?_⟩
  · show e (β * (x + y)) = e (β * x) * e (β * y)
    rw [← hadd]
    congr 1
    grind
  · have := hinj _ _ h
    grind

/-- From the tangent line and additivity: `e x * (1 - x) ≤ 1`. -/
theorem mul_one_sub_le_one (e : F → F) (hpos : ∀ x, 0 < e x)
    (hadd : ∀ x y, e (x + y) = e x * e y) (ht : Tangent e) (x : F) : e x * (1 - x) ≤ 1 := by
  have h1 := ht (-x)
  have h0 : e 0 = 1 := by
    have h := hadd 0 0
    have := Preorder.ne_of_gt (hpos 0)
    grind
  have h2 : e x * e (-x) = 1 := by
    rw [← hadd]
    have : x + -x = 0 := by grind
    rw [this, h0]
  have := OrderedRing.mul_le_mul_of_nonneg_left h1 (Preorder.le_of_lt (hpos x))
  grind

/-- **Rigidity.** If `e` and its rescaling by β both satisfy the tangent-line axiom, then
β = 1. The axiom is first-order and about `e` alone, so an SMT language can assume it and the
certificate's check C6 can sample it; with it, the axiomatized `exp` is temperature-rigid. -/
theorem tangent_rigid (e : F → F) (he : ExpAxioms e) (β : F) (ht : Tangent e)
    (hs : Tangent (fun x => e (β * x))) : β = 1 := by
  obtain ⟨hpos, hadd, _⟩ := he
  have hs' : ∀ x : F, 1 + x ≤ e (β * x) := hs
  have one : (0 : F) < 1 := OrderedRing.zero_lt_one
  have two : (0 : F) < 2 := by grind
  -- (1 + x)(1 - βx) ≤ e(βx)(1 - βx) ≤ 1 whenever 1 - βx ≥ 0
  have key : ∀ x : F, 0 ≤ 1 - β * x → (1 + x) * (1 - β * x) ≤ 1 := by
    intro x hx
    have a := OrderedRing.mul_le_mul_of_nonneg_right (hs' x) hx
    have b := mul_one_sub_le_one e hpos hadd ht (β * x)
    grind
  -- the mirror image, from the rescaled function: (1 + βx)(1 - x) ≤ 1 whenever 1 - x ≥ 0
  have key' : ∀ x : F, 0 ≤ 1 - x → (1 + β * x) * (1 - x) ≤ 1 := by
    intro x hx
    have a := OrderedRing.mul_le_mul_of_nonneg_right (ht (β * x)) hx
    have b := mul_one_sub_le_one (fun x => e (β * x)) (fun x => hpos _)
      (fun x y => by show e (β * (x + y)) = e (β * x) * e (β * y); rw [← hadd]; congr 1; grind)
      hs x
    grind
  rcases LinearOrder.trichotomy β 1 with hlt | heq | hgt
  · exfalso
    rcases LinearOrder.trichotomy 0 β with hpos' | hzero | hneg
    · -- 0 < β < 1: x = (1 - β)/2 makes key read (1 - β) ≤ β (1 - β)/2 < (1 - β)
      have hinv := Field.IsOrdered.inv_pos_iff.mpr two
      have hb : 0 < 1 - β := by grind
      obtain ⟨x, hxdef⟩ : ∃ x : F, x = (1 - β) * 2⁻¹ := ⟨_, rfl⟩
      have hxpos : 0 < x := by rw [hxdef]; exact OrderedRing.mul_pos hb hinv
      have hx2 : x * 2 = 1 - β := by
        rw [hxdef]
        have := Field.inv_mul_cancel (Preorder.ne_of_gt two)
        grind
      have hβx : β * x < 1 - β := by
        have := OrderedRing.mul_lt_mul_of_pos_left (show β < 2 by grind) hxpos
        grind
      have hk := key x (by grind)
      have := OrderedRing.mul_lt_mul_of_pos_left hβx hxpos
      grind
    · -- β = 0: x = 1 makes key read 2 ≤ 1
      have := key 1 (by grind)
      grind
    · -- β < 0: x = 1 makes key read 2 (1 - β) ≤ 1 with 1 - β > 1
      have := key 1 (by grind)
      have := OrderedRing.mul_lt_mul_of_pos_left (show (1 : F) < 1 - β by grind) two
      grind
  · exact heq
  · exfalso
    -- β > 1: x = (β - 1)/(2β) makes key' read (β - 1) ≤ β x = (β - 1)/2 < (β - 1)
    have hpos' : 0 < β := by grind
    have hb : 0 < β - 1 := by grind
    have h2β : 0 < 2 * β := OrderedRing.mul_pos two hpos'
    have hinv := Field.IsOrdered.inv_pos_iff.mpr h2β
    obtain ⟨x, hxdef⟩ : ∃ x : F, x = (β - 1) * (2 * β)⁻¹ := ⟨_, rfl⟩
    have hxpos : 0 < x := by rw [hxdef]; exact OrderedRing.mul_pos hb hinv
    have hx2 : x * (2 * β) = β - 1 := by
      rw [hxdef]
      have := Field.inv_mul_cancel (Preorder.ne_of_gt h2β)
      grind
    have hx1 : x < 1 := by
      rcases LinearOrder.trichotomy x 1 with h | h | h
      · exact h
      · exfalso; grind
      · exfalso
        have := OrderedRing.mul_lt_mul_of_pos_right h h2β
        grind
    have hβx : β * x < β - 1 := by
      have := OrderedRing.mul_lt_mul_of_pos_left hx1 h2β
      grind
    have hk := key' x (by grind)
    have := OrderedRing.mul_lt_mul_of_pos_left hβx hxpos
    grind

end Functional

end SoftmaxBundle

#print axioms SoftmaxBundle.expAxioms_scaled
#print axioms SoftmaxBundle.softmaxRowWith_scaled
#print axioms SoftmaxBundle.tangent_rigid
