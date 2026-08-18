# The blinded corpus pilot

**Date:** 2026-08-17
**Instance:** Lambda A10, sm_86, torch 2.7.0, triton 3.3.0
**Corpus:** 5 Triton softmax mutants, each authored by a separate agent under
`docs/protocol/mutant-authoring.md`
**Compare with:** `2026-08-17-provisional-corpus-pilot.md`, the contaminated run

---

## What blinding changed

Each mutant was authored by an agent given exactly two things: the correct Triton kernel and one
verbatim ISSTA subcategory description. None saw the property set, any contract, any tolerance,
any other mutant, or any indication that an oracle layer exists. Bodies are recorded verbatim.

| | contaminated | blinded |
|---|---|---|
| Gate rejection rate | **0.00** (5/5 admitted) | **0.20** (4/5 admitted) |
| `indexing_and_stride` detection | 0.778 | **0.333** |
| `operator_implementation` | 0.778 | **unreachable on the ladder** |
| Arms differentiated? | no | **no** |

The rejection rate moved exactly as predicted. Five-for-five is what an author who knows the
answer produces; a blinded author's attempts at a named class should sometimes miss, and one did.

## Result 1 — the null result survives blinding, so it is now a finding

| subcategory | allclose | reference | declarative | hybrid | tolerance-free |
|---|---|---|---|---|---|
| data_type_semantics | 0.778 | 0.778 | 0.778 | 0.778 | 0.000 |
| indexing_and_stride | 0.333 | 0.333 | 0.333 | 0.333 | **0.333** |
| branch_predication | 0.333 | 0.333 | 0.333 | 0.333 | 0.000 |
| special_value_handling | 0.333 | 0.333 | 0.333 | 0.333 | 0.000 |

**All four arms remain identical on every mutant.** Two independently authored corpora — one
contaminated, one blinded — agree. The contamination hypothesis for the tie is therefore no
longer sufficient, and the honest reading is that *on this task and this corpus the declarative
arm has no detection advantage over plain `allclose`.*

Why, mechanically: every one of these defects is numerically gross enough that a reference
comparison catches it. The bugs that would separate the arms are those where `allclose` passes
and a structural invariant still fails — numerically close but categorically wrong. None of the
five is that, and a corpus of five cannot be expected to contain one.

**`indexing_and_stride` is the exception worth noting.** Its detection is 0.333 and its
*tolerance-free* detection is also 0.333 — every group it was caught on was caught without a
tolerance. That is the declarative arm working as advertised, on one mutant.

## Result 2 — the ladder has a coverage hole, and only a blinded author found it

The rejected mutant invented a plausible tile-size specialization:

```python
if BLOCK >= 1024:
    half = BLOCK // 2
    ...
    d = tl.sum(tl.where(offs < half, e, 0.0), axis=0)   # sums only the lower half
```

The gate rejected it as "not broken on any group". That verdict is correct **and misleading**:
the defect is real, but its branch is dead at every ladder shape. `BLOCK` is the next power of
two above `n_cols`, the ladder's widest row is 129, so `BLOCK` never exceeds 256.

Measured directly:

| n_cols | BLOCK | branch | max abs error |
|---|---|---|---|
| 129 | 256 | dead | 7.5e-09 |
| 512 | 512 | dead | 1.9e-09 |
| 513 | 1024 | **taken** | **1.1e-04** |
| 1024 | 1024 | **taken** | **1.4e-02** |
| 4096 | 4096 | **taken** | **7.7e-03** |

Re-gated against `softmax_at_scale`, whose rows reach 8192: **ADMITTED**, detected 3/5 by all
four arms.

So the ladder cannot express **shape-specialized** defects. That matters more than one mutant:
"operator logic is incorrect or incomplete after specialization for type or tile shape" is the
single largest subcategory in the taxonomy at **80 of 301 bugs**, and specialization is the whole
mechanism. A corpus that cannot reach it under-samples the taxonomy's biggest class.

This is precisely what the contaminated corpus could not surface. Knowing the properties, the
compromised author wrote defects that fire everywhere; a blinded author had no such pull and
wrote one that fires only where the corpus does not look.

## Result 3 — blinding does not matter equally for every class

The `branch_predication` mutant returned by the blinded agent is `mask = offs <= n_cols` —
**character-identical** to the contaminated one. For that class the canonical off-by-one is
obvious enough that knowing the properties adds nothing.

So contamination is not uniform. It distorted `operator_implementation` severely and
`indexing_and_stride` materially, and left `branch_predication` untouched. A blinding claim
should be made per class, not globally.

## What this does not establish

- **Five mutants, one task, one backend, one seed.** Per-class rates rest on one kernel each.
- **No correct-but-different kernels were run here**, so no false-positive rate is reported.
- The arms tying is a statement about *this* corpus. A corpus containing a numerically-subtle
  invariant violation could still separate them, and constructing one deliberately would beg the
  question the blinding exists to answer.
- The at-scale re-gating used 5 groups, not 9; 3/5 and 0.333 are not directly comparable.

## What to do next

1. **Widen the ladder, or add a shape-specialization rung.** The taxonomy's largest class is
   currently unreachable.
2. **Grow the corpus per class.** One mutant per class cannot distinguish "this class does not
   differentiate the arms" from "this kernel does not".
3. **Run the correct-but-different kernels** so metric 2 has a denominator.
