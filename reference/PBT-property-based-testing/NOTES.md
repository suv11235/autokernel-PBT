# NOTES — synthesis, judgments, and open questions

Working notes for the property-based-testing (PBT) layer of autokernel-PBT. This is the
opinionated companion to [README.md](./README.md): the README says *what to read*, this file
says *what I think it means for us*.

Everything here that is a claim about a paper was checked against the paper or its abstract.
Where I could not verify something, it says so inline or in [§8](#8-unverified--not-found).

---

## 0. The one-paragraph version

The literature already contains a small, direct precedent for our central experiment:
Hughes' *How to Specify It!* (TFP 2019/2020) plants eight bugs in a binary-search-tree
implementation and measures which *kinds* of property catch them. His answer — model-based
(≈ our reference oracle) properties find every bug and find them ~10× faster than metamorphic
ones; metamorphic properties are individually weak but collectively strong; and metamorphic
properties are the right choice precisely when the model is expensive or would replicate the
implementation's own bugs — is the hypothesis we should expect to reproduce or refute in the
kernel domain. What makes the kernel domain different, and therefore worth a paper, is that
(a) the reference oracle is *cheap to write but expensive and imprecise to evaluate* (a GPU
run plus a tolerance argument), which inverts Hughes' cost model, and (b) the reference oracle
is systematically vulnerable in ways a BST model is not — reward hacking, tolerance laundering,
shape overfitting, and backend-dependent floating-point reassociation.

---

## 1. A taxonomy of property patterns, with worked kernel examples

### 1.1 The taxonomy

Three source taxonomies matter, and they mostly agree:

| Source | Categories |
|--------|-----------|
| Hughes, *How to Specify It!* (2020) | validity/invariant, postcondition, metamorphic (incl. preservation of equivalence), inductive, model-based |
| Wlaschin, *Choosing properties* (blog) | different paths same destination; there and back again; some things never change; the more things change the more they stay the same; solve a smaller problem first; hard to prove easy to verify; the test oracle |
| Barr et al., *Oracle Problem* survey (TSE 2015) | **specified**, **derived** (incl. pseudo-oracles and metamorphic relations), **implicit**, **lack of an automated oracle** |

Barr et al.'s axis is the one to adopt for the paper, because it is the axis our three arms
actually differ on. In their vocabulary:

- **Reference-implementation oracle** = a *pseudo-oracle* (Barr §5.1) — an independent
  implementation of the same function, the classic answer to the oracle problem.
- **Declarative-spec oracle** = *specified* + *derived/metamorphic* properties, with no
  second implementation computed.
- **Tier-2 backend properties** (sanitizer clean, no races, no OOB) = **implicit oracles**
  in exactly Barr's sense — they need no specification at all, because *any* program
  triggering them is wrong.

That third category is worth calling out separately in the paper. It is free, it has a
near-zero false-positive rate, and it is orthogonal to both other arms. Our Tier-1/Tier-2
split maps onto derived-vs-implicit cleanly.

For engineering purposes I'd use this working taxonomy, ordered by authoring cost:

| # | Pattern | Needs | FP risk | Cost |
|---|---------|-------|---------|------|
| P0 | **Implicit / crash-freedom** — no NaN where none expected, no sanitizer hit, no OOB, no race | telemetry only | very low | ~0 |
| P1 | **Range / validity invariant** — output in a known set | outputs | low | low |
| P2 | **Conservation / algebraic identity** — a scalar function of the output is pinned | outputs | medium (tolerance) | low |
| P3 | **Metamorphic relation** — relates two executions on related inputs | *case groups* | medium | medium |
| P4 | **Equivalence-preservation** — semantically equal inputs give equal outputs | case groups | medium | medium |
| P5 | **Differential / cross-config** — same kernel, different tiling/block size/dtype-of-accumulator | grouped executions | medium | low |
| P6 | **Model-based / reference oracle** — full output comparison | reference impl + tolerance | high (tolerance) | low to write, high to evaluate |

Note P5. It is a *free* extra arm we already have the machinery for: if the batch runner can
execute the same kernel under two launch configurations and store both rows with a shared
`group_id`, we get a differential oracle with no reference implementation and no algebra. In
the DL-library literature this is precisely EAGLE's move (ICSE 2022 — 17 equivalence rules
inside a single library, 20 bugs found) and Propilot's (arXiv:2606.06747). It is cheap and it
should be in the property arm, not the reference arm.

### 1.2 Worked properties

Notation: `x` is the input tensor, `y = K(x)` the kernel output, `≈` means "within the tolerance
policy of §4", `⊕` means the property needs a **case group** (two or more recorded executions
sharing a `group_id`).

#### softmax(x, axis=-1)

| ID | Tier | Kind | Property |
|----|------|------|----------|
| SM-1 | 1 | P1 | `0 ≤ y[i] ≤ 1` for all `i` — exact, no tolerance |
| SM-2 | 1 | P2 | `sum(y, axis=-1) ≈ 1` |
| SM-3 | 1 | P1 | `isfinite(y).all()` even when `x` contains values ≥ 700 (naive `exp(x)/sum(exp(x))` overflows in fp64 at ~709.78) |
| SM-4 | 1 | P3 ⊕ | **shift invariance**: `K(x + c) ≈ K(x)` for scalar `c`, including large `c` |
| SM-5 | 1 | P3 ⊕ | **monotone order preservation**: `argsort(y) == argsort(x)` along the axis — *exact, tolerance-free*, and catches reduction-index bugs |
| SM-6 | 1 | P3 ⊕ | **permutation equivariance**: `K(P x) ≈ P K(x)` for a permutation `P` along the softmax axis |
| SM-7 | 1 | P3 ⊕ | **temperature/scale**: `K(a·x)` for `a > 1` is *more* peaked — `max(K(a·x)) ≥ max(K(x))`; an inequality MR, not an equality |
| SM-8 | 1 | P2 | `-sum(y * log y) ≤ log(n)` (entropy bounded by uniform) |
| SM-9 | 1 | P1 | if `x` has a `-inf` mask entry, the corresponding `y` is exactly `0.0` |
| SM-10 | 2 | P0 | compute-sanitizer memcheck/racecheck clean; no `nan` produced under `--check-api-memory-access` |

> ⚠️ **Correction to the literature, worth a footnote in our paper.** Hatfield-Dodds'
> SciPy 2020 paper (§"Metamorphic properties") gives as its softmax metamorphic example
> `softmax(arr * factor) ≈ softmax(arr)`, calling it "scale-invariance". As printed this is
> **false** for any `factor ≠ 1`: softmax is *shift*-invariant, not *scale*-invariant —
> `softmax(a·x)_i = exp(a·x_i)/Σ exp(a·x_j)` is temperature scaling and sharpens the
> distribution. The correct relations are SM-4 (shift) and the inequality SM-7 (scale).
> This is a nice, concrete, citable illustration of the declarative arm's real hazard: a
> plausible-looking weak property published in a peer-reviewed venue that would generate
> false positives on a *correct* kernel. It argues for the "property validation" step in
> §5.3.

#### layernorm(x, γ, β, eps)

| ID | Tier | Kind | Property |
|----|------|------|----------|
| LN-1 | 1 | P2 | with `γ=1, β=0`: `mean(y, axis=-1) ≈ 0` and `var(y, axis=-1) ≈ 1` (up to the `eps` correction) |
| LN-2 | 1 | P3 ⊕ | **shift invariance**: `K(x + c, γ, β) ≈ K(x, γ, β)` |
| LN-3 | 1 | P3 ⊕ | **scale invariance**: `K(a·x, γ, β) ≈ K(x, γ, β)` for `a > 0` — unlike softmax, layernorm *is* scale invariant (modulo `eps`), so `eps` must be small relative to `a²·var(x)` for this to hold; this property is a good `eps`-handling detector |
| LN-4 | 1 | P3 ⊕ | **affine composition**: `K(x, γ, β) ≈ γ * K(x, 1, 0) + β` |
| LN-5 | 1 | P4 ⊕ | **row independence**: permuting or shuffling rows of `x` permutes rows of `y` identically; and `K(x)[i]` must not depend on `x[j]` for `j ≠ i` — test by perturbing one row and asserting all others are **bitwise identical** (tolerance-free, catches cross-row reduction/shared-memory bugs, which is the classic Triton tiling bug) |
| LN-6 | 1 | P1 | degenerate row (all elements equal) ⇒ `y ≈ β` and no NaN (this is the `1/sqrt(0+eps)` path) |
| LN-7 | 2 | P0 | racecheck clean — LN is a two-pass reduction, the prime site for missing `tl.debug_barrier`/`__syncthreads` |

LN-5 is the highest-value property in this table and it is worth generalizing: **for any
kernel with a batch/row axis, "independence along the batch axis" is a tolerance-free
metamorphic relation that directly targets the most common class of tiling bug.** I have not
found this stated as a named MR anywhere in the literature; it may be a small contribution
in its own right.

#### GEMM: `C = A @ B` (optionally `α·A@B + β·C0`)

| ID | Tier | Kind | Property |
|----|------|------|----------|
| GM-1 | 1 | P3 ⊕ | **linearity in A**: `K(A1 + A2, B) ≈ K(A1, B) + K(A2, B)` |
| GM-2 | 1 | P3 ⊕ | **scalar homogeneity**: `K(a·A, B) ≈ a·K(A, B)` |
| GM-3 | 1 | P3 ⊕ | **transpose duality**: `K(A, B)ᵀ ≈ K(Bᵀ, Aᵀ)` — catches a large fraction of index/stride swaps |
| GM-4 | 1 | P2 | **identity**: `K(A, I) ≈ A` and `K(I, B) ≈ B` — should be *exact* in IEEE arithmetic if the accumulation order is sane, so assert bitwise where possible |
| GM-5 | 1 | P3 ⊕ | **block decomposition**: split `B` column-wise into `[B1 | B2]`; `K(A, [B1|B2]) ≈ [K(A,B1) | K(A,B2)]` — pure tiling check |
| GM-6 | 1 | P3 ⊕ | **permutation**: `K(P A, B) ≈ P K(A, B)` for a row permutation `P` |
| GM-7 | 1 | P6' | **residual check, not equality** — LAPACK-style: `‖K(A,B) − A@B‖ / (n · ε · ‖A‖ · ‖B‖) ≤ τ`. This is the *right* form of the reference oracle for numerical code (see §4.2) and it is a genuinely different arm from naive `allclose` |
| GM-8 | 1 | P3 ⊕ | **sparsity/structure**: if `A` is upper-triangular and `B` upper-triangular, `K(A,B)` is upper-triangular — *exact zeros*, tolerance-free |
| GM-9 | 1 | P5 ⊕ | **split-K invariance**: same GEMM with `split_k=1` vs `split_k=4` agrees to within the reassociation bound of §4.3. A *failure* here is not necessarily a bug — it is the signal that a tolerance argument is required |
| GM-10 | 2 | P0 | register/shared-mem budget respected; no OOB on non-multiple-of-tile shapes (`M=127, K=1, N=1`) |

#### attention: `O = softmax(QKᵀ/√d + M) V`

| ID | Tier | Kind | Property |
|----|------|------|----------|
| AT-1 | 1 | P2 | rows of the implied attention matrix sum to 1 — testable *without materializing it* via `K(Q,K,V=1) ≈ 1` (set V to all-ones; output must be all-ones) |
| AT-2 | 1 | P1 | **convex-hull / range**: every output element lies within `[min(V[:,d]), max(V[:,d])]` per feature dim, because attention output is a convex combination of value rows. Tolerance-free up to fp rounding, and a *very* strong cheap check |
| AT-3 | 1 | P3 ⊕ | **key/value permutation equivariance**: permuting the K/V sequence axis jointly leaves `O` unchanged |
| AT-4 | 1 | P3 ⊕ | **query permutation equivariance**: permuting the Q sequence axis permutes `O` identically |
| AT-5 | 1 | P3 ⊕ | **causal-mask consistency**: with a causal mask, `O[:, :t]` must be **bitwise unchanged** when `K[t:], V[t:]` are replaced with arbitrary garbage. This is the single best causal-attention property — tolerance-free, and it catches the whole family of off-by-one mask bugs and future-token leaks |
| AT-6 | 1 | P3 ⊕ | **scale invariance of V**: `K(Q,K,a·V) ≈ a·K(Q,K,V)` (attention is linear in V) |
| AT-7 | 1 | P3 ⊕ | **additivity in V**: `K(Q,K,V1+V2) ≈ K(Q,K,V1) + K(Q,K,V2)` |
| AT-8 | 1 | P3 ⊕ | **shift invariance in logits**: adding a per-query constant to all of `Q·Kᵀ` (e.g. by shifting Q along a direction) leaves `O` unchanged |
| AT-9 | 1 | P1 | fully-masked row produces a defined result (all-zero or NaN per the spec) — *pick one and assert it*; this is a classic silent divergence between backends |
| AT-10 | 1 | P5 ⊕ | **chunk invariance**: FlashAttention-style online softmax must give the same answer for `block_n = 32` and `block_n = 128` within the reassociation bound |

**Observation for the paper.** Counting the tolerance-free entries above: SM-1, SM-5, SM-9,
LN-5, LN-6, GM-4, GM-8, AT-2, AT-5. That's roughly a third of the catalogue, and those are
the ones with essentially zero false-positive risk. **The declarative arm's real advantage
over the reference arm may not be "more bugs" but "bugs found without a tolerance argument."**
That is a sharper, more defensible claim than raw detection count, and it is measurable:
report detection rate *split by* whether the property required a tolerance.

### 1.3 What "case groups" must support

From the above, the group relations we actually need are:

1. `(x, x + c)` — additive shift
2. `(x, a·x)` — scalar scale
3. `(x, P x)` — permutation along a named axis
4. `(x, x')` where `x'` differs in exactly one row/block — locality probes
5. `(x, [x1 | x2])` — concatenation/split along a named axis
6. same `x`, two *launch configurations* — the P5 differential arm
7. same `x`, two *dtypes* / accumulator precisions

Items 6 and 7 are not input transformations at all; they are **execution-parameter** groups.
The `group_id` schema should therefore key on `(base_case_id, transform, execution_config)`,
not on inputs alone. Getting this wrong at the schema level would be expensive to fix later.

---

## 2. Hypothesis as driver vs. custom seeded generator

**Recommendation: use Hypothesis as a *strategy/generator library* and as the shrinking engine
for a separate on-hardware minimization phase, but do NOT use `@given` as the batch driver.
Write a thin custom driver over Hypothesis strategies. Do not write your own generators.**

This is a "both", but a specific and, I think, defensible one. The reasoning:

### 2.1 What `@given` actually does, and why it fights our architecture

`@given` owns the loop. It generates one example, calls the test function, observes
pass/fail, and adapts. Phases run in the order `explicit → reuse → generate → target →
shrink → explain` (documented `Phase` enum). Two consequences:

1. **Shrinking is not offline.** Hypothesis's reducer is *internal*: per MacIver & Donaldson
   (ECOOP 2020), it reduces the **choice sequence** consumed during generation and
   **re-runs the generator and the property** to check each candidate. Their own framing:
   "internal reduction works by re-generating test cases", and the reduced test "is one that
   could have been generated." In our architecture, "running the property" means *executing a
   kernel on a GPU/Trainium device*. So Hypothesis's shrinker cannot operate over the recorded
   table at all — it structurally requires a live execution loop.
2. **Batch-first is inverted.** Our design deliberately separates *generate → execute batch →
   evaluate oracles offline* so that N oracle strategies see byte-identical executions.
   `@given` interleaves generation with evaluation, so two oracle strategies run under
   `@given` would see *different* input sets, which is exactly the confound the record/replay
   design exists to eliminate. This is not a small point — it is the methodological core of
   the paper, and it is the thing the closest competitor (arXiv:2608.12700) gets wrong by
   re-seeding per gate.

### 2.2 What Hypothesis gives us that we should not rebuild

Real, and substantial:

- `hypothesis.extra.numpy`: `arrays(dtype, shape, *, elements, fill, unique)`,
  `array_shapes()`, `from_dtype()` (with `allow_nan`, `allow_infinity`, `allow_subnormal`,
  `min_magnitude`, `max_magnitude`), `broadcastable_shapes()`,
  `mutually_broadcastable_shapes()`, `basic_indices()`, `integer_array_indices()`. The
  edge-case biasing in the float generator (subnormals, ±0.0, signalling NaN, values near
  the exponent boundaries) is genuinely hard to reproduce and is where the bugs are.
- `hypothesis.extra.array_api` + `make_strategies_namespace(xp)`: an Array-API-standard
  namespace that works against **any** conforming array library — NumPy, PyTorch, CuPy, JAX,
  Dask. For a framework-agnostic kernel project this is close to purpose-built. (Treat as
  provisional; the docs mark array_api support as subject to change.)
- The shrinker itself, if we can give it a live loop — see §2.4.
- `hypothesis.target()`: targeted PBT in the Löscher–Sagonas sense (ISSTA 2017). For us the
  obvious target function is **relative error** — steer generation toward inputs that
  maximize `‖y − ref‖/tol`. Astropy did exactly this for time-precision testing (per
  Hatfield-Dodds, SciPy 2020). This is a strong, cheap win and it argues for keeping a live
  Hypothesis loop *somewhere* in the system.

### 2.3 What breaks if you try to use Hypothesis purely as a generator

I checked the documented surface. The honest answer is: **it works, but only through one
supported door, and the reproducibility guarantee you get is weaker than you need.**

- `SearchStrategy.example()` is documented as "designed for use in a REPL", warns outside
  interactive use, errors inside `@given`, and the docs explicitly say to use `@composite`
  or `data()` "for serious use instead." **Not viable for a corpus generator.**
- The supported door is `@given(...)` with `settings(phases=[Phase.generate], max_examples=N,
  database=None)` and a test body that appends the drawn value to a list and never fails.
  This is a real, documented use of the public API. It works. It is also slightly perverse,
  and it caps you at `max_examples` per invocation.
- Reaching into `hypothesis.internal.conjecture` (`ConjectureData`, `strategy.do_draw`) to
  draw from a strategy against your own byte source is the "clean" version — and it is
  explicitly outside the stability contract. The compatibility page states documented APIs
  "will not break except between major version bumps", while "undocumented attributes,
  modules, and behavior" may break **in patch releases**. `hypothesis.internal` is
  undocumented. For a research artifact whose recorded batches must stay reproducible across
  months of hardware runs, pinning to a private API in a library that ships weekly patch
  releases is a real, recurring maintenance tax.

**The decisive reproducibility fact:** Hypothesis does not promise that a given seed yields
the same examples across versions. The `derandomize` setting is documented as producing "the
same test cases until you update Hypothesis, Python, or the test function", and
`@reproduce_failure` is documented as making "no compatibility guarantees across Hypothesis
versions" and *erroring* if used on a different version. So a Hypothesis-generated corpus is
reproducible-by-artifact (you must keep the materialized inputs) but **not**
reproducible-by-seed across time. Our design already stores inputs in the recorded table, so
this is survivable — but it means the seed in our manifest is documentation, not a
regeneration recipe, and the paper must say so.

### 2.4 The recommended split

```
Phase A  generate    custom driver, seeded, deterministic, our own case-group logic
                     ├─ draws leaf values from hypothesis.extra.numpy / array_api strategies
                     │  via a supported @given(phases=[Phase.generate]) harvest, ONCE, then
                     │  materializes + hashes + persists the corpus
                     └─ constructs case groups (transform + execution_config) itself
Phase B  execute     batch runner, one pass per backend, persists
                     (inputs_hash, outputs, telemetry, group_id, exec_config)
Phase C  evaluate    all oracle strategies, offline, over the SAME recorded rows   ← the paper
Phase D  minimize    ONLY for failures. Live loop. Here you may use Hypothesis's @given +
                     internal shrinker, or a domain shrinker (§3.3). Costs GPU time; that's
                     acceptable because it runs on ~10 failures, not ~10,000 cases.
```

Case-group construction is the thing Hypothesis genuinely cannot do for us. `@given` has no
notion of "generate `x`, then also generate `x + c` and mark them related." You can fake it by
drawing `(x, c)` and constructing the pair inside the test body — which is fine under `@given`
but useless under batch-first, because the *pair* must become two rows sharing a `group_id`.
So the group layer is ours regardless of what generates the leaves. Given that, the marginal
cost of owning the driver is small, and the marginal benefit is large: determinism we control,
a corpus format we control, and no dependency on a private API.

**What I would NOT do:** write our own float/array generators. The edge-case density in
`hypothesis.extra.numpy`'s float strategies is the product of a decade of tuning and it is
where the bugs actually are. Rebuilding it is a guaranteed loss.

**Risk to log:** if Hypothesis's array_api extra remains provisional and changes shape, Phase A
needs a shim. Keep the strategy→corpus boundary narrow (one module) so that shim is cheap.

---

## 3. Shrinking numeric and array inputs

### 3.1 What the literature actually establishes

- **Delta debugging** (Zeller & Hildebrandt, TSE 2002) is the ancestor: `ddmin` is a
  greedy, divide-and-conquer minimization over a sequence of "changes", with a complexity
  bound and — crucially — an explicit treatment of the *unresolved* outcome, where a reduced
  input is neither passing nor failing but invalid. Every array shrinker inherits this problem.
- **Type-based / manual shrinking** (QuickCheck lineage): the shrinker is a separate function
  per type. Its failure mode is the **validity problem** — shrinks that violate a generator's
  implicit invariant.
- **Integrated shrinking** (Hedgehog, jqwik): shrinking is derived from the generator, so
  reduced values are by construction generatable.
- **Internal shrinking** (Hypothesis; MacIver & Donaldson, ECOOP 2020): shrink the *choice
  sequence*, re-generate. Cast as a **shortlex optimization problem** over choice sequences.
  Advantages claimed are *not* speed or minimality, but (i) every generator gets reduction for
  free and (ii) reduced cases are always generatable, so validity is preserved. They are
  explicit that it "can be a fair bit slower."
- **Its documented limitation is directly relevant to us.** In the Csmith evaluation,
  Hypothesis could not reduce below ~410 bytes because that is the smallest program Csmith
  can emit, whereas C-Reduce reaches a 14-byte `main`. Their framing: Hypothesis "is reducing
  against a harder validity oracle." **Translated to kernels: an internal shrinker can never
  reduce below the smallest case our generator can produce.** If our generator always emits
  a 4-D tensor with `dim ≥ 8`, no internal shrinker will ever hand us a 1×1 counterexample.
  Minimum-case design is therefore a *generator* decision, not a shrinker decision.
- **Keles, Miao & Lampropoulos, "Evaluating Shrinking" (arXiv:2608.09935, 2026)** measured
  QuickCheck vs Hedgehog vs Falsify on ETNA workloads using tree edit distance to an
  exhaustively-computed optimum, and found QuickCheck's structural shrinking "usually faster
  and… competitive on final counterexample quality; integrated shrinking does not by itself
  guarantee a performance or effectiveness advantage." Recent and unrefereed at time of
  writing, but it is the only head-to-head measurement I found, and it deflates the common
  claim that integrated/internal shrinking is simply better.

**Honest gap:** I found **no** work on shrinking *numeric array* inputs specifically. Delta
debugging, C-Reduce, hierarchical DD and the PBT shrinkers are all about structured
symbolic/tree data. Shrinking a `(B, H, S, D)` float tensor is not addressed anywhere I could
find. If that holds up, a principled array shrinker is a small but real contribution.

### 3.2 Why shrinking is architecturally awkward for us

Shrinking is inherently a **feedback loop**: propose a smaller case, *evaluate the property*,
keep or discard. For a Tier-1 property evaluated offline over recorded rows, "evaluate the
property" needs `y = K(x')` for a *new* `x'` — which means going back to the device. So:

- Shrinking cannot happen in Phase C. It is a separate Phase D with its own hardware budget.
- Shrinking cost should be reported in **kernel executions**, not wall time, and it belongs
  in the cost model of the oracle comparison. A property arm that finds a bug in 3 executions
  but needs 400 to minimize it is not obviously cheaper than one that finds it in 40 and
  minimizes in 20.
- **Metamorphic failures shrink badly by default**, because the minimizer must preserve the
  *group relation* while shrinking. Naively shrinking `x` and `x+c` independently destroys
  the relation. The shrinker must operate on `(base_case, transform)` and shrink the base,
  re-deriving the follow-up. This is the array analogue of Hypothesis's validity argument and
  it is a strong reason to shrink at the *case-spec* level rather than the tensor level.

### 3.3 What a domain-specific kernel shrinker should do

Ordered by expected value-per-unit-effort. Shrink **structure before values**, and shrink at
the level of the *case specification*, never the raw tensor.

1. **Shape reduction, dimension by dimension.** Binary-search each axis down, in a fixed
   order (batch → heads → sequence → feature). Preserve any shape *constraint* the kernel
   declares (multiple-of-16, power-of-two) — or, better, treat "does it still fail at a
   non-conforming shape?" as an additional bit of diagnostic information.
2. **Rank reduction.** Try dropping leading dims of size 1; try collapsing `(B,H,S,D)` to
   `(1,1,S,D)` early — this is usually the single biggest readability win.
3. **Dtype simplification.** If it fails in bf16, does it fail in fp32? A bug that survives
   the dtype ladder is a logic bug; one that vanishes is a precision bug. **This is a
   classification step disguised as a shrink, and it is the highest-information move
   available.** Do it second, before value shrinking.
4. **Value simplification, in this order:** (a) replace all non-essential elements with a
   single constant; (b) reduce magnitudes toward small integers; (c) drive toward exactly
   representable values (0.5, 1.0, 2.0) so the residual failure cannot be blamed on rounding;
   (d) only then remove special values (NaN/inf/subnormal/-0.0) one class at a time —
   and *report which special value was load-bearing*, since that is the diagnosis.
5. **Launch-config reduction.** Minimize grid/block/num_warps/num_stages/split_k. Often the
   counterexample is really about one tiling parameter.
6. **Group-relation preservation.** Shrink the base case; re-derive the follow-up from the
   transform. Never shrink follow-ups independently.
7. **Report the *un*-shrunk case too.** Hypothesis's "threshold problem" (referenced in
   Hatfield-Dodds 2020) is real here: the minimal failing case for a tolerance property is
   usually one that fails by `1.0001×` tolerance, which is maximally uninformative. For
   numeric oracles, **minimize the input while maximizing the violation magnitude** — i.e.
   shrink subject to `error/tol ≥ error₀/tol`, not merely `fails`. This is the single most
   important adaptation of shrinking to numerical testing and I did not find it stated
   anywhere; it falls out of combining `hypothesis.target()`'s logic with ddmin's.

Point 7 deserves emphasis. Standard shrinking optimizes for *smallness*. For a tolerance
oracle, smallness and diagnostic value are anti-correlated. A shrinker that preserves or
increases the violation ratio while reducing size is a different objective function, and
it is the right one.

---

## 4. Tolerance and floating point for oracle design

*(Consolidated with the numerical-testing literature findings; see README §5.)*

### 4.1 The default everyone actually uses, and why it is not defensible alone

`torch.testing.assert_close` compares `|actual − expected| ≤ atol + rtol·|expected|` with
documented per-dtype defaults:

| dtype | rtol | atol |
|-------|------|------|
| float64 | 1e-7 | 1e-7 |
| float32 | 1.3e-6 | 1e-5 |
| float16 | 1e-3 | 1e-5 |
| bfloat16 | **1.6e-2** | 1e-5 |
| other (ints, bool) | 0.0 | 0.0 |

Default `equal_nan=False`; infinities must match exactly; `check_dtype`, `check_device`,
`check_layout` default True, `check_stride` defaults False.

Three problems with using this as *the* oracle:

1. **The tolerance is dtype-indexed, not problem-indexed.** It knows nothing about the
   reduction length `n` or the conditioning of the input. A bf16 GEMM with `K = 8192` has a
   legitimate error budget vastly larger than one with `K = 8`; a fixed `rtol = 1.6e-2` is
   simultaneously too tight for the former and far too loose for the latter. **Too loose is
   the dangerous direction**: it is exactly the gap a wrong-but-close kernel hides in.
2. **`atol + rtol·|expected|` is meaningless where `expected ≈ 0`.** Catastrophic
   cancellation produces small outputs with large relative error, and `atol` then dominates
   and passes everything.
3. **It licenses tolerance laundering.** If the harness lets the kernel author pick the
   tolerance, the tolerance becomes a free parameter to make the test pass. This is the
   mechanism behind several of the reported kernel-benchmark integrity failures.

### 4.2 What to do instead: residual/backward-error checks

The strongest available prior art is the LAPACK test suite convention, which does not compare
outputs elementwise at all. It computes a normalized residual **test ratio**. Verified from
LAPACK Working Note 41 (Installation Guide), §7.1.1 and Tables 2/4, the ratios include:

| Test ratio | Checks |
|---|---|
| `‖LU − A‖ / (n ‖A‖ ε)` | factorization residual |
| `‖b − Ax‖ / (‖A‖ ‖x‖ ε)` | **solve backward-error residual** |
| `‖I − A A⁻¹‖ / (n ‖A‖ ‖A⁻¹‖ ε)` | inverse accuracy |
| `‖x − x̂‖ / (‖x̂‖ ε)` | forward error vs. computed solution |

and the sample input file in §7 carries the line `30.0   Threshold value of test ratio`.
**LAPACK's actual production tolerance is "the normalized residual must be under 30."**
That is the number to quote — a single dimensionless constant covering every routine, every
matrix size, and every precision. The crucial properties of this form:

- It is **dimensionless** and **scaled by `n·ε`**, so it automatically adapts to problem size
  and precision. `τ` is a genuine constant across dtypes and shapes, unlike `rtol`.
- It is a **backward-error** statement — "the computed answer is the exact answer to a nearby
  problem" — which is the correct notion of correctness for finite-precision linear algebra
  (Higham, *Accuracy and Stability of Numerical Algorithms*).
- It is robust to the near-zero problem, because the normalization uses norms, not elements.

**Recommendation: our reference-oracle arm should be `assert_close`-style *and* residual-style,
reported separately.** They are different oracles with different failure profiles, and
conflating them would understate the reference arm. GM-7 above is the residual form. I expect
the residual oracle to be strictly better than `allclose` and I expect that to be one of the
paper's cleaner results.

Two more comparison forms worth stealing, both read from source:

- **CUTLASS's symmetric relative test.** `include/cutlass/relatively_equal.h` uses
  `|a−b| < ε·(|a|+|b|)`, falling back to `|a−b| < ε·nonzero_floor` when either side is zero or
  `|a|+|b|` is below the floor. Production defaults in `tools/profiler/src/options.cu` are
  **ε = 0.05, nonzero_floor = 1/256**. (The profiler's `--epsilon` help text staledly claims
  the default is 0/bit-exact — trust the code.) This is better than `atol + rtol·|expected|`
  for us because it is **symmetric**: when comparing two backends there is no privileged
  "expected" side, and both are approximations.
- **CUTLASS's exact tier.** `test/unit/gemm/device/testbed.h` uses `TensorEquals` — the device
  GEMM unit tests demand **bit-exactness** against a host reference. That works only because
  the inputs are small integer-valued, where FP arithmetic is exact. This is a beautiful trick
  and we should adopt it as a mandatory first tier: *constrain the input distribution until the
  arithmetic is exact, then assert bitwise equality across all three backends.* Zero tolerance
  ambiguity, zero false positives, and it catches every indexing/masking/boundary bug.

Also note **JAX's undocumented n-scaling**: `jax._src.public_test_util._assert_numpy_close`
calls `np.testing.assert_allclose(atol=atol*a.size, rtol=rtol*b.size)` — it multiplies both
tolerances by the array element count. So JAX has independently arrived at "tolerance must
grow with problem size", just via a cruder (linear in `size`, not `√n`) correction than the
LAPACK/Higham–Mary form.

### 4.3 Reassociation: the bound you need for cross-backend and split-K

The classical worst-case bound for the error of a length-`n` floating-point sum is
`γ_n = n·u/(1 − n·u) ≈ n·u`, and for the inner products in a GEMM the standard bound is
`|fl(aᵀb) − aᵀb| ≤ γ_n · |a|ᵀ|b|`. That is the bound that licenses "different reduction
order ⇒ different answer, both correct."

But `n·u` is badly pessimistic in practice. Higham & Mary's probabilistic rounding-error
analysis (SISC 41(5), 2019) replaces the deterministic `γ_n` with a relaxed constant
proportional to **`√(n log n)·u`**, holding with probability bounded below independently of
`n`; the bounds are backward-error bounds, exact rather than first-order, and valid for any
finite `n`. Their 2020 follow-up (SISC 42(5)) sharpens this further: **for data with zero or
small mean the bound becomes O(u), independent of `n` entirely**, proved for summation and
extended to inner products and matrix products.

This has a direct, slightly uncomfortable consequence for us: **the correct tolerance depends
on the generator's data distribution, not just the dtype.** Zero-mean random tensors (our
default) sit in the n-independent regime. But the numerators inside softmax are `exp(·)` — all
positive — which puts the softmax denominator squarely back in the `√(n log n)` regime. So a
single tolerance policy cannot serve both, and the policy must be declared per-property,
derived from what the generator produces.

**This is the number to use for a *practically tight* reassociation tolerance**, and it is the
difference between a tolerance that catches bugs and one that does not:

| n (reduction length) | `n·u` (bf16, u≈2⁻⁸) | `√n·u` |
|---|---|---|
| 64 | 0.25 | 0.031 |
| 1024 | 4.0 (vacuous) | 0.125 |
| 8192 | 32 (vacuous) | 0.35 |

For bf16 with long reductions the deterministic bound is *vacuous* — it permits any answer.
This is a concrete, quantitative argument that **for low-precision long-reduction kernels the
reference oracle is not merely expensive, it is close to informationless**, and the
tolerance-free declarative properties (SM-5, LN-5, AT-2, AT-5, GM-4, GM-8) are the only
oracles carrying real signal. If that argument holds up under our own measurements, it is
the single strongest result the project could produce.

Practical guidance:
- **fp64/fp32**: `allclose` defaults are fine; the residual check is better.
- **fp16/bf16**: never use a fixed `rtol`. Use `τ · √n · u` with `τ` a small constant
  (start at 4–10, calibrate on known-good kernels), or accumulate the reference in fp32/fp64
  and compare against a *deterministic* rounding of it.
- **fp8 (E4M3/E5M2, per the OCP 8-bit FP spec)**: elementwise comparison against a
  higher-precision reference is nearly meaningless. Use structural and order properties, and
  compare *distributions* / relative error statistics rather than elements.
- **Always compute the reference in the highest precision available** (fp64 on CPU), and
  compare `kernel_output` against *both* `round_to_dtype(fp64_ref)` and `fp64_ref` — the gap
  between those two is the legitimate rounding budget, and reporting it makes the tolerance
  argument auditable instead of arbitrary.
- **Report the tolerance as a derived quantity, never as a knob.** Any tolerance the kernel
  author can adjust is not an oracle.

### 4.4 Tensor cores have a *systematic* bias, and it breaks the √n argument

Fasi, Higham, Mikaitis & Pranesh, *Numerical Behavior of NVIDIA Tensor Cores* (PeerJ CS 7:e330,
2021) characterized V100/T4 tensor cores by black-box experiment, because NVIDIA's PTX docs
state only that accumulation is "at least single precision" and that the accumulation order,
rounding, and subnormal handling are **unspecified**. What they found:

- Subnormals are natively supported (not flushed).
- Products are held **exactly** in a wider intermediate and are *not* rounded back to binary16
  before accumulation.
- Each output element incurs **at most four rounding errors**.
- Partial products are **not accumulated in a fixed order** — accumulation starts from the
  largest magnitude (the hardware sorts for significand alignment).
- **The additions use round-toward-zero, not round-to-nearest.**
- Only the final dot-product result is normalized; the internal accumulator carries at least
  2 extra significand bits.

The fifth point is the one that matters. **Round-toward-zero is a biased error, not a
zero-mean one**, so errors accumulate roughly linearly rather than cancelling as `√n`. Any
tolerance derived under a round-to-nearest, zero-mean assumption is *too tight* for a
tensor-core GEMM. Either add an explicit bias term, or — better — compute the reference through
the block-FMA model of Blanchard, Higham, Lopez, Mary & Pranesh (SISC 42(3), 2020) rather than
an idealized fp64 dot product.

**Action item: there is no published numerical characterization of AWS Trainium at all.** Until
we know its rounding mode, accumulation order, and subnormal handling, any cross-backend
tolerance for NKI is a guess — and a tolerance wide enough to cover an unknown systematic bias
is wide enough to hide real bugs. Running the Fasi et al. probe methodology against Trainium
is a self-contained, publishable-in-its-own-right piece of work, and it is a prerequisite for
the NKI arm of the study. Their test suite is public:
<https://github.com/north-numerical-computing/tensor-cores-numerical-behavior>.

### 4.5 Deriving tolerances instead of guessing them

The highest-leverage addition beyond fixed constants: run the fp64 reference N times under
**Monte Carlo Arithmetic** (Verificarlo, or an MCA shim) and take the spread of the outputs.
That spread is the number of digits the computation actually determines, and it is a *derived,
auditable* tolerance rather than a constant someone picked. Croci, Fasi, Higham, Mary &
Mikaitis (Royal Society Open Science 9:211631, 2022) is the citation, and it clarifies the tool
taxonomy: CADNA uses stochastic arithmetic (CESTAC); Verrou and Verificarlo use MCA, which is
strictly more general than stochastic rounding because it perturbs operation inputs as well as
rounding the result.

### 4.6 Reproducibility: set expectations with citations, not opinions

When someone asks why we can't just assert bitwise equality across CUDA/Triton/NKI:

- Bitwise-reproducible summation **is** achievable — Ahrens, Demmel & Nguyen's binned
  accumulators (ACM TOMS 46(3), 2020) give bitwise-identical results across different hardware
  resource counts in one read-only pass and one parallel reduction — but it costs **≈9n
  floating-point ops plus ≈3n bitwise ops** to sum n words, versus n for naive summation.
  No production GPU kernel will pay that.
- Intel, with total control over one CPU library, offers reproducibility only *conditionally*:
  oneMKL CNR requires a single executable and an unchanging thread count; Strict CNR lifts the
  thread-count requirement but only for `?gemm`, `?symm`, `?hemm`, `?trsm`, with an explicit
  performance warning.

Those two facts together are the cleanest possible argument that the oracle must be
tolerance-based (or tolerance-free-by-construction, per §1.2), not equality-based.

### 4.7 Non-determinism to plan for
- Run-to-run determinism is itself a testable property (bitwise identity across two
  executions of the same kernel on the same input), and it is tolerance-free. Atomics-based
  reductions will fail it; that may be intended, but it should be *declared*, not discovered.
- FTZ/DAZ behaviour differs by backend and by compiler flag; subnormal handling is a common
  silent divergence. Test it explicitly rather than filtering subnormals out of the generator.
- NaN/inf *propagation structure* (which output positions are NaN, and the sign of infinities)
  is a tolerance-free property and should be checked separately from values.

---

## 5. Tier-2 telemetry, generator design, and what the GPU literature constrains

### 5.1 compute-sanitizer: what it actually gives you, verified from NVIDIA's docs

`--tool` accepts exactly four values — `memcheck` (default), `racecheck`, `initcheck`,
`synccheck`. There is no fifth. Facts that shape the Tier-2 design:

- **`racecheck` only detects races in on-chip shared memory.** The docs are explicit:
  "Currently, this tool only supports detecting accesses to on-chip shared memory." **No
  compute-sanitizer subtool detects global-memory data races.** If our Tier-2 property is
  "kernel is race-free", that property is *false as stated* — it is "kernel is
  shared-memory-race-free." Say so, or a reviewer will. Global-memory races need iGUARD
  (SOSP 2021, runs detection on the GPU, covers scoped sync and Volta+ independent thread
  scheduling) or CURD (PLDI 2018).
- **The four tools do not compose.** racecheck and initcheck explicitly do not do
  memory-access error checking; you need four separate runs. That is a 4× multiplier on the
  Tier-2 execution budget and must be in the cost model.
- **CI hook**: `--error-exitcode <n>` exists specifically "to allow Compute Sanitizer to be
  integrated into automated test suites." That is the gate. `--suppressions <file.xml>`
  baselines known-benign warnings.
- **Output is XML only** (`--xml` + `--save`); there is no JSON. Our telemetry ingester needs
  an XML parser, not a JSON one.
- `--padding <bytes>` adds a red zone after every allocation so that back-to-back allocations
  cannot mask an OOB. Turn this on; it materially raises memcheck's sensitivity.
- `cuda-memcheck` was deprecated in CUDA 11.6 and **removed in 12.0**. Don't reference it.
- **There is no device-side AddressSanitizer on NVIDIA** (LLVM disabled sanitizers on NVPTX)
  and **no ThreadSanitizer for GPU device code on either vendor**. AMD ROCm *does* have a real
  device ASan (needs `--offload-arch=gfx90a:xnack+` and `HSA_XNACK=1`), with documented gaps:
  private/stack variables uninstrumented, `__shared__` modeled in global memory.

**The strongest argument for our whole approach comes from this corner.** Ghosh et al.,
*Chasing Elusive Memory Bugs in GPU Programs* (arXiv:2601.21552) documents **input-dependent
out-of-bounds accesses that only manifest under specific inputs and therefore elude all
existing runtime tools including compute-sanitizer**, plus intra-allocation OOBs that
allocation-granularity checkers structurally cannot see. Sanitizers are input-triggered
detectors with no input-generation strategy of their own. **A property-based generator layer
sitting on top of compute-sanitizer is exactly the complement**, and this is a peer-reviewable
framing of the Tier-2 contribution that does not depend on any of the 2026 preprints.

### 5.2 Generator design: one finding that should change our defaults

Sarkar's input-generation ablation (arXiv:2606.27396) compared seven strategies across 26 ops:

| Strategy | Recall | False positives |
|---|---|---|
| Boundary-only **shape** sampling | 78% | **0%** |
| Adversarial **value** sampling (NaN/Inf injection) | 99% | **94%** |

That 94% FP rate is the trap. Injecting NaN/Inf into inputs trips validators on *correct*
kernels, so the "high-recall" strategy is operationally useless. Meanwhile boundary shape
sampling was decisive for softmax tail-mask bugs (0% detection under regular strategies vs
100%/62% with boundary sampling). Caveat honestly: this is a solo-author unrefereed preprint
on self-seeded bugs, so treat the numbers as a hypothesis to re-measure, not a result. But
the *direction* is corroborated by peer-reviewed work: Rathnasuriya et al. (ISSTA 2026), which
characterizes 301 real code-generation bugs in Triton/TileLang tile programs, finds bugs
"tightly coupled to input shapes, data types, and backend targets" and frequently surfacing as
**silent wrong results**.

**Implication: our generator's primary axis should be (shape × dtype × backend), not element
values.** Element-value adversariality should be a separate, clearly-labelled tier whose FP
rate is measured and reported, not folded into the default corpus. This is a concrete design
decision the literature settles, and it is cheap to get wrong.

Also worth adopting: NNSmith's (ASPLOS 2023) **gradient-based search over inputs to avoid
floating-point exceptional values**, which exists precisely because NaN/Inf inputs cause both
missed bugs and false alarms. And Groce et al.'s **swarm testing** (ISSTA 2012): randomly omit
*features* per test run (e.g. disable masking, disable one dtype, restrict to one layout) to
get diversity that a single uniform configuration cannot reach. Swarm is a nearly-free addition
to a batch generator and it directly addresses the "our corpus is homogeneous" threat.

### 5.3 Three oracle ideas worth stealing outright

- **Status equivalence** (DeepREL, FSE 2022). Beyond value equivalence: kernel and reference
  must **both succeed or both raise**. Catches shape/bounds/dtype bugs with **no numerical
  tolerance at all**. This is a tolerance-free reference-oracle variant and it costs nothing.
  It belongs in the catalogue.
- **Equivalence rules within a single implementation** (EAGLE, ICSE 2022 — 17 hand-written
  rules, 20 bugs, no second backend required). **This is the answer for NKI**, where there is
  no second Trainium implementation to differentially test against. Note honestly that EAGLE
  found *fewer* bugs than mining-based fuzzers — but they were silent wrong-answer bugs, the
  expensive kind.
- **Design transformations so that shrinking is free** (spirv-fuzz, PLDI 2021). If your
  metamorphic transformations are *small and independent*, plain delta debugging over the
  *transformation sequence* gives reduction for free, and the minimized sequence also
  heuristically deduplicates bug reports. This is an architectural decision that must be made
  up front, not retrofitted — and it interacts directly with §3.3: shrink the case *spec*
  (a sequence of independent transforms), never the tensor.

### 5.4 Where formal methods already win, and therefore where we shouldn't compete

**VOLTA** (Dubey, Driscoll, Wei, Kayal, Sharma, Aiken — arXiv:2511.12638) is the first formal
equivalence checker for GPU kernels: **sound and complete** for a well-defined class where tile
sizes are statically known and per-thread branch targets and addresses are statically fixed
given `tid`. It verifies convolutions, matmuls, and attention. **Our value proposition is
precisely the kernels outside that statically-analyzable class** — data-dependent control flow,
dynamic shapes, autotuned configurations, and anything where the backend's numerics are
unspecified. Read VOLTA before finalizing scope and state the boundary explicitly; a reviewer
who knows it will otherwise ask why we are testing what can be proved.

### 5.5 The NKI situation, stated plainly

Verified from AWS's own documentation:

- `nki.simulate_kernel` runs on CPU via the Neuron compiler's simulator; all tensors must be
  `numpy.ndarray`; **it has no built-in correctness checking** — the doc's own example does a
  manual `np.allclose()`.
- `nki.baremetal` runs on a real NeuronDevice and emits NEFF + NTFF trace; correctness is again
  shown only via `assert np.allclose()`.
- The NKI getting-started guide contains **no validation guidance at all**.
- The real policy lives in `aws-neuron/nki-samples/CONTRIBUTING.md`: reference kernels need
  numeric accuracy tests via `nki.baremetal` against a CPU reference plus p99 latency
  assertions, and notably **"Kernels with only simulation tests will not be accepted."**

**So the entire NKI correctness story is "hand-write a NumPy reference and call `np.allclose`
on fixed shapes"** — precisely the oracle the 2026 Triton work demonstrates to be inadequate,
with no published testing or verification research on NKI whatsoever. This is the project's
most defensible novelty claim and the cheapest to defend.

---

## 6. Measuring bug-finding power defensibly

*(See README §7 for the sources.)*

### 6.1 The threat you must answer

The reviewer's question will be: *"your properties found more bugs — but were they the same
bugs, and did you just run more tests?"* Everything below is aimed at that.

Three papers settle the argument between them, and the reconciliation matters:

- **Andrews, Briand & Labiche (ICSE 2005)** concluded that "mutants, when using carefully
  selected mutation operators and after removing equivalent mutants, can provide a good
  indication of the fault detection ability of a test suite." They also found **hand-seeded
  faults were *harder* to detect than real faults** — human fault-seeding biases *against* the
  technique under test, which is convenient for us.
- **Just et al. (FSE 2014)**, 357 real Java faults: the coupling effect holds for **73%** of
  real faults, the mutation-score↔real-fault correlation **survives controlling for code
  coverage**, and is stronger than the statement-coverage correlation. But **17% of real faults
  are coupled to no mutant at all** — they call this "a fundamental limitation of mutation
  analysis." Quote that 17% as your own stated ceiling before a reviewer does.
- **Papadakis et al. (ICSE 2018)** is the rebuttal to the naive reading: reported correlations
  are largely "the results of the confounding effects of the test suite size." Uncontrolled
  correlations sit at 0.35–0.75; **controlling for suite size collapses them to ~0.05–0.20**
  on both Defects4J and CoreBench. But the part people skip: selecting the *top-ranked* suites
  by mutation score (rather than random suites of equal size) "improves significantly the fault
  detection."

**Reconciliation, stated defensibly:** mutation score is a usable *ordinal* comparator between
oracle strategies **only when the test budget is held fixed**, and its absolute value is not a
fault-detection probability. Our batch-first design holds the budget fixed by construction —
every arm sees the identical recorded batch. That is not a nice-to-have; it is the thing that
makes the comparison survive the Papadakis critique.

### 6.2 Recommended protocol

**Fault corpus — use three sources, and report them separately.**

1. **Mutants** (systematic, high N, weak individual validity). **Author them by hand**, as Etna
   (ICFP 2023) does, explicitly "allowing us to more readily maintain ground truth and ensure
   that every mutant violates some aspect of the property specification." Hand-authoring
   sidesteps the equivalent-mutant problem by construction: you never inject a mutant you have
   not confirmed is non-equivalent on at least one input. Bias operator choice toward what Just
   et al. found actually couples to real faults — conditional-operator replacement,
   relational-operator replacement, statement deletion — plus these kernel-specific classes
   that generic operators miss:
   - off-by-one in loop/tile bounds; `<` ↔ `<=`
   - swapped indices / strides (`i*ldb + j` ↔ `j*lda + i`)
   - dropped or moved `__syncthreads()` / `tl.debug_barrier()`
   - wrong reduction axis
   - accumulator dtype demotion (fp32 → fp16)
   - removed max-subtraction in softmax (the numerical-stability mutant)
   - wrong or missing mask on the boundary tile
   - `eps` dropped or moved inside/outside the sqrt
   - transposed operand in the MMA call
   - off-by-one in the causal mask
2. **Real historical bugs** (low N, high validity). Mine git history of Triton/CUTLASS/
   FlashAttention/vLLM kernels for fix commits and reconstruct the pre-fix kernel. Even
   10–20 of these is worth more to a reviewer than 1,000 mutants, and it is the
   Defects4J-style move that testing venues expect.

   ⚠️ **There is no Defects4J for GPU kernels or numerical code, and a reviewer will ask.** The
   nearest artifacts are Di Franco, Guo & Rubio-González's ASE 2017 characterization of 269
   real numerical bugs (NumPy/SciPy/LAPACK/GSL/Elemental — a bug *list*, not a runnable
   harness; correctness bugs are the largest class at ~37%, and 76/269 involve special values)
   and Rathnasuriya et al.'s ISSTA 2025 study of 397 real GPU numerical bugs mined from GitHub.
   Neither is a reproducible benchmark. **We have to construct our own corpus, which is exactly
   why the mutant methodology above has to be airtight.** Say this explicitly in threats to
   validity rather than letting it be discovered.
3. **LLM-generated wrong kernels** (medium N, high external validity, and *this is the
   population we actually care about*). Take model-generated kernels that pass the existing
   reference check and audit them. This is the arm that speaks directly to the project's
   motivation, and it is what the recent kernel-verification preprints do.

   For the *secondary*, automated mutant corpus (run a mutation tool at scale, to defuse "you
   cherry-picked your mutants"), filter in three stages and **report the survival rate at each**:
   (i) drop compile failures and mutants no input covers; (ii) drop duplicates by compiler
   equivalence (TCE-style: identical optimized binaries ⇒ equivalent) — the Papadakis 2019
   survey reports this removes up to 90% of equivalents at ≤10% effectiveness loss; (iii)
   compute the **dynamic disjoint/subsuming mutant set** and report scores on both the full and
   the disjoint set. Redundancy is not a minor concern: reported disjoint-mutant fractions are
   ~9% (Kintis et al.), 1.2% minimal mutants (Ammann et al.), 4% (Kurtz et al.), and the survey
   estimates a **>60% chance of compromised conclusions** in arbitrary experiments that ignore
   subsumption. A reviewer who knows this literature will look for exactly this table.

**Unit of measurement.** Adopt Etna's **task = (mutant, oracle configuration)** where the mutant
is known to be detectable in principle, and classify each task per arm as:

- **solved** — detected in *all* trials within budget
- **partially solved** — detected in ≥1 trial and missed in ≥1
- **unsolved**

Report all three. The weak-property arm will live disproportionately in "partially solved", and
collapsing that into a mean time-to-detect would hide the single most interesting effect in the
study. Etna presents this as a **task bucket chart** rather than a mean; copy that.

**Scoring.**
- Primary metric: **mutation score per oracle strategy** = fraction of non-equivalent mutants
  killed, over the *same recorded execution batch*. The batch-first design is what makes this
  comparison clean; say so loudly.
- **Report the hybrid arm's marginal contribution explicitly**: the set of tasks solved *only*
  by properties and *only* by the reference oracle. This is the actual scientific claim, and —
  importantly — it does not depend on any correlation argument, so it survives the Papadakis
  critique untouched. Lead with it.
- **Handle equivalent mutants explicitly.** For numerical kernels a large fraction of mutants
  will be *tolerance-equivalent* — they change the answer by less than any defensible
  tolerance. Do not silently drop them. Report them as their own category and note that they
  are precisely the mutants a reference oracle *cannot* catch by construction. That category
  is an argument for the declarative arm, not a nuisance.
- Report **detection rate split by whether the property required a tolerance** (see §1.2).
- Report **cost per bug caught**, in kernel executions and in authoring time (minutes to write
  the property set for one op). Nobody in the kernel literature reports oracle cost; this is
  cheap to measure and differentiating.
**False positives — use Google's definition, and pre-register a budget.**

Sadowski et al., *Lessons from Building Static Analysis Tools at Google* (CACM 61(4), 2018),
gives the definition to adopt wholesale:

> We consider an issue to be an "effective false positive" if developers did not take positive
> action after seeing the issue. If an analysis incorrectly reports an issue, but developers
> make the fix anyway to improve code readability or maintainability, that is not an effective
> false positive. If an analysis reports an actual fault, but the developer did not understand
> the fault and therefore took no action, that is an effective false positive.

Their operating thresholds: compile-time checks must be essentially zero-FP; **code-review
checks are allowed up to 10% effective false positives**, and Tricorder **auto-disables an
analyzer whose "not useful" click ratio exceeds 10%**.

Concretely for us: run every property against hand-verified-correct kernels (cuBLAS, PyTorch
native, reference Triton, reference NKI) and report per property — alarms raised, true bugs,
effective FPs, and an FP *cause* taxonomy (tolerance, legal reduction-order non-determinism,
unspecified NaN payload, generator producing out-of-contract input). Pre-register a **≤10%
effective-FP budget per property**, have alarms triaged by someone blind to which arm produced
them, and report inter-rater agreement. A property that exceeds the budget stays **in the
results table but out of the gated suite** — that mirrors Tricorder's disable-and-fix loop and
is far more honest than silently dropping it. A property arm with a 5% FP rate is unusable in
an agentic inner loop no matter what its detection rate is; this number decides deployability.

**Statistics.** Follow the fuzzing-evaluation hygiene rules (Klees et al., CCS 2018) and Arcuri
& Briand (STVR 2014). Their concrete instructions, verified:

- **Trials.** Arcuri & Briand's ideal is n ≥ 1000 runs per artifact, but they explicitly license
  the trade-off we need: "when a large number of artifacts can be used… but there are
  constraints in terms of execution time, then it is advisable to execute less runs per artifact
  (**though at least n = 10**) and use more artifacts." Etna defaults to 10 trials; Klees ran
  ≥30. **Use 30 per (arm, task) if per-run cost allows, 10 as the floor, and report total
  compute time so the choice is auditable.**
- **Tests.** Mann–Whitney U, not a t-test, because it makes no distributional assumptions —
  Klees, Etna and Arcuri & Briand all say this. **Always pair p-values with a non-parametric
  effect size, Vargha–Delaney Â₁₂**: Arcuri & Briand note that with enough runs "one would
  detect statistically significant differences on practically any experiment", so the effect
  size carries the claim, not the p-value. Correct for multiple comparisons (Holm or Bonferroni)
  across the arm-pair × workload grid, and say so. For solved/unsolved counts use McNemar or a
  paired proportion test rather than U.
- **Plot over time; don't report a single endpoint.** Klees demonstrate a case where AFLFast
  beats AFL at 5 h and the trend **reverses by 24 h** (p<10⁻¹³ one way, p=0.000105 the other).
  Their 24-hour floor is calibrated to whole-program fuzzing and does not transfer literally to
  kernel testing, but the underlying instruction does: justify the timeout and plot detection
  rate against budget.
- **Count ground-truth bugs, not failures.** Klees found AFL's coverage-unique crash counting
  inflates one bug into ~500 "unique" crashes, and even fuzzy stack hashing over-counts (~46
  hashes for one bug) *and* can under-count. Two properties failing on the same underlying fault
  is one bug. Deduplicate by root cause before counting.
- **Equalize the budget on two axes and report both.** Equal *number of generated inputs* and
  equal *wall-clock*. Reference oracles are far more expensive per input, so these two
  comparisons will disagree — and that disagreement is a genuine finding about deployability,
  not a nuisance to be averaged away.
- Coverage is a **secondary** measure only (Klees: "block or edge coverage can be used as a
  secondary measure"), never the headline.

**Known pitfalls to pre-empt in the writeup.**
- *Coverage confound* (Papadakis et al., ICSE 2018 found the mutation–real-fault correlation
  substantially weakens after controlling for coverage): control for the number of executions
  each arm sees. Our design does this by construction — same batch, different oracles — and
  that is the strongest methodological card we hold.
- *Mutant validity*: mutants that don't compile, or that fail on every input, inflate scores.
  Filter to mutants that compile and that pass at least one input.
- *Property-set cherry-picking*: pre-register the property catalogue per operator before
  running the mutants, or the declarative arm's score is not credible. Stronger: **have the
  mutants authored by someone other than the property author.** Note in your defence that
  Andrews et al. found hand-seeded faults are *harder* to detect than real ones, so
  hand-seeding biases against the technique under test, not for it.
- *Subject selection*: Klees found the median fuzzing paper used 7 programs with almost no
  overlap between papers. Name the kernel set, justify it, and release it.
- *Low-strength suites are noise*: Papadakis et al. 2019, citing Chekam et al., report "there
  is no practical difference between test criteria when relatively low-strength test suites are
  used", and that two studies below the strength threshold "may yield different findings, even
  when the experimenters follow identical experimental procedures." Make sure the batch is
  large enough that all arms are above that floor, and report the batch size.
- *Tolerance as a free parameter*: fix the tolerance policy before scoring, and report results
  under at least two policies (`allclose` defaults and residual/√n·u) to show the conclusion
  is not a tolerance artifact.

---

## 7. Open questions and where this project is novel

### 7.1 What is no longer novel (cite, don't claim)

The window narrowed substantially in mid-2026. These are established and must be cited as
background:

- "Reference/`allclose` oracles for LLM-generated kernels are inadequate" —
  arXiv:2606.20128, arXiv:2608.12700, arXiv:2509.14279, and the Sakana AI CUDA Engineer
  reward-hacking episode.
- "Declarative contracts/properties for ML kernels" as a *concept*, including a concrete
  twelve-item property catalogue — arXiv:2604.22032 (Kernel Contracts) + arXiv:2608.12700.
- "LLMs can author properties / MRs / postconditions" — arXiv:2307.04346, arXiv:2401.17019,
  arXiv:2310.01831 (nl2postcond).
- "Property oracles beat example oracles for LLM-generated code", and even "hybrid beats
  either" — arXiv:2506.18315 and arXiv:2510.25297. The latter is the sleeper threat: it
  already reports PBT-alone 68.75%, EBT-alone 68.75%, combined 81.25%. It is 16 HumanEval
  problems with no numerics and no kernels, so the differentiation is domain and scale —
  but the *shape* of our hybrid result is pre-empted unless we say why kernels differ.

### 7.2 What is still open

1. **Oracle strategy as the independent variable, scored by fault detection.** No paper runs
   reference vs declarative vs hybrid as a controlled three-arm comparison for kernels. And
   the 2026 PRISMA review of 83 LLM-oracle studies (arXiv:2607.05031) states outright that
   "Oracle quality is most often judged by resemblance to a known oracle rather than by
   whether injected faults are caught." That is a citable, authoritative statement of exactly
   our gap.
2. **Batch record/replay so every oracle sees byte-identical executions.** Genuinely
   unclaimed, and architecturally *contradicted* by the nearest competitor, which re-seeds
   and re-executes per gate — meaning its per-gate numbers are confounded by execution
   variance. This is a methodological contribution we can defend on first principles.
3. **Cross-backend oracle portability (CUDA / Triton / NKI).** Completely open. No PBT or
   metamorphic work targets AWS Trainium/NKI at all. "Does the same declarative property set
   transfer across three structurally different programming models, and which properties
   don't?" is a clean, answerable, unclaimed question.
4. **Cost-per-bug as an oracle-selection axis.** No kernel paper reports oracle cost. It
   matters enormously for an agentic/RL inner loop, where an oracle's latency caps the search
   rate.
5. **The tolerance-free subset hypothesis** (§1.2, §4.3): that in low-precision,
   long-reduction regimes the tolerance-based reference oracle is close to informationless
   and tolerance-free structural properties dominate. I have not seen this stated or measured.
   It is the most interesting thing in these notes and it is directly testable with our setup.
6. **Batch-axis independence as a named metamorphic relation** (LN-5, AT-5). Simple, general,
   tolerance-free, targets the dominant tiling bug class, and I could not find it named
   anywhere.
7. **Shrinking numeric array inputs**, and specifically shrinking that *preserves or maximizes
   the violation ratio* rather than only minimizing size (§3.3 point 7). No prior art found.
   Relatedly: no QuickCheck/Hypothesis-style PBT **with shrinking** has been applied to GPU
   kernels at all. Propilot is PBT for AI *compilers* and does not shrink.
8. **Numerical characterization of AWS Trainium** (§4.4). No published account of its rounding
   mode, accumulation order, subnormal handling, or accumulator width exists. The Fasi et al.
   tensor-core methodology and its public test suite give a ready template. This is a
   self-contained piece of work and a prerequisite for any defensible NKI tolerance.
9. **A cross-backend property layer for CUDA + Triton + NKI.** No differential testing layer
   spans these three. There is also no dedicated XLA / JAX / StableHLO fuzzing paper (WhiteFox
   treats XLA as one of three targets) — "the same StableHLO program on different backends must
   agree" is an obvious metamorphic target nobody has published on.
10. **A validated notion of "property coverage."** Nothing defines and empirically validates a
    coverage measure over *properties* (as opposed to over inputs or code). Nearest neighbours
    are combinatorial coverage of inputs (Goldstein et al., ESOP 2021) and Tyche's distribution
    visualizations (UIST 2024). Vikram et al.'s property-coverage-via-property-mutants
    (arXiv:2307.04346) is the closest operational metric and is directly reusable.

### 7.3 Recommended repositioning

Stop framing the contribution as "properties beat reference outputs" — that is now settled
literature. Frame it as:

> **Which oracle strategy, at what cost, catches which fault classes — measured over identical
> replayed executions, across three backends, with tolerance-free and tolerance-dependent
> detection reported separately.**

Treat the published property catalogues (Kernel Contracts' twelve gates, Propilot's tensor
property skeletons, EAGLE's equivalence rules) as *inputs* to the declarative arm rather than
as competitors. Cite arXiv:2607.05031's fault-detection gap as the motivation.

**Caution.** The most threatening kernel papers (2606.20128, 2606.27396, 2608.12700,
2604.22032) are solo- or duo-author, non-peer-reviewed arXiv preprints, two by the same
author. Their headline numbers deserve independent scrutiny before we build on them.
Propilot (2606.06747, Miryung Kim's group) is the one with strong pedigree. And this area is
moving at roughly one directly-competing preprint per month — **re-run the prior-art search
immediately before any submission.**

---

## 8. Unverified / not found

Things I looked for and could not confirm, or confirmed only indirectly. Do not cite these as
read without checking them yourself.

**Read only via secondary sources (metadata confirmed, contents not read from primary):**
- Chen, Kuo, Liu, Poon, Towey, Tse, Zhou, "Metamorphic Testing: A Review of Challenges and
  Opportunities", ACM CSUR 51(1) Art. 4, 2018, DOI 10.1145/3143561. ACM DL and the HKU/
  Nottingham OA mirrors returned 403.
- Chen, Jia, Yan, Ge, Zheng, Cheng, "A Miss Is as Good as A Mile: Metamorphic Testing for
  Deep Learning Operators", PACMSE 1(FSE) Art. 89, 2024, DOI 10.1145/3660796. **This is
  probably the single most relevant paper in the metamorphic literature to us** — 21 MRs for
  10 DL operators, explicitly targeting precision errors as well as logic errors. It is gold
  open access but ACM bot-blocked every automated fetch attempt. **Get it manually.**
- MT-DLComp (SIGMETRICS/POMACS 2022), Predoo (ISSTA 2021), Duo (IEEE Trans. Reliability 2021),
  LEMON (FSE 2020), CRADLE (ICSE 2019): venue/year verified, abstracts read from conference
  program pages and secondary summaries rather than the papers.
- LAPACK `THRESH = 30.0` convention: the residual-test-ratio *form* is well documented in the
  LAPACK Working Notes; I am confident about the form and about the existence of a small
  constant threshold, less so about quoting 30.0 as universal across all drivers. Check
  `LAPACK/TESTING/*.in` before citing a number.

**Searched for, apparently does not exist:**
- Any paper applying metamorphic testing to CUDA/Triton/accelerator *compute kernels* as
  such — i.e. MRs over tensor semantics executed on a GPU. Donaldson's work is graphics
  shader compilers; Meta/MT-DLComp/Propilot operate at the DL-operator or compiler level;
  Kaizen is LLM code translation. I did not exhaustively enumerate MET workshop proceedings
  2016–2024, so this is "not found on targeted search", not "proven absent."
- Any PBT or metamorphic work targeting AWS Trainium / NKI. NKI-Agent (arXiv:2607.04395) is
  the only NKI kernel-generation paper found and it verifies by compile-and-compare only.
- Any work on shrinking numeric *array* inputs.
- Any controlled comparison of oracle strategies for compute kernels.
- "PBT-GPT" — no paper, tool, or repo by that name. Do not cite it.

**Paywalled — metadata verified, contents not read. Get these manually if they become
load-bearing:**
- **Predoo: Precision Testing of Deep Learning Operators** (Zhang et al., ISSTA 2021). It is
  about establishing what numerical tolerance is *defensible per operator* — the central
  question our oracle must answer. High priority.
- **MT-DLComp: Metamorphic Testing of Deep Learning Compilers** (Xiao et al., SIGMETRICS/
  POMACS 2022).
- **An Investigation on Numerical Bugs in GPU Programs** (Rathnasuriya et al., ISSTA 2025) —
  397 real GPU numerical bugs. ACM 403s despite PACMSE being gold OA.
- BARRACUDA (PLDI 2017), CURD (PLDI 2018), GKLEE (PPoPP 2012), GRace (PPoPP 2011),
  MLIRSmith (ASE 2023), MLIRod (ISSTA 2024), GPUHarbor (ISSTA 2023).
- Higham, *Accuracy and Stability of Numerical Algorithms* (SIAM 2002) and Muller et al.,
  *Handbook of Floating-Point Arithmetic* (2018) — copyrighted books, no legitimate free PDF.
  Cite by chapter, carefully. The pairwise-summation `O(log n · u)` bound is asserted here from
  general knowledge, **not** verified against Higham's 1993 SIAM JSC paper, which is paywalled.
- DeMillo/Lipton/Sayward 1978; Jia & Harman's *published* TSE 2011 version (we have the CREST
  tech-report preprint, which may differ in pagination and minor content).
- FPDebug (PLDI 2012), S3FP (PPoPP 2014), Verrou — descriptions come from tool READMEs and
  citing papers, not the papers themselves.
- BugsInPy (FSE 2020) — the "493 bugs, 17 projects" figure is from listing pages and the GitHub
  README, not the paper.

**Bibliographic corrections found along the way** (fix these before any submission):
- **MLIRod is ISSTA 2024, not ICSE 2024.**
- **FuzzGPT's published title differs from its arXiv title**: the ICSE 2024 version is
  "Large Language Models are Edge-Case **Generators**…", not "…Fuzzers…". Cite the published one.
- **Lustig et al., ASPLOS 2019** is "A Formal Analysis of the NVIDIA PTX Memory **Consistency**
  Model."
- **CLsmith used 21 (device, compiler) configurations**, not the 19 several secondary sources
  report. Verified from the PDF.
- **Athena and Hermes are tool names, not paper titles.** Athena = "Finding Deep Compiler Bugs
  via Guided Stochastic Program Mutation" (OOPSLA 2015); Hermes = "Finding Compiler Bugs via
  Live Code Mutation" (OOPSLA 2016).
- **"A Survey of Compiler Testing" (Chen et al., CSUR 2020) and "Compiler Testing: A Systematic
  Literature Analysis" (Tang et al., FCS 2020) are two different papers.** The first is
  substantive; the second is bibliometric. Cite Chen et al. for content.
- **TVMfuzz is not a publication** — it exists only as a baseline inside Tzer's evaluation.
- **The OCP OFP8 Rev 1.0 (June 2023) spec contradicts itself**: §4.2 gives E4M3 bias 15 and
  E5M2 bias 7, while its own normative Table 1 gives E4M3 bias 7 and E5M2 bias 15. **Trust
  Table 1** (it matches every implementation). Prefer the December 2023 revision.
- **The CUTLASS profiler's `--epsilon` help text is stale**: it claims the default is 0
  (bit-exact) while the code default in `tools/profiler/src/options.cu` is 0.05. Report the code.

**Corrections to assumptions the project was carrying:**
- "Can Large Language Models Write Good Property-Based Tests?" (arXiv:2307.04346) is by
  **Vasudev Vikram, Caroline Lemieux, Joshua Sunshine, Rohan Padhye** — *not* Vasudevan or
  Goldstein. Harrison Goldstein is a PBT researcher (Tyche, property coverage) but is not an
  author of this paper. Fix before any submission.
- The softmax "scale invariance" metamorphic property as printed in Hatfield-Dodds (SciPy
  2020) is mathematically wrong; see the callout in §1.2.
