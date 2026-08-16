# PBT — Property-Based Testing for Compute Kernels

**Goal:** Everything needed to design, build, and *defend* a property-based testing layer for
compute kernels — the oracle strategies, the floating-point theory that makes them honest, the
GPU tooling they sit on, and the experimental methodology that makes the comparison publishable.

> ### Note on the acronym
> Throughout this repository **PBT means *property-based testing*** (Claessen & Hughes 2000) —
> generated inputs checked against general properties instead of hand-written examples.

## Core idea

A kernel test needs an **oracle**: something that says whether an execution was correct. This
folder is organized around the three strategies the project is comparing:

| Arm | Oracle | Barr et al.'s term | Costs |
|-----|--------|--------------------|-------|
| **Reference** | `output ≈ reference(input)` within tolerance | *pseudo-oracle* | cheap to write, expensive to run, needs a tolerance argument |
| **Declarative** | many weak properties / algebraic laws / metamorphic relations | *derived oracle* | expensive to write, cheap to run, many are tolerance-free |
| **Hybrid** | both | — | — |
| *(free extra)* | sanitizer-clean, race-free, no OOB | *implicit oracle* | ~zero cost, ~zero false positives |

Everything else here — shrinking, tolerance selection, mutation scoring — exists to make that
comparison honest.

## Recommended path

| Step | Section | Time | Outcome |
|------|---------|------|---------|
| 1 | [§1 Foundations](#1-foundations--canonical-sources) | ~2 hr | What a property *is*; the five kinds; the oracle taxonomy |
| 2 | [§2 Metamorphic testing](#2-metamorphic-testing) | ~2 hr | The declarative arm's actual literature |
| 3 | [§5 Numerical & floating point](#5-numerical--floating-point-testing) | ~3 hr | Why "≈" is the hard part; the LAPACK residual trick |
| 4 | [§8 Prior art: LLM kernels](#8-prior-art-pbt--llm-kernel-generation) | ~2 hr | **Read before writing any paper.** Novelty status |
| 5 | [§7 Evaluation methodology](#7-evaluation-methodology-for-testing-research) | ~2 hr | How the experiments will be judged |
| 6 | [§4 Implementations & tooling](#4-implementations--tooling) | ~1.5 hr | The Hypothesis-vs-custom decision |
| 7 | [§3 Shrinking](#3-shrinking) | ~1.5 hr | Only matters once you have failures |
| 8 | [§6 Compiler / GPU / systems](#6-compiler--gpu--systems-testing) | ~3 hr | Tier-2 telemetry; what's already been done |
| 9 | [§9 Antithesis](#9-antithesis-deterministic-simulation-and-recordreplay) | ~30 min | Where their model aligns with ours |

**If you have one hour:** Hughes, *How to Specify It!* (§1) and Hatfield-Dodds, *Falsify Your
Software* (§1). Those two contain most of the practical content.

**If you have three:** add Barr et al.'s oracle survey (§1), LAPACK Working Note 41 §7.1.1 (§5),
and Sarkar's *Correctness Illusion* (§8).

Then read [NOTES.md](./NOTES.md) — the synthesis, the worked kernel properties, and the
recommendations.

---

## 1. Foundations & canonical sources

**QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs** — Koen Claessen,
John Hughes. ICFP 2000, pp. 268–279.
[PDF](./papers/quickcheck-icfp2000.pdf) · <https://dl.acm.org/doi/10.1145/351240.351266>
The origin of PBT. Properties are written as ordinary functions returning booleans; the tool
generates random inputs and reports failures. Introduces user-definable generators, conditional
properties with `==>`, and the classification/coverage facilities for checking that your
generator actually produces interesting data.
*Relevance:* the ancestor of everything here, and the source of the type-based (separate
generator + separate shrinker) design that Hypothesis later rejected. Read for vocabulary,
not for technique.

**How to Specify It! A Guide to Writing Properties of Pure Functions** — John Hughes. TFP 2019
/ LNCS 12053:58–83, 2020.
[PDF](./papers/how-to-specify-it-hughes-tfp2019.pdf) ·
<https://research.chalmers.se/publication/517894/file/517894_Fulltext.pdf>
Presents **five generic approaches** to writing properties — validity/invariant properties,
postconditions, metamorphic properties (including preservation of equivalence), inductive
properties, and model-based properties — then plants **eight bugs** in a binary-search-tree
implementation and measures which kinds catch which. Model-based properties find every bug and
find them far faster (mean 5.8 tests to failure, vs 56 for metamorphic and 77 for
postconditions); metamorphic properties are individually weak but powerful in combination;
"metamorphic properties are essentially an axiomatization of the API… there is no guarantee
that this axiomatization is complete."
*Relevance:* **this is the single most important paper for this project.** It is a small,
careful version of the exact experiment we are running, with the opposite cost structure. His
conclusion — model-based properties are the best ROI *unless* the model is expensive or would
replicate the implementation's own bugs — is the hypothesis we test, and both of those escape
clauses apply to kernels. See NOTES §0 and §1.1.

**The Oracle Problem in Software Testing: A Survey** — Earl Barr, Mark Harman, Phil McMinn,
Muzammil Shahbaz, Shin Yoo. IEEE TSE 41(5):507–525, 2015.
[PDF](./papers/oracle-problem-survey-tse2015.pdf) ·
<https://doi.org/10.1109/TSE.2014.2372785>
Taxonomizes test oracles into **specified**, **derived** (including pseudo-oracles and
metamorphic relations), **implicit**, and **lack of an automated oracle**. Argues that oracle
automation is the bottleneck inhibiting broader test automation, since input generation has had
four decades of advances that "none of… address the issue of checking generated inputs."
*Relevance:* gives the project's three arms a standard vocabulary. Our reference oracle is a
*pseudo-oracle* (§5.1), our declarative properties are *derived* oracles, and our Tier-2
sanitizer checks are *implicit* oracles. Use this framing in the paper's related work.

**Falsify Your Software: Validating Scientific Code with Property-Based Testing** —
Zac Hatfield-Dodds. SciPy 2020 proceedings, pp. 162–167.
[PDF](./papers/validating-scientific-code-pbt-scipy2020.pdf)
Four property categories for scientific code — output bounds, round-trips, differential
testing, metamorphic properties — each demonstrated on a real NumPy or Astropy bug. **Uses
softmax as its running example**: shows the naive `exp(x)/exp(x).sum()` overflowing at
`np.exp([710.])`, which the property test finds "almost instantly." Also documents Astropy's
use of `hypothesis.target()` to steer generation toward larger errors, and the "threshold
problem" where a minimal failing example cannot distinguish a subtle bug from a serious one.
*Relevance:* the closest thing to a template for our Tier-1 properties, written by a Hypothesis
maintainer. ⚠️ **Its softmax "scale-invariance" property is mathematically wrong as printed**
(softmax is shift-invariant, not scale-invariant) — a useful, citable illustration of the
declarative arm's real hazard. See NOTES §1.2.

**Choosing Properties for Property-Based Testing** — Scott Wlaschin. Blog post, F# for Fun and
Profit, 2014. <https://fsharpforfunandprofit.com/posts/property-based-testing-2/>
Seven memorable property patterns: "different paths, same destination"; "there and back again"
(inverse); "some things never change" (invariant); "the more things change, the more they stay
the same" (idempotence); "solve a smaller problem first" (induction); "hard to prove, easy to
verify"; and "the test oracle."
*Relevance:* **blog-tier, and honest about it** — but it is the most-cited practitioner
taxonomy, Antithesis links it from their PBT page, and the names are genuinely good for
communicating with engineers. Use the names, cite Hughes and Barr for authority.

**Hypothesis: A New Approach to Property-Based Testing** — David MacIver, Zac Hatfield-Dodds
et al. JOSS 4(43):1891, 2019. [PDF](./papers/hypothesis-joss2019.pdf) ·
<https://doi.org/10.21105/joss.01891>
A two-page software paper. Establishes the citation and the framing of Hypothesis's departures
from QuickCheck (internal shrinking, the example database, no separate shrinker).
*Relevance:* the citable reference for Hypothesis. Not a substantive read.

**Etna: An Evaluation Platform for Property-Based Testing (Experience Report)** — Jessica Shi,
Alperen Keles, Harrison Goldstein, Benjamin Pierce, Leonidas Lampropoulos. PACMPL 7(ICFP)
Art. 218, 2023. [PDF](./papers/etna-icfp2023-lampropoulos.pdf) · arXiv:2603.27002
Motivated by the observation that "the PBT literature, though long on creativity, is short on
rigorous comparisons." Provides workloads across Rocq, Haskell, OCaml, Racket and Rust, using
**hand-authored mutants** (deliberately, so that ground truth is maintainable and every mutant
provably violates the property), tasks defined as (mutant, property) pairs, 10 trials, 60 s
timeouts, Mann–Whitney U, and a "task bucket" presentation instead of mean time-to-failure.
*Relevance:* **the experimental design to copy.** See NOTES §6.2.

**Property-Based Testing in Practice** — Harrison Goldstein, Joseph Cutler, Daniel Dickstein,
Benjamin Pierce, Andrew Head. ICSE 2024 (Distinguished Paper).
[PDF](./papers/pbt-in-practice-icse2024.pdf)
30 interviews with PBT users at Jane Street. Strengths: testing complex code and building
confidence beyond conventional tests; most uses fall into a small number of high-leverage
idioms. Weaknesses: the complexity of writing properties and generators, and **"the difficulty
of evaluating their effectiveness."**
*Relevance:* cite for motivation — practitioners cannot tell whether their properties are any
good, which is precisely what our experiment measures. Qualitative, single company; don't use
for quantitative claims.

**Do Judge a Test by its Cover: Combining Combinatorial and Property-Based Testing** —
Harrison Goldstein, John Hughes, Leonidas Lampropoulos, Benjamin Pierce. ESOP 2021, LNCS
12648:264–291 (open access). [PDF](./papers/do-judge-a-test-by-its-cover-esop2021.pdf)
Generalizes combinatorial *t*-way coverage to algebraic data types via sparse test descriptions,
then biases generators to cover feature combinations, finding bugs in fewer tests.
*Relevance:* a defensible **input-adequacy metric for a generator** that is not code coverage.
For us, the "features" are (dtype × layout × masked/unmasked × boundary shape × launch config)
— exactly the space NOTES §5.2 argues we should be sampling.

**Programmable Property-Based Testing** — Alperen Keles, Justine Frank, Ceren Mert, Harrison
Goldstein, Leonidas Lampropoulos. arXiv:2602.18545, 2026.
[PDF](./papers/2602.18545.pdf) · <https://arxiv.org/abs/2602.18545>
Argues PBT frameworks confine users to what the library authors anticipated, and proposes
"deferred binding abstract syntax" to turn properties into reifiable data structures
**decoupled from the mechanisms that execute them**, implemented in Rocq and Racket.
*Relevance:* the academic statement of our architectural instinct — separating the property
from its runner. If we build a custom driver, this is the paper that justifies it in principle.
Recent and unrefereed.

**Also present:** `targeted-pbt-issta2017.pdf` (Löscher & Sagonas, *Targeted Property-Based
Testing*, ISSTA 2017 — search-guided input generation, the ancestor of `hypothesis.target()`),
`automating-targeted-pbt-icst2018.pdf` (its automated follow-up),
`coverage-guided-pbt-fuzzchick-oopsla2019.pdf` (Lampropoulos, Hicks & Pierce, OOPSLA 2019),
`2508.14394.pdf` (*Tuning Random Generators: PBT as Probabilistic Programming*),
`2606.22616.pdf` (*Compositional Generator Equivalence*).

---

## 2. Metamorphic testing

The declarative arm's literature is substantially separate from PBT's and is where most of the
"no oracle available" thinking lives.

**Metamorphic Testing: A New Approach for Generating Next Test Cases** — T.Y. Chen, S.C. Cheung,
S.M. Yiu. HKUST tech report HKUST-CS98-01, 1998; posted to arXiv 2020.
[PDF](./papers/2002.12543.pdf) · <https://arxiv.org/abs/2002.12543>
The origin. Observes that *passing* test cases are normally discarded without further
exploitation, and proposes deriving new test cases from them using known properties of the
function — explicitly motivated by the absence of a complete oracle. Hughes' canonical example
from this line: you cannot easily check that a returned path is shortest, but you can check it
is no longer than the shortest path via a neighbour.
*Relevance:* the citation for the origin of "no oracle → check relations between executions."
Informal by modern standards; cite it, read the surveys.

**A Survey on Metamorphic Testing** — Sergio Segura, Gordon Fraser, Ana Sánchez, Antonio
Ruiz-Cortés. IEEE TSE 42(9):805–824, 2016.
[PDF](./papers/metamorphic-testing-survey-tse2016-segura.pdf) ·
<https://doi.org/10.1109/TSE.2016.2532875>
Systematic survey of ~119 papers, classified by MR identification, test-data generation,
execution, and application domain.
*Relevance:* the best structured taxonomy of *how MT is applied in practice*. Partly superseded
for coverage by the 2018 CSUR review, but better organized.

**Metamorphic Testing: A Review of Challenges and Opportunities** — T.Y. Chen, F-C. Kuo, H. Liu,
P-L. Poon, D. Towey, T.H. Tse, Z.Q. Zhou. ACM Computing Surveys 51(1) Art. 4, 2018.
<https://doi.org/10.1145/3143561> — paywalled, no local copy.
*Relevance:* the standard modern MT survey. ⚠️ **We could not retrieve this** (ACM and both
OA mirrors returned 403), so it is cited on metadata only. Retrieve manually before relying on
it. See NOTES §8.

**Metamorphic Testing: Testing the Untestable** — Sergio Segura, Dave Towey, Zhi Quan Zhou.
IEEE Software, 2020. [PDF](./papers/metamorphic-testing-untestable-ieee-software-2020-segura.pdf)
A short practitioner-facing overview.
*Relevance:* weak as a research contribution, but the right one-page citation when you need to
motivate MT to a non-specialist reader in two sentences.

**A Miss Is as Good as A Mile: Metamorphic Testing for Deep Learning Operators** — Jinyin Chen,
Chengyu Jia, Yunjie Yan, Jie Ge, Haibin Zheng, Yao Cheng. PACMSE 1(FSE) Art. 89, 2024.
<https://doi.org/10.1145/3660796> — gold OA but no local copy (see below).
**21 metamorphic relations for 10 widely-used DL operators**, explicitly targeting **precision
errors as well as implementation errors**, using the MRs to guide input generation and to trace
a precision error back to the responsible input. 32 bugs found across 9 versions of 5 DL
libraries, 14 previously unknown.
*Relevance:* **probably the most directly relevant paper in the entire metamorphic literature to
this project** — it is an operator-level MR catalogue with a precision-error focus, i.e. a
ready-made starting point for our declarative arm. ⚠️ It is gold open access but ACM
bot-blocked every automated fetch. **Download it manually; this is the highest-priority missing
item in this library.**

**EAGLE: Creating Equivalent Graphs to Test Deep Learning Libraries** — Jiannan Wang, Thibaud
Lutellier, Shangshu Qian, Hung Viet Pham, Lin Tan. ICSE 2022.
[PDF](./papers/eagle-icse2022.pdf) · <https://jiannanwang.github.io/files/eagle-icse22.pdf>
**17 hand-written equivalence rules** (different APIs, dtypes, optimizations) producing
equivalent computation graphs, cross-checked **within a single library** — no second backend
required. 20 bugs in TensorFlow and PyTorch, 9 previously unknown. Motivated explicitly by
CRADLE's need for two independent implementations.
*Relevance:* architecturally the closest analogue to our declarative arm, and **the answer for
NKI**, where there is no second Trainium implementation to differentially test against. Honest
read: it found fewer bugs than mining-based fuzzers, but they were silent wrong-answer bugs —
the expensive kind.

**Automatic Discovery and Cleansing of Numerical Metamorphic Relations (AutoMR)** — Bo Zhang,
Hongyu Zhang, Junjie Chen, Dan Hao, Pablo Moscato. ICSME 2019.
[PDF](./papers/automr-numerical-metamorphic-relations.pdf) · <http://hongyujohn.github.io/AutoMR.pdf>
Searches for *polynomial* MRs — input sub-relations restricted to linear expressions over
source/follow-up inputs, output sub-relations linear or quadratic — via particle swarm
optimization over multiple dynamic executions, then removes redundant MRs using constraint
solving and SVD.
*Relevance:* the most relevant MR-inference work for numerical kernels; the polynomial MR class
is a good fit for linear algebra (scaling, additivity, permutation). A plausible way to
*discover* properties rather than hand-writing them.

**GenMorph: Automatically Generating Metamorphic Relations via Genetic Programming** —
Jon Ayerdi, Valerio Terragni, Gunel Jahangirova, Aitor Arrieta, Paolo Tonella. arXiv:2312.15302
(TSE-format manuscript). [PDF](./papers/2312.15302.pdf)
Genetic-programming search for MRs over boolean, numerical and ordered-sequence data in Java
methods; generates MRs for 18 of 23 test methods and outperforms Randoop, EvoSuite and AutoMR
on fault detection.
*Relevance:* benchmarks directly against AutoMR, so cite the pair together. Java/unit-test
domain, so the transfer to tensors is not automatic.

**MR-Scout: Automated Synthesis of Metamorphic Relations from Existing Test Cases** — Congying
Xu, Valerio Terragni, Hengcheng Zhu, Jiarong Wu, Shing-Chi Cheung. ACM TOSEM 33(6) Art. 150,
2024. [PDF](./papers/2304.07548.pdf) · <https://arxiv.org/abs/2304.07548>
Mines MR-encoded test cases from 701 open-source projects, synthesizing 11,000+ MR instances;
>97% judged high-quality, +13.52% line coverage and +9.42% mutation score when used for test
generation.
*Relevance:* a different axis from AutoMR/GenMorph — it *extracts* MRs from existing tests
rather than searching numerically. Relevant if we mine properties from existing kernel test
suites (PyTorch's, Triton's).

**Predicting Metamorphic Relations for Testing Scientific Software: A Machine Learning Approach
Using Graph Kernels** — Upulee Kanewala, James Bieman, Asa Ben-Hur. STVR 26(3):245–269, 2016.
[Preprint PDF](./papers/kanewala-predicting-mrs-graph-kernels-stvr2016-preprint.pdf)
Builds a CFG/PDG representation of a function, extracts node/path features plus random-walk and
graphlet graph kernels, and trains a classifier to predict which of a fixed MR catalogue
(permutative, additive, multiplicative, inclusive, exclusive) hold.
*Relevance:* the closest existing work to "automatically decide which weak properties apply to
this kernel." Companion: **Predicting Metamorphic Relations for Matrix Calculation Programs**
(Rahman & Kanewala, MET 2018, [PDF](./papers/predicting-mrs-matrix-calculation-programs-met2018.pdf))
— the single most on-target prior work for MRs on linear algebra, though only a short workshop
paper.

**Metamorphic Testing for (Graphics) Compilers** — Alastair Donaldson, Andrei Lascu. MET 2016
(short paper). [PDF](./papers/metamorphic-testing-graphics-compilers-met2016-donaldson.pdf)
Four pages. Introduces metamorphic compiler testing via **opaque value injection** — inject
values the compiler cannot constant-fold, then require output invariance — applied to OpenGL
shading language compilers. Explicitly frames EMI (§6) as an instance of metamorphic testing.
*Relevance:* tiny but conceptually strong; the opaque-injection technique transfers directly to
CUDA/Triton kernels and is a cheap Tier-1 property we should implement.

**Kaizen: Metamorphic Fuzzing and Differential Testing for LLM-Translated HPC Applications** —
Ludwig, Anklesaria, Jin, Pophale, et al. arXiv:2607.04058, 2026.
[PDF](./papers/2607.04058.pdf)
Tests LLM translations *from CUDA* to OpenMP/OpenACC/Kokkos/SYCL using metamorphic source
mutation plus grammar-based input fuzzing plus differential testing, across 16 scientific
applications. Key finding: **compilation success is a poor proxy for correctness**, and semantic
errors are frequently input-dependent.
*Relevance:* the closest published work to the project's *translation* workstream. Recent and
unrefereed — treat conclusions as preliminary.

**Also present:** `1804.01954.pdf` (Kanewala & Bieman, *Testing Scientific Software: A Systematic
Literature Review*, IST 2014 — the "why numerical code has an oracle problem" citation),
`2406.05397.pdf` (MR generation, state of the art — short vision paper; the substantive version
is ACM TOSEM 34(5), 2025, paywalled), `2605.13898.pdf` (MT × LLMs systematic survey, 93 studies,
2026), `2507.22610.pdf` (MT of deep *code models* — likely off-target), `cradle-icse2019.pdf`
(CRADLE, ICSE 2019 — cross-backend differential + anomaly localization; superseded by EAGLE),
`2204.08734.pdf` (Muffin, ICSE 2022 — differential testing during *training*, i.e. gradient
correctness).

---

## 3. Shrinking

**Simplifying and Isolating Failure-Inducing Input** — Andreas Zeller, Ralf Hildebrandt.
IEEE TSE 28(2):183–200, 2002. [PDF](./papers/delta-debugging-zeller-tse2002.pdf)
The `ddmin` delta-debugging algorithm: greedy divide-and-conquer minimization over a sequence of
changes, with complexity bounds and an explicit treatment of the *unresolved* outcome, where a
reduced input is neither passing nor failing but invalid.
*Relevance:* the ancestor of every shrinker. The "unresolved" case is exactly the validity
problem that plagues array shrinking — shrink a tensor's shape and you may break a kernel
precondition rather than reproduce the bug.

**Test-Case Reduction via Test-Case Generation: Insights from the Hypothesis Reducer** —
David MacIver, Alastair Donaldson. ECOOP 2020, LIPIcs 166:13:1–13:27 (open access).
[PDF](./papers/hypothesis-reducer-ecoop2020.pdf) ·
<https://doi.org/10.4230/LIPIcs.ECOOP.2020.13>
Describes **internal reduction**: rather than reducing the generated value, reduce the sequence
of *random choices* consumed during generation, and re-generate. Cast as a shortlex optimization
over choice sequences. The claimed advantages are explicitly **not** speed or minimality —
"reduction quality or performance are not included among the major advantages" — but that every
generator gets reduction for free, and that reduced cases are always generatable, which
sidesteps the validity problem. Their Csmith evaluation shows the flip side: Hypothesis cannot
reduce below ~410 bytes because that is the smallest program Csmith can emit, whereas C-Reduce
reaches a 14-byte `main`; Hypothesis "is reducing against a harder validity oracle."
*Relevance:* **the decisive paper for the Hypothesis-vs-custom decision.** Internal reduction
re-runs the property, which for us means re-executing on hardware — so it cannot operate over a
recorded batch. And its floor is set by the *generator*, not the shrinker: if our generator
never emits a 1×1 tensor, no internal shrinker will ever hand us one. See NOTES §2.1 and §3.1.

**Evaluating Shrinking (Experience Report)** — Alperen Keles, George Miao, Leonidas
Lampropoulos. arXiv:2608.09935, 2026. [PDF](./papers/2608.09935.pdf) ·
<https://arxiv.org/abs/2608.09935>
Measures QuickCheck vs Hedgehog vs Falsify on ETNA workloads, using **tree edit distance to an
exhaustively-computed optimum** as an effectiveness metric plus shrink time as a cost metric.
Conclusion: "QuickCheck's structural shrinking is usually faster and remains competitive on
final counterexample quality; integrated shrinking does not by itself guarantee a performance or
effectiveness advantage."
*Relevance:* the only head-to-head measurement I found, and it deflates the common claim that
integrated/internal shrinking is simply better. Recent and unrefereed — but it means "we didn't
use Hypothesis's shrinker" is a defensible position, not a deficiency.

**Integrated versus Manual Shrinking** — Edsko de Vries. Well-Typed blog, 2019.
<https://well-typed.com/blog/2019/05/integrated-shrinking/>
The clearest explanation of why integrated shrinking (Hedgehog, jqwik) exists: type-based
shrinkers do not know the generator's implicit validity constraints, so they produce minimized
*invalid* counterexamples.
*Relevance:* blog-tier but the best available explanation of the distinction. jqwik's docs make
the same argument (<https://jqwik.net/property-based-testing.html>).

**Test-Case Reduction and Deduplication Almost for Free with Transformation-Based Compiler
Testing (spirv-fuzz)** — Alastair Donaldson, Paul Thomson, Vasyl Teliman, Stefano Milizia,
André Perez Maselco, Antoni Karpiński. PLDI 2021.
[PDF](./papers/spirv-fuzz-pldi2021.pdf)
If transformations are **small and independent**, plain delta debugging over the transformation
*sequence* shrinks a bug-inducing test for free — and the minimized sequence also heuristically
deduplicates bug reports.
*Relevance:* **the most underrated paper here for our engineering design.** "Design your
metamorphic transformations so the shrinker is trivial" is an architectural decision that must
be made up front, not retrofitted. It also directly supports shrinking at the *case-spec* level
rather than the tensor level. See NOTES §3.3 and §5.3.

**Swarm Testing** — Alex Groce, Chaoqiang Zhang, Eric Eide, Yang Chen, John Regehr. ISSTA 2012.
[PDF](./papers/swarm-testing-issta2012.pdf) · <https://agroce.github.io/issta12.pdf>
Randomly *omit features* per test run rather than enabling everything uniformly. Better coverage
for a fixed budget, and less need to hand-tune the test configuration.
*Relevance:* not shrinking, but generation — and nearly free to add to a batch generator. For
us, "features" are masking, layouts, dtypes, boundary shapes, launch configurations. Directly
addresses the "our corpus is homogeneous" threat.

---

## 4. Implementations & tooling

Assessments here are from official documentation, read directly. The decision writeup is
NOTES §2.

**Hypothesis (Python)** — <https://hypothesis.readthedocs.io/>
The relevant surfaces:
- **`hypothesis.extra.numpy`**: `arrays(dtype, shape, *, elements, fill, unique)`,
  `array_shapes()`, `from_dtype()` (with `allow_nan`, `allow_infinity`, `allow_subnormal`,
  `min_magnitude`, `max_magnitude`), `broadcastable_shapes()`,
  `mutually_broadcastable_shapes()`, `basic_indices()`, `integer_array_indices()`.
- **`hypothesis.extra.array_api`**: `make_strategies_namespace(xp, *, api_version=None)`
  returns a namespace of the same strategies against **any Array-API-conforming library** —
  NumPy, PyTorch, CuPy, JAX, Dask. See the Quansight Labs write-up:
  <https://labs.quansight.org/blog/2021/10/hypothesis-array-api>.
- **Reproducibility**: `@seed`, `derandomize`, `@reproduce_failure`, the `ExampleDatabase` /
  `DirectoryBasedExampleDatabase` (`.hypothesis/examples`), and the `Phase` enum
  (`explicit, reuse, generate, target, shrink, explain`).
- **Observability**: `HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY` writes JSONL per-test-case
  observations to `.hypothesis/observed/`, with fields `type, status, status_reason,
  representation, arguments, how_generated, features, coverage, timing, metadata` — and
  optional **`choice_nodes`** (the low-level choice sequence) under `OBSERVABILITY_CHOICES`.
- `hypothesis.target()` (targeted PBT), `hypothesis.stateful.RuleBasedStateMachine`,
  `hypothesis.extra.ghostwriter` (which can emit `idempotent()`, `roundtrip()`, `equivalent()`,
  `binary_operation()` and `ufunc()` property skeletons automatically).

*Relevance and honest assessment:* **Hypothesis works well as a generator library and badly as
our driver.** The documented facts that decide it: (i) `SearchStrategy.example()` is "designed
for use in a REPL", warns outside interactive use, and the docs direct you to `@composite` or
`data()` instead — so the only supported batch-harvest route is `@given` with
`phases=[Phase.generate]`; (ii) the compatibility page states documented APIs "will not break
except between major version bumps" while "undocumented attributes, modules, and behavior" may
break **in patch releases**, and `hypothesis.internal.conjecture` is undocumented; (iii)
`derandomize` produces the same cases only "until you update Hypothesis, Python, or the test
function", and `@reproduce_failure` makes **no cross-version guarantees and errors** on a
different version. So a Hypothesis corpus is reproducible-by-artifact, not
reproducible-by-seed. The observability JSONL output is, notably, *already* half of a
record/replay design and worth studying as a schema reference.

**QuickCheck (Haskell)** — <https://hackage.haskell.org/package/QuickCheck>. Type-based
generators and separate `shrink` functions. *Relevance:* the baseline the shrinking literature
compares against; per §3, still competitive.

**PropEr (Erlang)** — <https://proper-testing.github.io/>. Home of targeted PBT
(`targeted-pbt-issta2017.pdf`) and of Quviq QuickCheck's stateful-testing lineage.
*Relevance:* the targeted-search work is the transferable part; the BEAM specifics are not.

**fast-check (JavaScript/TypeScript)** — <https://fast-check.dev/>. **jqwik (Java)** —
<https://jqwik.net/>, a strong proponent of integrated shrinking. **proptest (Rust)** —
<https://proptest-rs.github.io/proptest/>, which uses a value-tree/integrated model closer to
Hypothesis than to QuickCheck. *Relevance:* useful for surveying design choices in the paper's
related work; none of them beat Hypothesis for array support.

**Array/tensor-specific tooling:** the honest finding is that **no PBT library is designed for
tensors.** `hypothesis.extra.numpy` and `hypothesis.extra.array_api` are the state of the art,
and neither has a notion of case groups, launch configurations, or numerical tolerance. That
gap is the project's tooling contribution.

---

## 5. Numerical & floating-point testing

This is where the reference oracle either becomes rigorous or becomes theatre. Full synthesis
in NOTES §4.

**What Every Computer Scientist Should Know About Floating-Point Arithmetic** — David Goldberg.
ACM Computing Surveys 23(1):5–48, 1991.
[PDF](./papers/goldberg-1991-what-every-cs-should-know-fp.pdf) ·
<https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>
Distinguishes **benign cancellation** (subtracting exact quantities; with a guard digit,
relative error < 2ε by his Theorem 2) from **catastrophic cancellation** (operands already carry
rounding error, so subtraction removes the accurate leading digits). Theorem 8 gives the Kahan
summation bound. Explicitly warns that an optimizer applying real-number algebra will fold
`C = (T−S)−Y` to zero and destroy compensated summation.
*Relevance:* the tutorial entry point, and that last warning is exactly the compiler hazard our
Triton/CUDA/NKI layer faces. For theorems, cite Higham.

**LAPACK Working Note 41: Installation Guide for LAPACK** — Anderson, Dongarra, Ostrouchov et al.
[PDF](./papers/lapack-lawn41-installation-testing-guide.pdf) ·
<https://www.netlib.org/lapack/lawnspdf/lawn41.pdf>
§7.1.1 and Tables 2/4 document the **test ratio** convention: rather than comparing outputs
elementwise, LAPACK computes normalized residuals — `‖LU − A‖/(n‖A‖ε)`, `‖b − Ax‖/(‖A‖‖x‖ε)`,
`‖I − AA⁻¹‖/(n‖A‖‖A⁻¹‖ε)` — and the sample input file carries `30.0   Threshold value of test
ratio`.
*Relevance:* **the strongest declarative-spec-oracle precedent that exists**, and the single
most actionable item in this section. Dividing the residual by everything that legitimately
scales it — norms, problem size, machine epsilon — yields one dimensionless threshold (30) valid
across every routine, size and precision. This is what our reference arm should do instead of
`allclose` with a magic `rtol`. See NOTES §4.2.

**A New Approach to Probabilistic Rounding Error Analysis** — Nicholas Higham, Théo Mary.
SIAM J. Sci. Comput. 41(5):A2815–A2835, 2019.
[PDF](./papers/higham-mary-probabilistic-rounding-error-analysis-simax2019.pdf)
Replaces the classical worst-case constant `γₙ = nu/(1−nu)` with a relaxed constant proportional
to **`√(n log n)·u`** holding with probability bounded below independently of `n`. The bounds are
backward-error bounds, exact rather than first-order, and valid for any finite `n` — unlike
CLT-based arguments.
*Relevance:* the rigorous justification for the widely-used √n rule of thumb, and the number to
use for a practically tight reassociation tolerance. For bf16 with long reductions the classical
`n·u` bound is *vacuous*; see the table in NOTES §4.3.

**Sharper Probabilistic Backward Error Analysis for Basic Linear Algebra Kernels with Random
Data** — Nicholas Higham, Théo Mary. SIAM J. Sci. Comput. 42(5):A3427–A3446, 2020.
[PDF](./papers/higham-mary-sharper-probabilistic-backward-error-simax2020.pdf)
For data with **zero or small mean**, the bound sharpens from `O(√n·u)` to **`O(u)`, independent
of n** — proved for summation and extended to inner products and matrix products.
*Relevance:* directly consequential. Our generators typically draw zero-mean tensors, which puts
us in the n-independent regime — but softmax's `exp(·)` numerators are all-positive, which does
not. **The correct tolerance depends on the generator's distribution, not just the dtype.**

**Numerical Behavior of NVIDIA Tensor Cores** — Massimiliano Fasi, Nicholas Higham, Mantas
Mikaitis, Srikara Pranesh. PeerJ Computer Science 7:e330, 2021 (CC-BY).
[PDF](./papers/fasi-higham-mikaitis-pranesh-numerical-behavior-nvidia-tensor-cores-2020.pdf)
NVIDIA's PTX docs state only that accumulation is "at least single precision" and that the
accumulation order, rounding, and subnormal handling are **unspecified**. By black-box
experiment on V100/T4 the authors determine: subnormals are natively supported; products are
held exactly and not rounded back to binary16 before accumulation; each output element incurs at
most four rounding errors; partial products are accumulated **starting from the largest
magnitude**; **the additions use round-toward-zero, not round-to-nearest**; and the internal
accumulator carries at least 2 extra significand bits. Public test suite:
<https://github.com/north-numerical-computing/tensor-cores-numerical-behavior>.
*Relevance:* **operationally the most important paper in this section.** Round-toward-zero is a
*systematic* bias, so tensor-core errors accumulate roughly linearly rather than cancelling as
√n — any tolerance derived under a round-to-nearest assumption is too tight. And since **no
comparable characterization of AWS Trainium exists**, running this methodology against Trainium
is a prerequisite for the NKI arm. See NOTES §4.4.

**Mixed Precision Block Fused Multiply-Add: Error Analysis and Application to GPU Tensor Cores**
— Blanchard, Higham, Lopez, Mary, Pranesh. SIAM J. Sci. Comput. 42(3):C124–C141, 2020.
[PDF](./papers/blanchard-higham-mixed-precision-block-fma-tensor-cores-sisc2020.pdf)
A block-FMA error model generalizing scalar FMA to matrix-argument tensor units, applied to
Volta/Turing, comparing TC16 vs TC32 accumulation.
*Relevance:* the model our GEMM tolerance formula should be derived from for tensor-core paths —
and the right thing to run the reference through, instead of an idealized fp64 dot product.

**FP8 Formats for Deep Learning** — Micikevicius, Stosic, Burgess, Cornea, Dubey, et al.
arXiv:2209.05433, 2022. [PDF](./papers/2209.05433.pdf) · <https://arxiv.org/abs/2209.05433>
Proposes **E4M3** and **E5M2**; E5M2 follows IEEE 754 conventions for special values while E4M3
extends dynamic range by not representing infinities and using one mantissa pattern for NaN.
Two facts that constrain testing: FP8 matmul instructions **produce higher-precision (fp32)
outputs**, so an fp8 GEMM's accumulator is fp32; and **per-tensor scaling factors are handled in
software**, chosen so the max magnitude lands near the format max, with overflow saturating.
*Relevance:* any fp8 property test must parameterize over the scale factor and probe the
saturation boundary. Companion: [OCP 8-bit FP spec](./papers/ocp-8bit-floating-point-specification-ofp8-r1.0.pdf)
— E4M3 bias 7, max 448; E5M2 bias 15, max 57,344. ⚠️ Rev 1.0's §4.2 contradicts its own
normative Table 1 on the biases; trust Table 1.

**Algorithms for Efficient Reproducible Floating Point Summation** — Ahrens, Demmel, Nguyen.
ACM TOMS 46(3) Art. 22, 2020.
[PDF](./papers/demmel-ahrens-nguyen-reproducible-summation-toms2020.pdf) · ReproBLAS:
<https://bebop.cs.berkeley.edu/reproblas/>
Defines reproducibility as bitwise-identical results across runs with different hardware
resources, and achieves it with a "binned number" accumulator in one read-only pass and one
parallel reduction — at a cost of **≈9n floating-point operations plus ≈3n bitwise operations**
per n-word sum.
*Relevance:* the citation for "bitwise reproducibility is achievable but nobody will pay for
it." Pair with Intel's oneMKL Conditional Numerical Reproducibility docs
(<https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-0/obtaining-numerically-reproducible-results.html>),
which grant reproducibility only under a fixed thread count and, for Strict CNR, only for
`?gemm`/`?symm`/`?hemm`/`?trsm`. Together they justify a tolerance-based oracle to any
stakeholder in two sentences.

**FLiT: Cross-Platform Floating-Point Result-Consistency Tester and Workload** — Sawaya,
Bentley, Briggs, Gopalakrishnan, Ahn. IISWC 2017.
[PDF](./papers/flit-iiswc2017-cross-platform-fp-consistency.pdf)
Compiles a kernel collection across compilers, flags and platforms and collects results into a
database. Motivating incidents: a compiler-introduced FMA made a Community Earth System Model
run unreliable; architectural heterogeneity caused an MPI inconsistency that took a week to
root-cause. The authors argue explicitly that **acceptable variability is application-defined
and a tool must not hard-code it** — they report variability and leave interpretation to the
domain scientist.
*Relevance:* philosophically the most important paper here for our design. It argues for a third
oracle tier that **measures and records cross-backend divergence without asserting on it.**

**Varity: Quantifying Floating-Point Variations in HPC Systems Through Randomized Testing** —
Ignacio Laguna. IPDPS 2020. [PDF](./papers/varity-ipdps2020-laguna.pdf)
Generates random FP programs for host and device, compiles with every compiler on the system,
and differentially tests across (compiler, architecture) pairs. 50,000 experiments on POWER9 +
V100 found programs producing significantly different results for the same input.
*Relevance:* the closest existing analogue to our cross-backend problem. Companion:
**Testing GPU Numerics: Finding Numerical Differences Between NVIDIA and AMD GPUs**
([PDF](./papers/2410.09172.pdf), arXiv:2410.09172) — >600,000 tests, differences traced to
**math library calls**, fp64-vs-fp32 handling, and HIPIFY translation itself. Math-library
divergence matters enormously for softmax (`exp`) and layernorm (`rsqrt`).

**Automatically Improving Accuracy for Floating Point Expressions (Herbie)** — Panchekha,
Sanchez-Stern, Wilcox, Tatlock. PLDI 2015 (Distinguished Paper).
[PDF](./papers/herbie-pldi2015-panchekha.pdf)
Estimates and **localizes** rounding error by sampling points rather than by static analysis,
then applies rewrite rules, series expansions, and **regime inference** — splitting the input
domain and using different rewrites per region.
*Relevance:* its sampled-error localization is the technique for deciding which part of a
softmax/layernorm kernel caused a tolerance failure. Its regime idea argues our tolerances
should be input-regime-dependent, not global constants.

**Stochastic Rounding: Implementation, Error Analysis, and Applications** — Croci, Fasi, Higham,
Mary, Mikaitis. Royal Society Open Science 9:211631, 2022 (CC-BY).
[PDF](./papers/croci-fasi-higham-mikaitis-stochastic-rounding-rsos2022.pdf)
Survey tying stochastic rounding to error analysis. Clarifies the tool taxonomy: CADNA uses
stochastic arithmetic (CESTAC); Verrou and Verificarlo use **Monte Carlo Arithmetic**, which is
strictly more general because it perturbs operation inputs and outputs as well as rounding
results.
*Relevance:* MCA gives a principled way to **derive** a tolerance — run the fp64 reference N
times under randomized rounding and take the spread. That spread is the number of digits the
computation actually determines. See NOTES §4.5. Companion: `1509.01347.pdf` (Verificarlo,
ARITH 2016 — an LLVM extension, so it captures the effect of compiler optimizations on accuracy).

**A Comprehensive Study of Real-World Numerical Bug Characteristics** — Di Franco, Guo,
Rubio-González. ASE 2017. [PDF](./papers/numerical-bug-characteristics-ase2017.pdf)
269 numerical bugs from NumPy, SciPy, LAPACK, GSL, Elemental; taxonomy of accuracy /
special-value / convergence / correctness bugs. Correctness (~37%) is the largest class;
76/269 involve special values.
*Relevance:* our prior distribution over fault classes, and the closest thing to a real-fault
corpus for numerical code. It is a bug *list*, not a runnable benchmark.

**Also present:** `muller-on-the-definition-of-ulp-inria-rr5504.pdf` (Muller, *On the definition
of ulp(x)* — read this before making ULP a first-class assertion unit; the competing definitions
disagree near binade boundaries), `precimonious-sc2013-rubio-gonzalez.pdf` (SC'13 precision
tuning), `rubio-gonzalez-icse2020-error-inducing-fp-inputs.pdf` (symbolic execution to find
error-inducing FP inputs — the open, citable successor to S3FP),
`fpbench-nsv2016-damouche.pdf` (FPBench/FPCore — the precedent for making the error measure an
explicit declared part of a test), `demmel-nguyen-fast-reproducible-fp-summation-arith2013.pdf`,
`cadna-round-off-error-propagation-jezequel-chesneaux.pdf`, `1811.05618.pdf` (bisecting which
file causes compiler-induced variability).

**Tolerance defaults, for reference** (verified from official docs/source, NOTES §4.1):
`torch.testing.assert_close` uses `|a−e| ≤ atol + rtol·|e|` with rtol/atol of 1.3e-6/1e-5
(fp32), 1e-3/1e-5 (fp16), **1.6e-2**/1e-5 (bf16), 1e-7/1e-7 (fp64); `equal_nan=False`.
JAX's `_default_tolerance` uses one value as both atol and rtol (1e-6 fp32, 1e-3 fp16, 1e-2
bf16, 1e-1 fp8) and then **multiplies both by the array element count**. CUTLASS uses a
*symmetric* relative test `|a−b| < ε(|a|+|b|)` with a nonzero floor, production defaults
ε = 0.05 and floor = 1/256 — while its device GEMM *unit tests* demand **bit-exactness**,
which works only because they use small integer-valued inputs.

---

## 6. Compiler / GPU / systems testing

Where the Tier-2 properties live, and where much of the "generate programs, check relations"
technique was invented.

**Finding and Understanding Bugs in C Compilers (Csmith)** — Xuejun Yang, Yang Chen, Eric Eide,
John Regehr. PLDI 2011. [PDF](./papers/csmith-pldi2011.pdf) ·
<https://users.cs.utah.edu/~regehr/papers/pldi11-preprint.pdf>
Random C program generation whose defining property is generating programs that cover a large
subset of C **while avoiding undefined and unspecified behavior** — which is what makes an
automatic wrong-code oracle possible at all. 325+ previously unknown bugs over three years.
*Relevance:* the transferable lesson is that the hard engineering is not "generate random
programs" but "generate random inputs with a decidable oracle." For us that means staying inside
well-defined numerics — no NaN/Inf injection in the default corpus (NOTES §5.2).

**Compiler Validation via Equivalence Modulo Inputs (EMI/Orion)** — Vu Le, Mehrdad Afshari,
Zhendong Su. PLDI 2014 (Distinguished Paper). [PDF](./papers/emi-orion-pldi2014.pdf)
Profile a program on some inputs, then stochastically prune *unexecuted* code to derive variants
that must behave identically **on those inputs**. Sidesteps having to generate valid programs
from scratch.
*Relevance:* **conceptually one of the most important papers for the project.** EMI is a
metamorphic relation, and its "equivalent only with respect to these inputs" framing is exactly
the right weakening for kernels, where full equivalence is undecidable but input-relative
equivalence is testable. Follow-ups: [Athena](./papers/athena-oopsla2015.pdf) (OOPSLA 2015 —
adds insertions, MCMC-guided) and [Hermes](./papers/hermes-live-code-mutation-oopsla2016.pdf)
(OOPSLA 2016 — mutates live code too). Read the three in order.

**A Survey of Compiler Testing** — Junjie Chen, Jibesh Patra, Michael Pradel, Yingfei Xiong,
Dan Hao, Lu Zhang, Hongyu Zhang. ACM Computing Surveys 53(1), 2020.
[PDF](./papers/compiler-testing-survey-csur2020.pdf) ·
<https://www.software-lab.org/publications/csur2019_compiler_testing.pdf>
Organizes the field along four axes: test program construction, **test oracles**, test execution
efficiency, and actionability.
*Relevance:* the right survey, and its oracle taxonomy is a good frame for positioning our
properties. ⚠️ Do not confuse with `1810.02718.pdf` (Tang et al., *Compiler Testing: A Systematic
Literature Analysis*), which is bibliometric, not technical.

**Many-Core Compiler Fuzzing (CLsmith)** — Christopher Lidbury, Andrei Lascu, Nathan Chong,
Alastair Donaldson. PLDI 2015.
[PDF](./papers/clsmith-many-core-compiler-fuzzing-pldi2015.pdf)
Random generation of **deterministic, communicating OpenCL kernels**, plus an injection
mechanism enabling EMI on kernels that otherwise have little dynamically-dead code. Campaign
over **21 (device, compiler) configurations** spanning CPU, GPU, accelerator, FPGA and emulator.
*Relevance:* the direct ancestor. "Make kernels deterministic so you can differentially test
them" is the same problem we face with Triton and NKI. Companion:
[CUDAsmith](./papers/cudasmith-compsac2020.pdf) (COMPSAC 2020) ports the recipe to CUDA with
differential testing plus EMI against NVCC and Clang-CUDA.

**Automated Testing of Graphics Shader Compilers (GraphicsFuzz)** — Donaldson, Evrard, Lascu,
Thomson. PACMPL 1(OOPSLA) Art. 93, 2017. [PDF](./papers/graphicsfuzz-oopsla2017.pdf)
Pure metamorphic testing against the no-oracle problem: apply semantics-preserving
transformations to high-value real shaders and treat rendering mismatches as bugs. Defects in
every GPU/driver configuration tested, 60+ issues reported.
*Relevance:* the best demonstration that metamorphic testing beats the oracle problem in a domain
where outputs are underspecified — a close analogy to floating-point kernel outputs. The
production experience reports ([ECOOP 2020](./papers/graphicsfuzz-production-ecoop2020.pdf),
[ICST 2023](./papers/graphicsfuzz-industrial-deployment-icst2023.pdf)) cover flake handling and
triage cost, which academic papers usually omit and which will dominate our effort.

**GPUVerify: A Verifier for GPU Kernels** — Betts, Chong, Donaldson, Qadeer, Thomson. OOPSLA
2012. [PDF](./papers/gpuverify-oopsla2012.pdf)
Static verification of race- and divergence-freedom for CUDA/OpenCL via *synchronous, delayed
visibility* semantics, reducing kernel verification to sequential analysis of two threads.
*Relevance:* sound by construction, so it finds races that never manifest at runtime — at the
cost of false positives and required loop invariants. The complement to our dynamic Tier-2.

**NNSmith: Generating Diverse and Valid Test Cases for Deep Learning Compilers** — Jiawei Liu,
Jinkun Lin, Fabian Ruffy, Cheng Tan, Jinyang Li, Aurojit Panda, Lingming Zhang. ASPLOS 2023.
[PDF](./papers/2207.13066.pdf) · <https://arxiv.org/abs/2207.13066>
Lightweight per-operator shape/dtype specifications solved with SMT to generate valid DNN
graphs, **gradient-based search over inputs to avoid floating-point exceptional values**, and
differential testing. 72 new bugs across TVM, TensorRT, ONNXRuntime and PyTorch.
*Relevance:* the exceptional-value problem it solves is exactly what will plague our
tolerance-based oracle, and the gradient-guided input search is directly adoptable.

**DeepREL: Fuzzing Deep-Learning Libraries via Automated Relational API Inference** — Yinlin
Deng, Chenyuan Yang, Anjiang Wei, Lingming Zhang. ESEC/FSE 2022.
[PDF](./papers/deeprel-fse2022.pdf)
Infers API relations, then applies two oracles: **value equivalence** and **status equivalence**
(both must succeed or both must fail). 162 bugs, 106 new.
*Relevance:* **status equivalence is the underrated idea** — "kernel and reference must both
succeed or both raise" catches shape/bounds/dtype bugs with **no numerical tolerance at all**.
Free to implement, belongs in our catalogue.

**Chasing Elusive Memory Bugs in GPU Programs** — Ghosh, Nayak, Thallikar Shyam, Basu.
arXiv:2601.21552, 2026. [PDF](./papers/2601.21552.pdf)
Documents **input-dependent out-of-bounds accesses that only manifest under specific inputs and
therefore elude all existing runtime tools, including compute-sanitizer**, plus intra-allocation
OOBs that allocation-granularity checkers structurally cannot see.
*Relevance:* **the strongest peer-reviewable argument for the Tier-2 half of this project.**
Sanitizers are input-triggered detectors with no input-generation strategy; a property-based
generator layer on top of compute-sanitizer is exactly the complement.

**Characterizing Real-World Bugs in Tile Programs for Automated Bug Detection** — Rathnasuriya,
Song, Majoju, Moharir, Li, Yang, Xie. ISSTA 2026. [PDF](./papers/2605.19652.pdf) ·
arXiv:2605.19652
401 GitHub bug reports, **301 code-generation bugs in tile-based GPU frameworks** (predominantly
Triton, some TileLang). Bugs are "tightly coupled to input shapes, data types, and backend
targets" and often surface as **silent wrong results**.
*Relevance:* the strongest *peer-reviewed* empirical grounding for our generator design. Its
shape × dtype × backend axis is precisely the space our corpus should sample. Companion:
[Simulee](./papers/1905.01833.pdf) (ICSE 2020 — 319 CUDA bugs, synchronization-focused).

**Equivalence Checking of ML GPU Kernels (VOLTA)** — Dubey, Driscoll, Wei, Kayal, Sharma, Aiken.
arXiv:2511.12638, 2025. [PDF](./papers/2511.12638.pdf)
The first formal equivalence checker for GPU kernels — **sound and complete** for a class where
tile sizes are statically known and per-thread branch targets and addresses are statically fixed
given `tid`. Verifies convolutions, matmuls, attention.
*Relevance:* **read before finalizing scope.** It defines the boundary where formal methods
already win; our value proposition is precisely the kernels outside that class (data-dependent
control flow, dynamic shapes, autotuned configs, unspecified backend numerics). State the
boundary explicitly or a reviewer will ask.

**Hunting CUDA Bugs at Scale with cuFuzz** — Mohamed Tarek Ibn Ziad, Christos Kozyrakis
(NVIDIA/Stanford). OOPSLA 2026. [PDF](./papers/cufuzz-oopsla2026-nvidia.pdf)
Identifies three obstacles to GPU fuzzing — kernel-level fuzzing causing false positives, no
device-side coverage feedback, tool incompatibility between coverage and sanitization — and
addresses them with NVBit instrumentation and separated sanitization passes. **43 previously
unknown bugs.**
*Relevance:* vendor state of the art for whole-program CUDA fuzzing. Read it to know what is
already covered, so our kernel-level property layer targets the complement.

**NVIDIA Compute Sanitizer docs** —
<https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>
`--tool` accepts exactly `memcheck` (default), `racecheck`, `initcheck`, `synccheck`.
*Relevance and load-bearing caveat:* **`racecheck` only detects races in on-chip shared memory**
— "Currently, this tool only supports detecting accesses to on-chip shared memory" — and **no
subtool detects global-memory races.** The four tools do not compose (four runs, a 4× budget
multiplier). `--error-exitcode <n>` exists specifically for automated test suites and is the CI
gate; output is **XML only**, no JSON; `--padding` adds red zones; `--suppressions` baselines
known-benign warnings. `cuda-memcheck` was deprecated in CUDA 11.6 and removed in 12.0. There is
**no device-side ASan on NVIDIA** and **no TSan for GPU device code on either vendor**; AMD ROCm
does have a real device ASan. For global-memory races see
[iGUARD](./papers/iguard-sosp2021.pdf) (SOSP 2021 — runs detection on the GPU, covers scoped
sync and Volta+ independent thread scheduling) and CURD (PLDI 2018, paywalled).

**Also present:** `2202.09947.pdf` (Tzer, OOPSLA 2022 — joint IR + pass-sequence mutation for
TVM; the Triton analogue is mutating the kernel jointly with its autotune config),
`2201.06589.pdf` (FreeFuzz, ICSE 2022), `2212.14834.pdf` (TitanFuzz, ISSTA 2023),
`2304.02014.pdf` (FuzzGPT — ⚠️ published ICSE 2024 title is "…Edge-Case **Generators**…"),
`2310.15991.pdf` (WhiteFox, OOPSLA 2024 — the closest thing to an XLA fuzzing paper),
`2208.02193.pdf` (HirGen, ISSTA 2023 — combines differential *and* metamorphic oracles with an
ablation; a good evaluation template), `2310.20078.pdf` (TorchProbe — 20 new bugs in the PyTorch
compiler **and Triton**), `dlcompiler-bugs-study-fse2021.pdf` (603 DL-compiler bugs by root
cause — read first for the prior distribution), `ptx-memory-model-asplos2019.pdf`,
`gpu-concurrency-weak-behaviours-asplos2015.pdf`, `weak-memory-gpu-errors-pldi2016.pdf`
(memory stressing + schedule fuzzing to force latent weak-memory bugs to manifest),
`gklee-ppopp2012.pdf`, `scord-isca2020.pdf`, `ld-taco2017-gpu-race-detection.pdf` (value-based
snapshot diffing — no instrumentation needed, so it transfers to Triton and NKI),
`2603.05725.pdf`, `2601.01048.pdf`, `2505.20207.pdf`, `2604.02106.pdf`, `esbmc-gpu-scp2017.pdf`,
`sun-compiler-bug-study-issta2016.pdf`, `spirv-fuzz-pldi2021.pdf` (also in §3).

**AWS Trainium / NKI:** no published testing or verification research exists. The official
surface is `nki.simulate_kernel`
(<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.simulate_kernel.html>)
and `nki.baremetal`; neither has built-in correctness checking, and both docs demonstrate
correctness with a manual `np.allclose()`. The real policy is in
<https://github.com/aws-neuron/nki-samples/blob/main/CONTRIBUTING.md>, which requires
`nki.baremetal` accuracy tests against a CPU reference and notes "Kernels with *only* simulation
tests will not be accepted." See NOTES §5.5 — this is the project's clearest novelty claim.

---

## 7. Evaluation methodology for testing research

How the project's own experiments will be judged. Protocol in NOTES §6.

**Evaluating Fuzz Testing** — George Klees, Andrew Ruef, Benji Cooper, Shiyi Wei, Michael Hicks.
CCS 2018. [PDF](./papers/1808.09700.pdf) · <https://arxiv.org/abs/1808.09700>
Surveys 32 fuzzing papers and finds essentially all of them methodologically deficient. Concrete
recommendations: multiple trials with statistical tests; benchmarks with **known** bugs;
performance measured in ground-truth bugs, not crash counts or coverage heuristics; documented
seed choices including the empty seed; and long timeouts with performance **plotted over time**.
Demonstrates a case where AFLFast beats AFL at 5 h and the trend **reverses by 24 h**, and shows
AFL's crash deduplication inflating one bug into ~500 "unique" crashes.
*Relevance:* the field's standard methodology stick. A testing-venue reviewer will expect us to
have internalized it, and the recommendations transfer to oracle-strategy comparison directly.

**A Hitchhiker's Guide to Statistical Tests for Assessing Randomized Algorithms in Software
Engineering** — Andrea Arcuri, Lionel Briand. STVR 24(3):219–250, 2014.
[PDF](./papers/hitchhikers-guide-statistical-tests-stvr2014.pdf)
Use **Mann–Whitney U**, not a t-test (randomized SE algorithms violate normality). Always pair
p-values with a non-parametric effect size, **Vargha–Delaney Â₁₂**, because with enough runs
"one would detect statistically significant differences on practically any experiment." Run
n ≥ 1000 per artifact if you can — but explicitly licenses the trade-off we need: with execution
constraints, "execute less runs per artifact (**though at least n = 10**) and use more artifacts."
*Relevance:* our licence to run 10–30 trials across many kernels rather than 1000 trials across
three.

**Are Mutants a Valid Substitute for Real Faults in Software Testing?** — René Just, Darioush
Jalali, Laura Inozemtseva, Michael Ernst, Reid Holmes, Gordon Fraser. FSE 2014.
[PDF](./papers/mutants-valid-substitute-real-faults-fse2014.pdf)
357 real faults across 5 Java programs. The coupling effect holds for **73%** of real faults;
mutation score correlates with real-fault detection **even after controlling for code coverage**,
more strongly than statement coverage does. But **17% of real faults are coupled to no mutant at
all** — "a fundamental limitation of mutation analysis." Conditional-operator, relational-operator
and statement-deletion mutants are disproportionately the ones that couple.
*Relevance:* the positive case for mutants, and the source of the 17% ceiling we should state
ourselves before a reviewer does. Also tells us which operator families to prioritize.

**Are Mutation Scores Correlated with Real Fault Detection?** — Mike Papadakis, Donghwan Shin,
Shin Yoo, Doo-Hwan Bae. ICSE 2018.
[PDF](./papers/mutation-scores-real-fault-detection-icse2018.pdf)
The rebuttal: reported correlations are largely "the results of the confounding effects of the
test suite size." Uncontrolled correlations of 0.35–0.75 **collapse to ~0.05–0.20** once suite
size is controlled, on both Defects4J and CoreBench. But selecting the *top-ranked* suites by
mutation score (rather than random suites of the same size) "improves significantly the fault
detection."
*Relevance:* **the reviewer's objection, pre-written.** The answer is that our batch-first
architecture holds the budget fixed by construction — every arm evaluates over the identical
recorded batch. Say it explicitly.

**Is Mutation an Appropriate Tool for Testing Experiments?** — James Andrews, Lionel Briand,
Yvan Labiche. ICSE 2005. [PDF](./papers/is-mutation-appropriate-tool-icse2005.pdf)
Concludes that "mutants, when using carefully selected mutation operators and after removing
equivalent mutants, can provide a good indication of the fault detection ability of a test
suite." Also finds **hand-seeded faults were harder to detect than real faults**.
*Relevance:* that second finding is useful in our defence — hand-seeding biases *against* the
technique under test, not for it. Tiny subject programs, no coverage control; don't cite alone.

**Mutation Testing Advances: An Analysis and Survey** — Papadakis, Kintis, Zhang, Jia,
Le Traon, Harman. Advances in Computers 112:275–378, 2019.
[PDF](./papers/mutation-testing-advances-survey-2019.pdf) ·
<https://mutationtesting.uni.lu/survey.pdf>
§9.3–9.4 are the methodologically load-bearing part. On **subsumed mutants**: "one test technique
might achieve a significant advantage over another by killing redundant [rather] than
non-redundant mutants", with an estimated **>60% chance of compromised conclusions** in arbitrary
experiments; disjoint mutants are ~9% of all mutants, minimal mutants 1.2–4%. Explicit
recommendation to identify and discard subsumed mutants before any assessment, with a greedy
dynamic algorithm given. On equivalence: compiler-equivalence (TCE) techniques remove "up to 90%
of the equivalent mutants, with approximately 10% or less of test effectiveness loss." On
**suite strength**: "there is no practical difference between test criteria when relatively
low-strength test suites are used."
*Relevance:* **the single most useful item for defending our methodology.** Everything in
NOTES §6.2's filtering pipeline comes from here.

**Defects4J: A Database of Existing Faults to Enable Controlled Testing Studies for Java
Programs** — René Just, Darioush Jalali, Michael Ernst. ISSTA 2014.
[PDF](./papers/defects4j-issta2014.pdf)
357 real bugs (now 835+) from real Java projects, each with a minimized bug-inducing diff, buggy
and fixed versions, and at least one triggering test.
*Relevance:* the template for the real-fault arm. ⚠️ **No equivalent exists for GPU kernels or
numerical code** — see `2504.17977.pdf` (*From Bugs to Benchmarks*, a survey of defect datasets)
to justify that claim, and NOTES §6.2 for what to do instead.

**Lessons from Building Static Analysis Tools at Google** — Caitlin Sadowski, Edward Aftandilian,
Alex Eagle, Liam Miller-Cushon, Ciera Jaspan. CACM 61(4):58–66, 2018.
[PDF](./papers/lessons-static-analysis-google-cacm2018.pdf)
Defines the **"effective false positive"**: an issue where developers did not take positive
action, regardless of whether the tool was technically right. Operating thresholds: compile-time
checks essentially zero-FP; **code-review checks allowed up to 10% effective FPs**; Tricorder
auto-disables an analyzer whose "not useful" ratio exceeds 10%.
*Relevance:* the citation for a defensible FP budget, with the denominator being developer action
rather than tool-author ground truth. Adopt the definition wholesale.

**Can Large Language Models Write Good Property-Based Tests?** — Vasudev Vikram, Caroline
Lemieux, Joshua Sunshine, Rohan Padhye. arXiv:2307.04346.
[PDF](./papers/2307.04346.pdf) · <https://arxiv.org/abs/2307.04346>
40 Python library API methods across GPT-4, Gemini-1.5-Pro and Claude-3-Opus. Evaluates
validity, soundness, and a new **property coverage** metric computed via property mutants.
Models produce a valid+sound PBT in 2.4 samples on average, but GPT-4 synthesizes correct PBTs
for only **21%** of the properties extractable from API documentation.
*Relevance:* two things. The 21% figure is the argument for a hybrid rather than a pure-property
approach. And property-coverage-via-property-mutants is a **directly reusable metric** — the
nearest thing to a validated notion of property adequacy. ⚠️ Note the authorship: this is
*not* by Vasudevan or Goldstein.

**Also present:** `tyche-pbt-effectiveness-uist2024.pdf` (Tyche, UIST 2024 — visualizing
generator distribution and input-space coverage for Hypothesis; the closest tooling for
"is my property suite any good?"), `jia-harman-mutation-survey-crest-tr-09-06.pdf` (Jia & Harman
2011, the standard survey — this is the CREST preprint of the TSE version),
`etna-icfp2023-lampropoulos.pdf` (also §1).

---

## 8. Prior art: PBT × LLM kernel generation

**Read this section before writing any paper.** The novelty window narrowed sharply in mid-2026.
Full assessment in NOTES §7.

⚠️ **Quality caution up front:** several of the most-threatening kernel papers below are
solo- or duo-author, non-peer-reviewed arXiv preprints — two by the same author — evaluated on
self-seeded bugs. Their *framing* is established prior art and must be cited; their *numbers*
are not established results.

**The Correctness Illusion in LLM-Generated GPU Kernels** — Dipankar Sarkar. arXiv:2606.20128,
June 2026. [PDF](./papers/2606.20128.pdf) · <https://arxiv.org/abs/2606.20128>
Attacks the premise that KernelBench, TritonBench and GEAK evaluate correctness adequately with
"fixed-shape, small-sample allclose-style checks." Builds a controlled corpus of 24 kernels (15
correct controls, 9 with documented transcription errors) and re-evaluates under op-schema-aware
seeded fuzzing against an fp64 CPU reference, flagging 9/9 buggy and passing 15/15 correct,
identically across five GPU classes.
*Relevance:* **threatens the project's motivation** — "reference/allclose oracles for LLM kernels
are inadequate" is now published and must be cited as established rather than claimed. It does
**not** threaten the oracle-strategy comparison: it compares only reference-oracle *variants*,
has no declarative or hybrid arm, and uses replay solely to reproduce individual failures.

**Test-Input Generation for Tensor Programs: What Actually Finds Kernel Bugs** — Dipankar Sarkar.
arXiv:2606.27396, June 2026. [PDF](./papers/2606.27396.pdf)
Compares **seven test-input-generation strategies** across 26 ops. Boundary-only shape sampling
is the "operationally safe winner" at 78% recall / 0% false positives; adversarial value sampling
hits 99% recall but **94% false positives**, because NaN/Inf injection trips validators on
correct kernels too. Boundary shape sampling is decisive for softmax tail-mask bugs.
*Relevance:* **the single most actionable paper for our generator design** (NOTES §5.2), and the
closest thing to our experimental design — but its axis is *input generation*, not *oracle
strategy*. Cite it and state the orthogonality. Its related work also usefully observes that
kernel-level metamorphic relations "[have] not widely propagated to the LLM-kernel ecosystem."

**A Contract-Grade Verifier for LLM-Generated GPU Kernels, and a Native Blackwell Backward for
the Gated-Linear-Recurrence Family** — Rishi Shah, Rishav Shrestha. arXiv:2608.12700, 13 Aug 2026.
[PDF](./papers/2608.12700.pdf) · <https://arxiv.org/abs/2608.12700>
Twelve adversarial gates, split into **tolerance-free** (NaN/Inf position and sign parity;
subnormal FTZ parity; bitwise run-to-run determinism and non-aliasing) and tolerance-based
(diverse inputs, autograd-vs-finite-difference, shape generalization, reordered summation within
√N rounding bounds, adversarial reductions, precision ladders, device residency, register/shared
memory budget). Audit of **2,638 machine-generated kernels already accepted by existing systems**:
**39.5% broken beyond any tolerance argument, 62.1% violating ≥1 property.**
*Relevance:* **the biggest threat to the property catalogue** — several properties we would
naturally propose are already published as a named gate list with large-N results. But it does
**not** compare oracle strategies (single ground-truth strategy, no per-strategy statistics) and
it **re-seeds and re-executes per gate**, which is architecturally the opposite of batch
record/replay — meaning its per-gate numbers are confounded by execution variance. Treat its
twelve gates as an *input* to our declarative arm. Posted the day before this library was built;
**re-run the prior-art search before submission.**

**Kernel Contracts: A Specification Language for ML Kernel Correctness Across Heterogeneous
Silicon** — Cooper Veit. arXiv:2604.22032, April 2026. [PDF](./papers/2604.22032.pdf)
A formal specification framework with eight components — identifier, scope, precondition,
postcondition, tolerance, **reference oracle**, measurement protocol, violation signature — and
twelve contract categories over precision, ordering, compiler-induced and exceptional-value
failure modes. Requires three-state calibration: each contract needs ≥1 conforming implementation
and ≥1 violating implementation that *passes standard functional testing*. Applied to three real
incidents including the Sakana AI CUDA Engineer reward-hacking episode.
*Relevance:* threatens the "declarative contract for kernels" framing. Note it makes "reference
oracle" a *field inside* a contract rather than a competing strategy — which is exactly the axis
we want to separate and measure. It is a position/framework paper with no oracle-strategy
experiment.

**Tensor Algebraic Property Skeletons: Amplifying Property-Based Testing for AI Compilers
(Propilot)** — Yuxin Qiu, Ben Limpanukorn, Seongmin Lee, Jiyuan Wang, Qian Zhang, Miryung Kim.
arXiv:2606.06747, June 2026. [PDF](./papers/2606.06747.pdf) · <https://arxiv.org/abs/2606.06747>
LLM-driven agentic PBT for DL compilers. Encodes tensor algebra as reusable **property
skeletons**, each with operator constraints and oracle templates, instantiated into executable
PBTs (paired computation graphs + inputs + expected semantic relations as oracles), with each
candidate validated for applicability and safety before execution. 212 TVM operators, 20
skeletons, 4,579 PBTs; 49% redundancy reduction versus direct LLM generation.
*Relevance:* **the strongest-pedigree adjacent work** and the closest published architecture to
ours. Threatens "LLM-generated declarative tensor-algebra properties" as a novel idea. Does not
threaten our target: the system under test is a *compiler*, not a kernel; there is no
reference-vs-property comparison and no record/replay. Read it before finalizing the design; the
skeleton abstraction is probably reusable.

**LLM-Based Test Oracles: Source-of-Authority Taxonomy — A Systematic Literature Review** —
Ali Hassaan Mughal, Muhammad Bilal. arXiv:2607.05031, July 2026 (rev. Aug 2026).
[PDF](./papers/2607.05031.pdf) · <https://arxiv.org/abs/2607.05031>
PRISMA review screening 2,436 records to 83 studies, read along three axes: the source of an
oracle's authority, its form, and its adjudication mechanism. "Just over half of the corpus
reaches a verdict with no specification at all." And, verbatim: **"Oracle quality is most often
judged by resemblance to a known oracle rather than by whether injected faults are caught."**
*Relevance:* **the best gap citation available.** A 2026 systematic review of 83 studies stating
that oracle strategies are not evaluated by fault detection is close to an explicit endorsement
of this project's experiment. Lead the motivation with it.

**ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants** — Mai, Guo, Ding, Li, Yu, Guo,
Wang, Zhao, Kozyrakis, Yuan. arXiv:2604.18616, April 2026. [PDF](./papers/2604.18616.pdf)
A Pythonic DSL with tag functions propagating symbolic annotations, and data-flow invariants
verified **at compile time** via abstract interpretation over a layout algebra plus SMT, at zero
runtime overhead; violations yield concrete counterexamples naming offending threads/elements.
99–104% of hand-optimized throughput on MI300X.
*Relevance:* **strengthens** us. It is the leading declarative-spec-driven kernel-generation work,
but its invariants are structural/layout, checked statically, and serve performance guidance.
Ours are semantic/numerical, checked dynamically, and evaluated *as oracles*. Complementary, and
good evidence that declarative kernel specs are a live direction.

**Oracle practice in the kernel benchmarks** (these are catalogued in
`reference/L2-benchmarks/`; local copies of the two most-cited are kept here for convenience):
**KernelBench** (ICML 2025, [PDF](./papers/2502.10517.pdf)) checks correctness with `torch.allclose` against
the reference module on **5 sets of random inputs at fixed shapes**; its Appendix B.2 justifies
n=5 by reporting that of 100 generated kernels, results were 0/5 or 5/5 with "no partial
correctness observed" — a claim worth interrogating, since it suggests either that their bug
distribution is gross rather than subtle, or that the oracle is insensitive. **TritonBench**
(ACL Findings 2025, [PDF](./papers/2502.14752.pdf)), **ParEval** (arXiv:2401.12554) and
**MultiKernelBench** (arXiv:2507.17773) use the same family of fixed-shape reference checks;
see `reference/L2-benchmarks/README.md`.
**Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization**
([PDF](./papers/2509.14279.pdf)) documents exploitable benchmark loopholes — omitting redundant operations,
overfitting input settings, non-generalizing implementations — and adds a new verifier workflow.
Sakana's own account of the AI CUDA Engineer episode is at
<https://x.com/SakanaAILabs/status/1892992938013270019>: the system found a memory exploit in
the evaluation harness that bypassed the correctness check, partly by reusing results from an
earlier PyTorch run; after excluding contaminated tasks the aggregate speedup over 200
KernelBench tasks fell from **3.13× to 1.49×**.

**Non-kernel LLM × PBT work** (established, cite rather than claim):
`2506.18315.pdf` (Property-Generated Solver — PBT breaks the "cycle of self-deception" where
LLM-written example tests share the code's flaws; +23–37% relative pass@1 over TDD baselines),
`2510.25297.pdf` (⚠️ **the sleeper threat** — on 16 HumanEval problems, PBT alone and
example-based tests alone each detect 68.75% of bugs while **combining both reaches 81.25%**;
our hybrid result's *shape* is pre-empted unless we say why kernels differ),
`2510.09907.pdf` (Agentic PBT at Python-ecosystem scale), `2406.06864.pdf` (metamorphic relations
over *paraphrased prompts* — a genuinely different mechanism), `2401.17019.pdf` (LLM-generated
executable MRs), `2310.01831.pdf` (nl2postcond, FSE 2024 — LLM-written postconditions catch 64
real Defects4J bugs), `2405.03786.pdf` (TOGLL, ICSE 2025), `2408.15815.pdf` (MR-Adopt),
`2305.01210.pdf` (EvalPlus), `2207.10397.pdf` (CodeT), `2305.14591.pdf` (ALGO),
`2511.12294.pdf` (ProofWright — agentic formal verification of CUDA, NVIDIA/Stanford),
`2607.16241.pdf` (KernelBench-Verified), `2607.04395.pdf` (NKI-Agent — the only NKI
kernel-generation paper; verification is compile-and-compare only).

---

## 9. Antithesis: deterministic simulation and record/replay

**Property-Based Testing** — Antithesis docs.
<https://antithesis.com/docs/resources/property_based_testing/>
The project's starting reference. Frames PBT at three levels — unit (Hypothesis, FsCheck),
program (AFL-style fuzzing), and distributed systems (their platform) — and argues the value is
that "random choice of inputs can find 'unknown unknowns'." Cites QuickCheck and Wlaschin.
*Relevance:* a good orientation piece, thin on technique. Its real value is the pointer to their
assertion taxonomy below.

**Assertions / Sometimes Assertions** — Antithesis docs.
<https://antithesis.com/docs/concepts/properties_assertions/assertions/>
The taxonomy, with their exact names: **`always(...)`** (a single false evaluation fails the
property); **`alwaysOrUnreachable(...)`** (must hold whenever reached, but "never reached" also
passes — for optional or rare paths); **`sometimes(...)`** ("should be true at least once during
the run… especially useful for checking whether tests exercise meaningful scenarios, not just
lines"); and **`reachable()` / `unreachable()`**. Properties are aggregated across runs by the
**`message` parameter**, which acts as the stable identifier — changing the message breaks
historical tracking, while moving the assertion does not.
*Relevance:* **`sometimes` is the idea to steal, and it is directly applicable.** A property
suite that never exercises the masked path, the ragged tail tile, the subnormal input, or the
`split_k > 1` configuration passes *vacuously*. `sometimes` assertions turn "did we actually test
this?" into a first-class, machine-checkable artifact rather than a hope. Given our offline
oracle evaluation over a recorded table, `sometimes` becomes trivially cheap: it is a `GROUP BY`
over the batch. **We should implement this.** The `message`-as-stable-identifier convention is
also worth copying for our property IDs.

**Deterministic Simulation Testing** — Antithesis docs.
<https://antithesis.com/docs/resources/deterministic_simulation_testing/>
Runs "regular non-deterministic software inside a deterministic hypervisor", paired with
property-based testing/fuzzing and fault injection, so that "execution can be rolled back and
inspected at multiple points in time" ("multiverse debugging"). The lineage is FoundationDB's
simulation-first testing framework, via founder Will Wilson.
*Relevance and honest divergence assessment:*
- **Aligns:** the core conviction that reproducibility is an architectural property, not a
  testing convenience; and that a recorded execution should be re-examinable many times.
- **Diverges, importantly:** their determinism is achieved by *controlling the execution
  environment* (a hypervisor intercepting all sources of non-determinism) so that the *same*
  execution can be replayed and branched. **We cannot do this** — a GPU or Trainium device is
  not deterministically simulable, floating-point reduction order is a legitimate hardware
  degree of freedom, and there is no hypervisor that will give us bitwise-reproducible tensor
  cores. So our record/replay is **record-outputs-and-replay-oracles**, not
  record-execution-and-replay-execution. We can re-evaluate many oracles over one execution; we
  cannot re-execute deterministically or branch a run.
- **Consequence to state in the paper:** our design is weaker than Antithesis's in reproduction
  power and stronger in *comparative* power. Because the outputs are frozen in the table, N
  oracle strategies see byte-identical executions — which is precisely the confound elimination
  the experiment needs, and which Antithesis's model does not itself provide.
- **Their autonomous exploration does not transfer** either: it explores a state space of
  schedules and injected faults, which for a kernel is largely a fixed dataflow. The transferable
  half is `sometimes`.

**Further reading (blog-tier, useful for engineering context):** the CockroachDB write-up on
taming demonic nondeterminism
(<https://www.cockroachlabs.com/blog/demonic-nondeterminism/>) and WarpStream's account of
applying DST to a whole SaaS
(<https://www.warpstream.com/blog/deterministic-simulation-testing-for-our-entire-saas>).
Neither is research; both are honest about the engineering cost.

---

## Conventions

- PDFs are named `{arxiv_id}.pdf` for arXiv papers, or a descriptive slug for non-arXiv
  (e.g. `csmith-pldi2011.pdf`), matching `reference/README.md`.
- arXiv canonical URL: `https://arxiv.org/abs/{id}`.
- Refresh a paper: `curl -fsSL -o papers/ID.pdf "https://arxiv.org/pdf/ID.pdf"`.
- Everything downloaded here is arXiv, LIPIcs/PACMPL/PeerJ open access, an author's own posted
  copy, or an institutional repository copy. **Paywalled work is cited by URL only** and listed
  in NOTES §8.
- All entries are indexed in `reference/manifest.csv` with level `PBT`.

## Next

→ [NOTES.md](./NOTES.md) — the property taxonomy with worked kernel examples, the
Hypothesis-vs-custom-generator decision, tolerance guidance, the experimental protocol, and the
open questions.
