# The softmax bundle cannot see temperature: the gate admits softmax(2x), 1/n and softmax(−x)

**Date:** 2026-10-01
**Instance:** Apple M4 Pro, macOS 26.6.1, CPU only
**Environments:** gate verdicts in the project's pinned environment (Python 3.10.20, NumPy 1.26.4,
OpenBLAS) and again in the one `2026-08-18-autoresearch-loop-pbt-gate.md` was measured under
(Python 3.12.13, NumPy 2.5.2, Accelerate): **identical, cell for cell**, the log-ratio column
included. The training numbers in §5 come from the second; §7 says why.
**Harness:** `experiments/autoresearch-pbt/pbt_gate.py`, unmodified — `check_task(kernel, "softmax")`,
`GATE_SEED = 42`, 9 groups, NumPy backend
**Companion:** `2026-08-18-autoresearch-loop-pbt-gate.md`, §3 (the saboteur matrix) and §5 (the
fast-exp blind spot)
**Provenance:** first seen in a throwaway check earlier the same day; re-measured here from code
that stays in the repo, and every cell of that check reproduced.

**Headline.** The property gate admits softmax at the wrong temperature. It decides on the
declarative arm alone, and every law in the softmax bundle holds for softmax(βx) at **every** real
β — so softmax(2x), the uniform 1/n and the ranking-reversed softmax(−x) pass it on 9 of 9 groups,
while `allclose`, the reference arm and the hybrid arm reject each of them on 7 of 9, the most any
arm can score on this ladder. It is the **first detection difference between arms** this project
has measured, and it runs against the declarative arm. Downstream, the autoresearch loop's
`val_bpb` happens to reject the 1/n softmax — but it cannot see softmax(2x), and a loop that
optimized kernel speed would converge on the 1/n: the gate admits it, and written as a constant
fill it runs **52–54× faster** than the loop's softmax.

---

## 1. Construction

Stated before the numbers, per the repo standard.

| | |
|---|---|
| gate | `pbt_gate.check_task`, unmodified. Its decision is the declarative arm's alone (`pbt_gate.py:122`); all four arms are reported |
| bundle | `kernels/tasks/softmax/acceptance.yaml`, unmodified: `outputs_are_finite`, `values_in_unit_interval`, `rows_sum_to_one`, `shift_invariance` (relation `shift_rows`) |
| cases | `GATE_SEED = 42`, `GATE_GROUPS = 9`: one group per ladder rung, each a base case plus one `shift_rows` partner — 18 executions per kernel |
| unit | groups failed of 9, per arm, exactly as `check_task` reports them |
| reproduce | `cd experiments/autoresearch-pbt && ../../.venv/bin/python saboteurs.py` prints §2's table |
| pinned by | `experiments/autoresearch-pbt/test_gate_saboteurs.py` (§6) |

The temperature family, with the parameter named `x` because the backend calls `kernel(**inputs)`:

```python
def softmax_beta(beta):
    b = np.float32(beta)
    def kernel(x):
        z = np.multiply(x, b, dtype=np.float32)
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=-1, keepdims=True)
    return kernel
```

At β = 1 this is bit-for-bit the stock kernel, and it is the control: whatever the other rows get
wrong, they get wrong through β alone. Three more rows give the table its reading:

- **constant fill** — β = 0 written the way a speed-optimizing loop would write it,
  `np.full(x.shape, 1/n)`, with no arithmetic at all;
- **fast-exp** — `candidates/r3_softmax_fastexp.py`, the companion's §5 blind spot, for contrast;
- **positive control** — exp(x − max), never divided: the repo's canonical normalization bug, and
  the one row the gate must *reject*. Without it a gate that admitted everything would satisfy the
  whole matrix.

## 2. The result

Groups failed of 9, and the gate's decision:

| kernel | allclose | reference | declarative | hybrid | gate | log-ratio spread |
|---|---|---|---|---|---|---|
| softmax(x), β = 1 (control) | 0 | 0 | 0 | 0 | admits | 4.8 |
| **softmax(2x)** | 7 | 7 | **0** | 7 | **admits** | 5.2e7 |
| **uniform 1/n, as softmax(0x)** | 7 | 7 | **0** | 7 | **admits** | 5.2e7 |
| **softmax(−x)**, ranking reversed | 7 | 7 | **0** | 7 | **admits** | 1.04e8 |
| **uniform 1/n, as a constant fill** | 7 | 7 | **0** | 7 | **admits** | 5.2e7 |
| fast-exp softmax | 0 | 0 | 0 | 0 | admits | 48 |
| exp(x − max), never divided (positive control) | 7 | 7 | 7 | 7 | rejects | 4.3 |

**Attribution, which the counts alone hide.** On the four temperature rows the hybrid arm catches
through `matches_reference` — its reference half; the declarative half passed, so nothing
short-circuited. On the positive control it catches through `rows_sum_to_one` — its declarative
half. Same 7/9, opposite halves. The reference arm's test ratios on the temperature rows run from
1.1e6 to 4.0e6 against a threshold of 30: these are not near misses for anyone.

**The two uncaught groups** are the single-column rungs (1, 1) and (17, 1), where the softmax of a
single element is 1.0 at any β, and exp(x − max) of one element is 1.0 too. Every kernel in the
table is *correct* there, so 7 of 9 is the ceiling for every arm (CLAUDE.md, open obligation 3).

**The log-ratio column is a measurement, not a property.** It is the worst row spread of
(log y − x), computed in float64 and expressed in float32 eps, over the same 18 cases. No arm reads
it and the gate does not consult it; §3 says why it is printed.

## 3. Why it happens

In exact arithmetic every law in the bundle holds for softmax(βx), for every real β:

- **finite, in [0, 1], rows sum to one** — any softmax output is a probability vector;
- **shift invariance** — softmax(β(x + c)) = softmax(βx + βc) = softmax(βx), because βc is itself a
  per-row constant.

So the bundle's solution set contains the whole one-parameter family, including β = 0 (a constant)
and every β < 0 (the ranking reversed). It contains far more — softmax(f(x − max x)) passes for any
finite elementwise f — but this family is the simplest member, and it contains the cheapest kernel
there is.

**None of the four laws relates an output value to an input value.** Three read the output alone.
The fourth compares the output with itself, under a transform the whole family respects. The law
that ties output to input is the log-ratio law: log y_i − x_i is constant along a row (equivalently
y_i / y_j = e^(x_i − x_j)). With rows summing to one it pins the output exactly — y_i = e^(x_i + k)
and Σ y_i = 1 force e^k = 1 / Σ_j e^(x_j). Its spread separates every row of §2: 4.8 eps for the
exact kernel, 48 for fast-exp, and |β − 1| times the row's range for the temperature family
(5.2e7 at β = 2 and β = 0, which share |β − 1| = 1; twice that at β = −1).

The positive control is the mirror image. Its log-ratios are exact (log y − x = −max x, spread
4.3 eps) and its row sums are wrong; the temperature family's row sums are exact and its log-ratios
are wrong. Each law sees exactly what the other cannot, which is why the bundle, which has row sums
and no log-ratio law, catches the positive control and none of the four temperature rows.

**It is not a near miss, and no threshold would fix it.** On softmax(2x) the declarative arm's
worst margins are a shift ratio of 6.49 and a row-sum ratio of 0.36, both against 30. A sweep over β
on the same 18 cases (characterization only, one seed): the declarative arm fails **0 of 9** groups
at every one of 16 values tested in [−8, 8], while the reference arm fails 7 of 9 at every β ≠ 1.
The first declarative detection is at **β = 16, one group, through `shift_invariance`** — the
float32 rounding of x + c amplified by β (worst shift ratio 4.1 at β = 1, 14.2 at β = 4, 25.8 at
β = 8, 31.9 at β = 16). That is the arm measuring float error, not the law. In real arithmetic the
family satisfies all four laws exactly, so there is no tolerance to tighten.

**Relation to the companion's §5.** There the bundle could not price `exp`: a 4.5e-6 error lived
inside every tolerance. This is the same structural gap at five orders of magnitude more, by the
reference arm's own ratio (~10^6 here against ~14 there) — and with no tolerance involved at all.
§5 was a blind spot inside the bundle's tolerances; this one is outside the bundle's laws.

## 4. It is the gate's choice of arm that admits them

The hybrid arm rejects all four temperature kernels. The gate does not consult it: it decides on
the declarative arm, because `allclose` false-positives on 5 of 9 groups of a correct layernorm,
so a loop gated on `allclose` cannot accept its own baseline (companion §4). That choice was
justified against `allclose` alone; the companion's threats already record that the decisive arm
was chosen, not measured against the alternatives. This is the first measurement against one:
across the six wrong rows of §2, the hybrid arm rejects five and the declarative arm rejects one.
Both admit the correct kernel, and both admit fast-exp.

**The first detection difference between arms.** Every earlier measurement found the arms tied on
broken kernels: 20 mutants across three corpora with zero disagreements
(`2026-08-18-arm-differentiation-null-result.md`), and six more saboteurs in the companion's §3. The
one differentiation so far was on false positives (`2026-08-19-false-positive-rate.md`). Here the
arms disagree on four rows, and every disagreement is the declarative arm missing what a reference
catches. That does not overturn the null result; it sharpens it. Where a trusted reference exists
the declarative arm had no detection advantage — on this family it has a disadvantage.

## 5. What catches it downstream: val_bpb, partly; a speed objective, nothing

The autoresearch loop never scores the kernel. It scores a model that *calls* the kernel, so a
kernel the gate admits still has to survive training. Measured: `best/train.py` with only
`softmax`'s body replaced, one seed (`AUTORESEARCH_SEED=0`), the 60 s budget, NumPy 2.5.2 /
Accelerate (§7 says why that environment):

| softmax inside `best/train.py` | val_bpb | steps | what `runner.py` does with it |
|---|---|---|---|
| correct (stock) | 2.2292 | 1595 | — |
| softmax(2x) | **2.2279** | 1612 | **would keep it** on this seed: 2.2279 < 2.2292 |
| uniform 1/n, as a constant fill | 3.7897 | 1736 | reverts it |
| uniform 1/n, as softmax(0x) | NaN | 1658 | reverts it (`NaN < best` is False) |
| softmax(−x) | NaN | 1659 | reverts it |

The stock row reproduces the companion's winner (2.2280, sd 0.0147 over three seeds;
`best/train.py` is `r6_best_nolrmult`). Three different mechanisms are at work, and none is the gate:

- **The 1/n softmax is rejected by the objective, as expected.** Uniform attention throws away
  everything attention computes, and the fill pays +1.56 bpb for it despite 9% more steps. That the
  fill also lets most queries see future keys inside their block did not help it.
- **softmax(2x) is invisible to val_bpb, and there is nothing for val_bpb to see.**
  softmax(2·q·k) = softmax((2q)·k), and the query projection is learned, so the model absorbs the
  factor; the function class is unchanged. A model trained around the wrong kernel is a fine
  model. The kernel is still not softmax, and every other caller of it — inference against
  weights trained with a correct softmax, any consumer outside this loop — gets the wrong answer.
- **softmax(0x) and softmax(−x) are rejected by the model's causal mask, not by the objective.**
  The mask is additive −inf (`best/train.py`, `_mask`), so masked scores reach `softmax` as −inf:
  0 · (−inf) is NaN, and −(−inf) = +inf turns `z − max(z)` into inf − inf. Both runs warn exactly
  there. The gate draws finite N(0, 1) inputs plus finite shifts, so it never generates the one
  input class this call site always produces; here the model acted as an oracle on a domain the gate
  does not cover.

**A loop that optimizes kernel speed has no such backstop.** Its objective is the kernel's runtime,
which carries no correctness signal, so the gate is the only correctness pressure in that loop.
Measured on (512, 4096) float32, median of 7 × 20 calls, three timing runs across both
environments:

| kernel | ms/call | vs the loop's softmax |
|---|---|---|
| `best/train.py` softmax | 3.27–3.38 | 1.00× |
| softmax(βx), as written above, any β | 3.52–3.67 | 0.92–0.96× |
| uniform 1/n, as a constant fill | 0.060–0.065 | **52–54×** |

As written, the temperature kernels are *slower* — one extra multiply pass — so softmax(2x) and
softmax(−x) are blind spots but not reward hacks. Their β = 0 member, written without arithmetic, is
both: the gate admits it, it is 52–54× faster, and nothing in a speed objective pushes back. It is
the optimum of any loop whose only correctness check is this gate. The companion called the gate
"the half of the fitness function that keeps the other half meaningful"; on this family that half
is not there.

## 6. How the baseline is pinned

`experiments/autoresearch-pbt/saboteurs.py` holds the seven kernels and, for each, every arm's
expected (groups failed, first-failing property) and the gate's expected decision.
`test_gate_saboteurs.py` asserts them, one assertion per test, each the sole catcher of a defect the
other two cannot see. Verified by breaking the gate five ways in memory, without editing any source
file — each breakage fails exactly one test, on exactly its own rows:

| gate defect injected | test that fails | rows that fail |
|---|---|---|
| decide on the hybrid arm instead of the declarative one | the decision | the 4 temperature rows |
| admit every kernel | the decision | the positive control |
| reference arm blind (threshold ∞) | the pairing | 4 temperature rows + positive control |
| `allclose` arm blind | the pairing | 4 temperature rows + positive control |
| `shift_invariance` abstains on every group | the judgement | all 6 rows the declarative arm passes |

The third test exists because `check_task` counts only FAIL groups: an arm that abstained everywhere
would read "0 of 9 failed", exactly like an arm that judged and approved, and the pairing alone
would stay green. It spies on the declarative arm the gate itself built and requires every result
it produced on those rows to be a PASS.

## 7. Threats to validity

- **Author-chosen, not blinded.** The family was built by reasoning *from* the bundle's
  invariances — the opposite of `docs/protocol/mutant-authoring.md`. It demonstrates a structural
  gap; it is not a detection rate and must not be pooled with the corpus's.
- **One task, one seed, one dtype, one backend**: softmax, `GATE_SEED = 42`, 9 groups, float32,
  NumPy. Against that, every cell and the log-ratio column are identical under NumPy 1.26.4 /
  OpenBLAS and 2.5.2 / Accelerate.
- **7 of 9 is a ceiling, not a rate.** Both single-column rungs make every kernel here correct.
- **The gate counts abstention as a pass.** `check_task` counts FAIL groups only. The matrix guards
  its own rows (§6); the gate is still fail-open for any other kernel whose arm abstains.
- **The gate's domain has no −inf; the model's call site always does.** That is why the model
  rejects two of these kernels and the gate could not. A gate run on the call site's real inputs
  would score those two rows differently.
- **Training: one seed per row.** The companion's pooled sd is ~0.013, so softmax(2x)'s 2.2279
  against 2.2292 is a tie, not an improvement; "the runner would keep it" holds on this seed and
  is a coin flip in general. The fill's +1.56 and the two NaNs are far outside that noise.
- **The project's pinned environment does not reproduce the loop.** `pyproject.toml` pins
  `numpy<2`, which resolves here to NumPy 1.26.4 on OpenBLAS. Under it `best/train.py` takes 283
  steps in its 60 s instead of 1595 (9.6k vs 54.4k tok/s) and stops near val_bpb 3.28 — a regime
  where attention barely matters: the constant fill scored 3.33 there, within 0.05 of the correct
  kernel. The training numbers above come from NumPy 2.5.2 / Accelerate, the environment the
  companion records. Gate verdicts do not depend on the environment; the strength of `val_bpb` as a
  backstop depends on it heavily.
- **Speed: one machine, one shape, NumPy.**

## 8. What is NOT claimed

- **Not that the loop found these.** None of the companion's 26 candidates was a temperature kernel
  — every ledger row reads `softmax/reference=0/9`, which no temperature kernel can — and its 0
  in-loop rejections stand. Like the companion's §3, the matrix shows the gate *can* fail;
  it is not evidence that it did.
- **Not that the declarative arm is weaker in general.** One task, one constructed family. The
  claim is narrower: the softmax bundle is incomplete — its solution set is strictly larger than
  softmax — and a gate that decides on it inherits the gap.
- **Not that `val_bpb` is a backstop.** It rejected the 1/n softmax on merit, rejected two kernels
  only through NaN from this model's −inf mask, and is blind to softmax(2x) by construction.
- **Not that the hybrid arm should decide.** It rejects five of the six wrong rows here, but it needs
  a reference, which is what the declarative arm exists to do without, and it was not measured as a
  gate.
- **Not a fix.** The decisive arm and the bundle are unchanged on purpose: the current bundle is the
  baseline the planned completion experiment is measured against, and §6 keeps it reproducible. The
  log-ratio spread is a measurement printed beside the table, not a proposed property. Its
  floating-point form — the tolerance, and what to do when an output underflows to 0 and log y is
  −∞ — is that experiment's to design.
- **Not that the temperature family is the bundle's only gap.** softmax(f(x − max x)) passes for
  any finite elementwise f. The family is the simplest member, and it contains the cheapest kernel.

## 9. What the completion experiment inherits

- **A baseline that is meant to break.** A completed bundle should move the declarative cell of
  the four temperature rows from 0 to 7, flip the gate from admits to rejects on them, and leave the
  correct kernel and the positive control as they are. `test_gate_saboteurs.py` fails when that
  happens, by design; the rows are then re-recorded, not loosened until they pass. The fast-exp
  row is the open one: whether a completed law catches a 48-eps spread against the exact kernel's
  4.8 is a tolerance decision for that experiment, and the matrix records which way it went.
- **A precise target.** Over the reals, {finite, unit interval, rows sum to one, shift invariance}
  admits softmax(βx) at every β — a counterexample family, not a single counterexample — and adding
  the log-ratio law makes the solution set exactly softmax. That is a statement about the oracle,
  checkable before any kernel is run.
- **A spec, before code.** Completing the bundle edits `kernels/tasks/softmax/acceptance.yaml` and
  adds a property under `src/autokernel_pbt/props/`, so under `docs/adr/0001-sdd-tdd.md` it is a
  feature and needs `specs/features/NNNN-*/spec.md` and `acceptance.yaml` first. This measurement
  needed neither: it adds a document and test fixtures under `experiments/`, outside the
  `src/` and `harness/` code the SDD workflow governs, and changes no feature's behaviour.
