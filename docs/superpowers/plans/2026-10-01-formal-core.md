# Formal Adherence Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The language-neutral checks that decide whether an agent's formal proof of kernel–spec adherence certifies anything — each shown, before any formal language or agent exists, to catch exactly the cheat it is for.

**Architecture:** Additive, under `src/autokernel_pbt/formal/`. An `Attempt` is what a language adapter will hand the harness; a `Ground` is what it is judged against, realized once over the whole ladder through the existing `Generator` and `NumpyBackend`; six checks turn the pair into a `Certificate`. Nothing in the oracle layer, the driver or the storage changes.

**Tech Stack:** Python 3.10+, NumPy (<2), pytest, ruff. No new dependencies, no formal toolchain.

**Spec:** `docs/superpowers/specs/2026-10-01-agent-formal-adherence-design.md` — read §4 (what the agent produces) and §5 (the six checks) before starting; this plan builds exactly those, and only those.

## Global Constraints

- **Never `git commit`.** Stage, then `scripts/git_commit_clean.sh -m "subject" -m "body"`, then verify `git branch --show-current` is `formal-adherence`.
- `filterwarnings = ["error"]`: `pytest -m "not gpu" -q` must end green and silent. CI also runs `ruff check src tests` and `--cov-fail-under=95`.
- Every new assertion is the **unique** catcher of at least one saboteur, paired with its exact expected message.
- Bad **data** (an agent's reading misbehaving, a kernel failing) fails closed; a bad **call** (a ground that cannot judge) raises.
- `str`-mixin enums carry `__str__ = str.__str__`.
- From the design: the agent writes everything; the harness fixes only the theorem's *shape*; the harness owns tolerance; an attempt with no executable reading fails closed; no hidden kernel may be indistinguishable from the reference within the rounding budget.
- The spec test is committed red in Task 0 and turns green in Task 3, as feature 0008's did (`fce213f`).

## Review Focus

1. **An empty hidden set** — spec completeness would pass vacuously; `realize` must refuse it. Pinned in Task 2.
2. **A hidden kernel that is never wrong beyond the rounding budget** — it would make completeness unjudgeable for that kernel; refused. Pinned in Task 2.
3. **A model reading that returns a differently shaped array** — must fail model fidelity, never crash. Pinned in Task 3.
4. **A reading that returns something other than a boolean** (an array, `None`) — must fail closed rather than be read by truthiness. Pinned in Task 3.
5. **A reading that writes into its inputs, or rebinds one** — must be contained to its own check: the arrays every later check reads are read-only, and so is the mapping that holds them. Pinned in Tasks 2 and 3.

---

### Task 0: ADR 0002 and the feature 0009 spec (red)

**Files:**
- Create: `docs/adr/0002-formal-track.md`
- Create: `specs/features/0009-formal-core/spec.md`
- Create: `specs/features/0009-formal-core/acceptance.yaml`
- Create: `tests/spec/test_0009_formal_core.py`
- Modify: `specs/README.md` (feature index)
- Modify: `docs/superpowers/specs/2026-08-14-kernel-property-oracle-layer-design.md:449`

**Interfaces:**
- Produces: the six test node ids every later task must create, exactly as named in `acceptance.yaml`.

- [ ] **Step 1: Branch.**

```bash
git checkout -b formal-adherence
git branch --show-current
```
Expected: `formal-adherence`.

- [ ] **Step 2: Write `docs/adr/0002-formal-track.md`.**

```markdown
# ADR 0002: A formal track, narrowed to agent-written proofs about kernel code

## Status

Accepted (2026-10-01)

## Context

The parent design (`docs/superpowers/specs/2026-08-14-kernel-property-oracle-layer-design.md`,
§10) lists formal verification as a non-goal, and `reference/PBT-property-based-testing/NOTES.md`
§5.4 cedes equivalence checking of statically analyzable GPU kernels to VOLTA.

Two things changed. The property gate was measured admitting wrong kernels its laws cannot see
(`docs/measurements/2026-10-01-softmax-temperature-blind-spot.md`) — a limit of testing, not of any
one tolerance. And the project's direction is now that the kernel-writing agent also writes formal
proofs, in every formal language, cooperating with property-based testing rather than competing
with it.

## Decision

1. The non-goal is narrowed, not removed. In scope: agent-written proofs, in any formal language,
   that a given kernel adheres to a spec the agent also writes — and the measurement of how such
   proofs fail. Out of scope: equivalence checking of statically analyzable kernels, which VOLTA
   covers; human-written proofs; verifying the Triton compiler or the device's IEEE behaviour.
2. The formal track issues certificates. It is not an oracle arm and is not scored as one.
3. Proofs are post-hoc certificates on kernels PBT already accepted, never the optimization loop's
   filter.
4. The design is `docs/superpowers/specs/2026-10-01-agent-formal-adherence-design.md`.

## Consequences

- Features 0009 (the language-neutral core), 0010 (pilot adapters) and 0011 (the pilot run), each
  spec-first under ADR 0001.
- Formal toolchains are optional, like the GPU stack: their tests are gated behind a marker and
  skipped when the toolchain is absent, so CI stays CPU-only and toolchain-free.
- Every formal artifact's claims are executed by the PBT machinery — the same cases, backends and
  rounding budget — so the two tracks check each other rather than running side by side.
```

- [ ] **Step 3: Point the parent design's non-goal at the ADR.** Replace line 449 of the parent design,

```markdown
- Formal verification — the literature already wins where it applies; we do not compete there
```

with

```markdown
- Formal verification — the literature already wins where it applies; we do not compete there.
  Narrowed by `docs/adr/0002-formal-track.md`: agent-written proofs about kernel code are in scope;
  equivalence checking of statically analyzable kernels is not.
```

- [ ] **Step 4: Write `specs/features/0009-formal-core/spec.md`.**

```markdown
# Feature 0009: The formal adherence core

## Problem

An agent's proof that a kernel adheres to its spec certifies nothing by itself when the agent also
wrote the spec, the model of the kernel and the theorem: each can be fitted to make the proof go
through. Before any formal language or agent is involved, the checks that decide whether a proof
counts have to exist, and have to be shown to catch each way of cheating them.

## Scope

1. **The attempt** — what a language adapter hands the harness: the executable readings of the
   agent's spec, model and hypotheses; its axioms; what the language's checker found.
2. **The ground** — what an attempt is judged against: the whole ladder of generated cases, the
   reference, the kernel's own executions, and a hidden set of wrong kernels — refused if it
   cannot judge.
3. **Harness-owned comparisons** — closeness budgeted at the kernel output's dtype and the input's
   accumulation length, never by the spec.
4. **Six checks and the certificate record** — proof, spec soundness, spec completeness, model
   fidelity, non-vacuity, axioms. A certificate is issued only if all six ran and passed.

## Non-goals

- Any formal language or toolchain (feature 0010)
- Any agent run, and persisting certificates (feature 0011)
- Establishing that an adapter's readings come from the proved definitions — the adapter's
  obligation, specified with it

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
```

- [ ] **Step 5: Write `specs/features/0009-formal-core/acceptance.yaml`.**

```yaml
feature_id: "0009"
feature_name: formal-core
version: 1

criteria:
  - id: AN_HONEST_ATTEMPT_IS_CERTIFIED
    description: an attempt whose spec, model, hypotheses, proof and axioms are all sound passes all six checks
    check:
      type: unit_test
      test: tests/unit/formal/test_checks.py::test_an_honest_attempt_is_certified

  - id: EACH_CHEAT_HAS_ONE_CATCHER
    description: every cheating attempt is rejected by exactly the check it targets, and by no other
    check:
      type: unit_test
      test: tests/unit/formal/test_checks.py::test_each_cheat_is_rejected_by_exactly_its_own_check

  - id: NO_READING_FAILS_CLOSED
    description: an attempt with no executable reading cannot be certified
    check:
      type: unit_test
      test: tests/unit/formal/test_checks.py::test_an_attempt_without_executable_readings_is_not_certified

  - id: AGENT_CODE_CANNOT_ABORT_OR_CORRUPT_THE_CHECKS
    description: a reading that raises, returns a non-boolean, writes into its inputs or returns the wrong shape fails its own check and no other
    check:
      type: unit_test
      test: tests/unit/formal/test_checks.py::test_misbehaving_agent_code_fails_only_its_own_check

  - id: THE_HARNESS_OWNS_THE_TOLERANCE
    description: closeness is budgeted at the kernel output's dtype and the input's accumulation length, whatever the spec computes in
    check:
      type: unit_test
      test: tests/unit/formal/test_comparisons.py::test_the_budget_is_the_outputs_dtype_not_the_specs_arithmetic

  - id: A_GROUND_THAT_CANNOT_JUDGE_IS_REFUSED
    description: an empty hidden set, a hidden kernel never wrong beyond the rounding budget, a hidden kernel that fails to run, and a non-finite reference are each refused before any check runs
    check:
      type: unit_test
      test: tests/unit/formal/test_ground.py::test_a_ground_that_cannot_judge_is_refused
```

- [ ] **Step 6: Write the spec test.** Copy `tests/spec/test_0008_corpus_and_metrics.py` exactly, replacing every `0008` with `0009`, `corpus-and-metrics` with `formal-core`, and the `ACCEPTANCE` path:

```bash
sed -e 's/0008/0009/g' -e 's/corpus-and-metrics/formal-core/g' \
  tests/spec/test_0008_corpus_and_metrics.py > tests/spec/test_0009_formal_core.py
grep -n "ACCEPTANCE =\|feature_id\|def test_" tests/spec/test_0009_formal_core.py
```
Expected: `ACCEPTANCE = "specs/features/0009-formal-core/acceptance.yaml"`, three `test_0009_*` functions.

- [ ] **Step 7: Register the feature** in `specs/README.md`, beneath the 0008 row:

```markdown
| [0009](./features/0009-formal-core/spec.md) | Formal adherence core | in progress |
```

- [ ] **Step 8: Run it red.**

Run: `.venv/bin/python -m pytest tests/spec/test_0009_formal_core.py -q`
Expected: `test_0009_acceptance_file_is_wellformed` passes; `test_0009_every_criterion_names_an_existing_file` and `test_0009_every_criterion_is_collectable` FAIL, naming `tests/unit/formal/...`.

- [ ] **Step 9: Commit.**

```bash
git add docs/adr/0002-formal-track.md specs/features/0009-formal-core tests/spec/test_0009_formal_core.py specs/README.md docs/superpowers/specs/2026-08-14-kernel-property-oracle-layer-design.md
scripts/git_commit_clean.sh -m "spec: add ADR 0002 and feature 0009, the formal adherence core" -m "The parent design ruled formal verification out. ADR 0002 narrows that to agent-written proofs about kernel code, and keeps equivalence checking of statically analyzable kernels with VOLTA, where it belongs." -m "Feature 0009 comes first because no proof an agent writes can be scored until the checks that decide whether it counts exist and have each caught the cheat they are for. The spec test is red until they do."
git branch --show-current
```

---

### Task 1: The harness-owned comparison

**Files:**
- Create: `src/autokernel_pbt/formal/__init__.py`
- Create: `src/autokernel_pbt/formal/comparisons.py`
- Create: `tests/unit/formal/__init__.py` (empty)
- Test: `tests/unit/formal/test_comparisons.py`

**Interfaces:**
- Consumes: `residual_ratio`, `within_threshold` (`props/tolerance.py`); `PRIMARY_INPUT` (`props/backends/base.py`).
- Produces: `Comparisons(dtype: np.dtype, n: int)` with `Comparisons.for_output(inputs: Mapping[str, np.ndarray], dtype) -> Comparisons` and `Comparisons.close(actual, expected) -> bool`.

- [ ] **Step 1: Write the failing test** — `tests/unit/formal/test_comparisons.py`:

```python
"""The comparison an agent's spec is read with belongs to the harness."""

from __future__ import annotations

import numpy as np

from autokernel_pbt.formal.comparisons import Comparisons


def _softmax32(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    z = np.multiply(x, np.float32(beta), dtype=np.float32)
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _definition(x: np.ndarray) -> np.ndarray:
    wide = np.asarray(x, dtype=np.float64)
    e = np.exp(wide - np.max(wide, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


X = np.random.default_rng(3).normal(size=(8, 8)).astype(np.float32)
FLOAT32_ROWS_OF_8 = Comparisons(dtype=np.dtype(np.float32), n=8)


def test_close_forgives_rounding_and_nothing_more():
    """A float32 softmax is close to the definition; one at twice the temperature is not."""
    assert FLOAT32_ROWS_OF_8.close(_softmax32(X), _definition(X))
    assert not FLOAT32_ROWS_OF_8.close(_softmax32(X, beta=2.0), _definition(X))


def test_close_broadcasts_a_scalar_and_nothing_else():
    """Row sums can be compared with 1.0; a (8, 1) column is not stretched over (8, 8)."""
    y = _softmax32(X)
    assert FLOAT32_ROWS_OF_8.close(np.sum(y, axis=-1), 1.0)
    assert not FLOAT32_ROWS_OF_8.close(y, y[:, :1])


def test_the_budget_is_the_outputs_dtype_not_the_specs_arithmetic():
    """Summing float32 outputs in float64 must not tighten the budget to float64's.

    The same float64 row sums are close to one under the float32 budget the harness binds,
    and far from it under a float64 budget — so the budget comes from the bound dtype, never
    from whatever the spec chose to compute in. And `for_output` binds the output's dtype
    and the input's last-axis length, which is what the reference arm normalizes by.
    """
    sums = np.sum(_softmax32(X).astype(np.float64), axis=-1)
    assert FLOAT32_ROWS_OF_8.close(sums, 1.0)
    assert not Comparisons(dtype=np.dtype(np.float64), n=8).close(sums, 1.0)
    assert Comparisons.for_output({"x": X}, np.float32) == FLOAT32_ROWS_OF_8
```

- [ ] **Step 2: Run it red.**

Run: `.venv/bin/python -m pytest tests/unit/formal/test_comparisons.py -q`
Expected: `ModuleNotFoundError: No module named 'autokernel_pbt.formal'`.

- [ ] **Step 3: Implement.** `src/autokernel_pbt/formal/__init__.py`:

```python
"""Agent-written formal proofs of kernel–spec adherence: the language-neutral core."""
```

`src/autokernel_pbt/formal/comparisons.py`:

```python
"""The comparison an agent's spec is evaluated with: the harness's, never the agent's.

A spec over the reals says ``y = softmax(x)``. Its float reading cannot say that and mean it,
because rounding makes exact equality false for every correct kernel, so something has to decide
how close is close enough. If the agent decided, it could widen the margin until its spec accepted
anything — the formal twin of a tolerance tuned until the test passes. So the harness binds the
budget for each case, in the reference arm's own units, and the spec only ever asks whether two
things are close.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from autokernel_pbt.props.backends.base import PRIMARY_INPUT
from autokernel_pbt.props.tolerance import residual_ratio, within_threshold


@dataclass(frozen=True)
class Comparisons:
    """Closeness for one case, budgeted by the harness.

    ``dtype`` is the kernel output's dtype — the precision whose rounding is forgiven — and
    deliberately not the dtype of whatever the spec computes with: a spec that sums float32
    outputs in float64 would otherwise be held to float64's budget and reject every correct
    kernel. ``n`` is the accumulation length, read from the input as the reference arm reads it.
    """

    dtype: np.dtype
    n: int

    @classmethod
    def for_output(cls, inputs: Mapping[str, np.ndarray], dtype: np.dtype | type) -> Comparisons:
        x = inputs[PRIMARY_INPUT]
        n = x.shape[-1] if x.ndim and x.shape[-1] >= 1 else 1
        return cls(dtype=np.dtype(dtype), n=n)

    def close(self, actual: np.ndarray, expected: np.ndarray | float) -> bool:
        """Whether ``actual`` is within the rounding budget of ``expected``.

        A scalar ``expected`` is broadcast, so a spec can ask whether row sums are close to one.
        Nothing else is: broadcasting a (2, 1) array over (2, 3) is how ``np.allclose`` hides a
        wrong-shaped output, and a shape mismatch here is simply not close.
        """
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        if expected.ndim == 0:
            expected = np.broadcast_to(expected, actual.shape)
        return within_threshold(residual_ratio(actual, expected, dtype=self.dtype, n=self.n))
```

- [ ] **Step 4: Green, then saboteur-check.**

Run: `.venv/bin/python -m pytest tests/unit/formal/test_comparisons.py -q` — expect 3 passed. Then, one at a time, confirm the test dies: (a) change `dtype=self.dtype` to `dtype=actual.dtype` — only `test_the_budget_...` fails; (b) delete the `if expected.ndim == 0` branch — only `test_close_broadcasts_...` fails; (c) replace `n=self.n` with `n=1` — at least one of the three fails. Restore after each.

- [ ] **Step 5: Commit.**

```bash
git add src/autokernel_pbt/formal/__init__.py src/autokernel_pbt/formal/comparisons.py tests/unit/formal/__init__.py tests/unit/formal/test_comparisons.py
scripts/git_commit_clean.sh -m "feat: add the harness-owned comparison for formal specs" -m "A spec's float reading needs a tolerance, and an agent allowed to choose one can widen it until its spec accepts anything. The harness binds the budget per case instead, at the kernel output's dtype and the input's accumulation length, in the reference arm's units." -m "The dtype is the output's, not whatever the spec computes in: summing float32 outputs in float64 would otherwise hold a correct kernel to float64's rounding."
git branch --show-current
```

---

### Task 2: The attempt, and the ground it is judged against

**Files:**
- Create: `src/autokernel_pbt/formal/attempt.py`
- Create: `src/autokernel_pbt/formal/ground.py`
- Test: `tests/unit/formal/test_ground.py`

**Interfaces:**
- Consumes: `Comparisons` (Task 1); `Generator`, `NumpyBackend`, `kernel_inputs`, `OUTPUT_NAME`, `Status`, `Task`, `SOFTMAX`, `softmax_reference`.
- Produces:
  - `Readings(spec, model, hypotheses)`; `spec(inputs, output, cmp) -> bool`, `model(inputs) -> ndarray`, `hypotheses(inputs) -> bool`
  - `Axiom(name: str, mentions: frozenset[str], holds_on: Callable[[np.random.Generator], bool] | None)`
  - `ProofCheck(accepted: bool, escape_hatches: tuple[str, ...], statement_has_shape: bool, derives_false: bool)`
  - `Attempt(language: str, kernel_id: str, readings: Readings | None, axioms: tuple[Axiom, ...], proof: ProofCheck)`
  - `Ground(task, reference, kernel, hidden: Mapping[str, Kernel], primitives: frozenset[str], seed: int = 42)`
  - `CaseRecord(case_id, inputs, reference, kernel: ndarray | None, hidden: Mapping[str, ndarray])`
  - `realize(ground) -> tuple[CaseRecord, ...]`

- [ ] **Step 1: Write the failing test** — `tests/unit/formal/test_ground.py`:

```python
"""The ground: realized once, over the whole ladder, and refused if it cannot judge."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from autokernel_pbt.formal.ground import Ground, realize
from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference


def correct(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_2x(x: np.ndarray) -> np.ndarray:
    return correct(np.multiply(x, np.float32(2.0), dtype=np.float32))


def uniform_fill(x: np.ndarray) -> np.ndarray:
    return np.full(x.shape, 1.0 / x.shape[-1], dtype=x.dtype)


def crashes(x: np.ndarray) -> np.ndarray:
    msg = "a hidden kernel that cannot run"
    raise RuntimeError(msg)


def crashes_on_3x7(x: np.ndarray) -> np.ndarray:
    if x.shape == (3, 7):
        msg = "the kernel under proof fails on one rung"
        raise RuntimeError(msg)
    return correct(x)


HIDDEN = {"softmax_2x": softmax_2x, "uniform_fill": uniform_fill}
GROUND = Ground(
    task=SOFTMAX,
    reference=softmax_reference,
    kernel=correct,
    hidden=HIDDEN,
    primitives=frozenset({"exp"}),
)


def test_realize_records_the_whole_ladder_with_read_only_inputs():
    """18 cases: nine rungs, each a base case and one shift_rows partner.

    Every record carries the reference, the kernel's output and every hidden kernel's, and
    inputs no reading can write into — so one misbehaving reading cannot change what every
    later check sees.
    """
    records = realize(GROUND)
    assert len(records) == 18
    assert all(set(r.hidden) == set(HIDDEN) for r in records)
    assert all(r.kernel is not None for r in records)
    assert not any(r.inputs["x"].flags.writeable for r in records)


def test_a_kernel_under_proof_that_fails_is_recorded_not_refused():
    """The kernel under proof failing is data: those cases record no output, and the
    rest are kept. Only the model-fidelity check, in Task 3, can act on it."""
    records = realize(replace(GROUND, kernel=crashes_on_3x7))
    failed = [r.case_id for r in records if r.kernel is None]
    assert len(failed) == 2
    assert all(r.kernel is not None for r in records if r.case_id not in failed)


#: Each defect paired with the message only its own guard produces.
GROUND_DEFECTS = {
    "no_hidden_kernels": (lambda: replace(GROUND, hidden={}), r"no hidden wrong kernels"),
    "a_hidden_kernel_never_wrong": (
        lambda: replace(GROUND, hidden={**HIDDEN, "correct_twin": correct}),
        r"never wrong by more than the rounding budget",
    ),
    "a_hidden_kernel_that_crashes": (
        lambda: replace(GROUND, hidden={**HIDDEN, "crashes": crashes}),
        r"hidden kernel 'crashes' failed on case",
    ),
    "a_non_finite_reference": (
        lambda: replace(GROUND, reference=lambda x: np.full_like(x, np.nan)),
        r"reference is non-finite",
    ),
}


@pytest.mark.parametrize("name", sorted(GROUND_DEFECTS))
def test_a_ground_that_cannot_judge_is_refused(name: str):
    factory, expected = GROUND_DEFECTS[name]
    with pytest.raises(ValueError, match=expected):
        realize(factory())
```

- [ ] **Step 2: Run it red.**

Run: `.venv/bin/python -m pytest tests/unit/formal/test_ground.py -q`
Expected: `ModuleNotFoundError: No module named 'autokernel_pbt.formal.ground'`.

- [ ] **Step 3: Implement `src/autokernel_pbt/formal/attempt.py`.**

```python
"""An attempt, as the harness sees it: what a language adapter extracted from the agent's work.

The agent writes a spec, a model of the kernel, hypotheses, a theorem and a proof in some formal
language, and none of it is Python. An adapter runs the language's checker and supplies the
*executable readings* — the same definitions, evaluated on concrete inputs — through which the
language-neutral checks test everything the proof takes for granted. An ``Attempt`` is everything
those checks may read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from autokernel_pbt.formal.comparisons import Comparisons

Inputs = Mapping[str, np.ndarray]

#: Does this output meet the spec for these inputs? Equality goes through the harness's ``cmp``.
SpecReading = Callable[[Inputs, np.ndarray, Comparisons], bool]
#: What the agent says the kernel computes.
ModelReading = Callable[[Inputs], np.ndarray]
#: Does the theorem claim anything about these inputs?
HypothesesReading = Callable[[Inputs], bool]


@dataclass(frozen=True)
class Readings:
    """The executable reading of an attempt's spec, model and hypotheses.

    They must come from the same definitions the proof is about — never a twin the agent wrote
    separately, which would let it test one thing and prove another. Guaranteeing that is the
    adapter's obligation (feature 0010); here a reading is only its shape.
    """

    spec: SpecReading
    model: ModelReading
    hypotheses: HypothesesReading


@dataclass(frozen=True)
class Axiom:
    """An assumption the proof rests on beyond the language's base library.

    ``mentions`` is every identifier the axiom's statement refers to, as the adapter parsed it.
    ``holds_on`` draws a sample, evaluates the axiom under its intended reading, and says whether
    it held. ``None`` means the axiom has no reading, and an axiom nobody can test fails closed.
    """

    name: str
    mentions: frozenset[str]
    holds_on: Callable[[np.random.Generator], bool] | None


@dataclass(frozen=True)
class ProofCheck:
    """What the language's own tooling found, as the adapter reports it.

    ``statement_has_shape`` is the one thing the harness fixes about the theorem: for all inputs,
    hypotheses imply spec(inputs, model(inputs)). Without it the checks could not find the spec,
    the model and the hypotheses inside the statement.
    """

    accepted: bool
    escape_hatches: tuple[str, ...]
    statement_has_shape: bool
    derives_false: bool


@dataclass(frozen=True)
class Attempt:
    """One agent attempt at one kernel in one language. No readings fails closed."""

    language: str
    kernel_id: str
    readings: Readings | None
    axioms: tuple[Axiom, ...]
    proof: ProofCheck
```

- [ ] **Step 4: Implement `src/autokernel_pbt/formal/ground.py`.**

```python
"""What an attempt is judged against: the cases, the reference, the kernel's own executions, and
wrong kernels the agent never sees.

Realized once, before any check runs, and refused if it cannot judge. A ground with no wrong
kernels would let spec completeness pass vacuously, and a wrong kernel never wrong by more than
the rounding budget would ask a spec to reject what the harness itself cannot tell from the
reference. Both are bad *calls* — cheap to fix and silent if accepted — so both raise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from autokernel_pbt.formal.comparisons import Comparisons
from autokernel_pbt.props.backends.base import OUTPUT_NAME, Status, kernel_inputs
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.tasks import Task

Kernel = Callable[..., np.ndarray]


@dataclass(frozen=True)
class Ground:
    """The judge's side of one (task, kernel) pair. Never shown to the agent.

    ``primitives`` are the functions an axiom may be about: those the informal spec itself uses,
    such as ``exp`` for softmax.
    """

    task: Task
    reference: Kernel
    kernel: Kernel
    hidden: Mapping[str, Kernel]
    primitives: frozenset[str]
    seed: int = 42


@dataclass(frozen=True)
class CaseRecord:
    """One generated case, with every output a check may compare against.

    ``inputs`` are read-only copies held in a read-only mapping, so a reading that writes into an
    array, or rebinds a name, raises instead of silently changing what every later check sees.
    ``kernel`` is ``None`` where the kernel under proof failed to run: there is nothing there for
    a model to be faithful to.
    """

    case_id: str
    inputs: Mapping[str, np.ndarray]
    reference: np.ndarray
    kernel: np.ndarray | None
    hidden: Mapping[str, np.ndarray]


def realize(ground: Ground) -> tuple[CaseRecord, ...]:
    """Run the reference, the kernel and every hidden kernel over the whole ladder.

    The whole ladder, always: fewer groups than shapes leaves rungs unexercised, which the
    generator warns about, and the degenerate rungs are exactly where wrong kernels look right.
    """
    if not ground.hidden:
        msg = (
            "no hidden wrong kernels: with nothing to reject, spec completeness would pass "
            "vacuously"
        )
        raise ValueError(msg)
    backend = NumpyBackend()
    groups = Generator(ground.task.domain, ground.seed).generate(len(ground.task.domain.shapes))
    never_wrong = set(ground.hidden)
    records = []
    for group in groups:
        for case in group.cases:
            inputs = MappingProxyType(_read_only_copies(kernel_inputs(case)))
            reference = _reference(ground.reference, inputs, case)
            hidden = {}
            for name, wrong in ground.hidden.items():
                output = _run(backend, wrong, case)
                if output is None:
                    msg = (
                        f"hidden kernel {name!r} failed on case {case.case_id!r}; the ground "
                        f"cannot use it"
                    )
                    raise ValueError(msg)
                if not Comparisons.for_output(inputs, output.dtype).close(output, reference):
                    never_wrong.discard(name)
                hidden[name] = output
            kernel = _run(backend, ground.kernel, case)
            records.append(CaseRecord(case.case_id, inputs, reference, kernel, hidden))
    if never_wrong:
        msg = (
            f"hidden kernel(s) {sorted(never_wrong)} are never wrong by more than the rounding "
            f"budget; no spec can be asked to reject what the harness cannot tell from the "
            f"reference"
        )
        raise ValueError(msg)
    return tuple(records)


def _read_only_copies(inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    copies = {}
    for name, array in inputs.items():
        copy = np.array(array, copy=True)
        copy.flags.writeable = False
        copies[name] = copy
    return copies


def _reference(reference: Kernel, inputs: Mapping[str, np.ndarray], case: Case) -> np.ndarray:
    # Suppressed, then checked, as the reference arm does: under filterwarnings=error an overflow
    # would abort the run, and in production it would pass silently as inf.
    with np.errstate(all="ignore"):
        expected = np.asarray(reference(**inputs))
    if expected.dtype.kind in "fc" and not np.all(np.isfinite(expected)):
        msg = f"the reference is non-finite on case {case.case_id!r}; it cannot anchor any check"
        raise ValueError(msg)
    return expected


def _run(backend: NumpyBackend, kernel: Kernel, case: Case) -> np.ndarray | None:
    result = backend.run(kernel, case)
    if result.status != Status.OK or OUTPUT_NAME not in result.outputs:
        return None
    return result.outputs[OUTPUT_NAME]
```

- [ ] **Step 5: Green, then saboteur-check.**

Run: `.venv/bin/python -m pytest tests/unit/formal/test_ground.py -q` — expect 6 passed. Then confirm each guard is the sole catcher of its own case: delete (a) the empty-hidden guard — only `[no_hidden_kernels]` fails; (b) the `never_wrong` raise — only `[a_hidden_kernel_never_wrong]`; (c) the `output is None` raise — only `[a_hidden_kernel_that_crashes]`; (d) the finiteness check in `_reference` — only `[a_non_finite_reference]`; (e) `copy.flags.writeable = False` — only `test_realize_records_the_whole_ladder_...`. The `MappingProxyType` wrapper is pinned in Task 3 by `model_rebinds_an_input`. Restore after each.

- [ ] **Step 6: Commit.**

```bash
git add src/autokernel_pbt/formal/attempt.py src/autokernel_pbt/formal/ground.py tests/unit/formal/test_ground.py
scripts/git_commit_clean.sh -m "feat: add the formal attempt and the ground it is judged against" -m "An attempt is what a language adapter will hand the harness: executable readings of the agent's spec, model and hypotheses, its axioms, and what the language's checker found. The ground is realized once, over the whole ladder, through the existing generator and backend." -m "A ground that cannot judge is refused rather than allowed to pass something vacuously: no hidden kernels, a hidden kernel never wrong beyond the rounding budget, one that cannot run, a non-finite reference. Inputs are read-only copies, so one misbehaving reading cannot change what every later check sees."
git branch --show-current
```

---

### Task 3: The six checks and the certificate

**Files:**
- Create: `src/autokernel_pbt/formal/checks.py`
- Test: `tests/unit/formal/test_checks.py`

**Interfaces:**
- Consumes: everything Tasks 1 and 2 produce.
- Produces: `Check` (six members), `Finding(check, passed, detail)`, `Certificate(language, kernel_id, findings)` with `.failed -> frozenset[Check]` and `.certified -> bool`, `certify(attempt, ground) -> Certificate`, `AXIOM_SAMPLES = 64`.

- [ ] **Step 1: Write the failing test** — `tests/unit/formal/test_checks.py`:

```python
"""The six checks, each shown to catch exactly the cheat it exists for.

Each cheat is an attempt an agent rewarded for "proof accepted" could produce without proving
adherence. It differs from the honest attempt in one respect and must be rejected by exactly one
check, with that check's own message — the repo's unique-catcher standard, applied to proofs. A
cheat caught by two checks would leave one of them unproven.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from autokernel_pbt.formal.attempt import Attempt, Axiom, ProofCheck, Readings
from autokernel_pbt.formal.checks import Certificate, Check, Finding, certify
from autokernel_pbt.formal.ground import Ground
from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference

#: The tile a single-block kernel covers. The ladder's (7, 129) rung is wider.
TILE = 64


def definition(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """The agent's spec, read in float64: softmax(beta * x)."""
    z = beta * np.asarray(x, dtype=np.float64)
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_beta(beta: float):
    b = np.float32(beta)

    def kernel(x: np.ndarray) -> np.ndarray:
        z = np.multiply(x, b, dtype=np.float32)
        e = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    return kernel


correct = softmax_beta(1.0)


def uniform_fill(x: np.ndarray) -> np.ndarray:
    return np.full(x.shape, 1.0 / x.shape[-1], dtype=x.dtype)


def rows_rolled(x: np.ndarray) -> np.ndarray:
    return np.roll(correct(x), 1, axis=0)


def columns_reversed(x: np.ndarray) -> np.ndarray:
    return correct(x)[:, ::-1].copy()


def tail_dropping(x: np.ndarray) -> np.ndarray:
    """The softmax of the first TILE columns, zeros after: a single-block kernel with BLOCK < n."""
    y = np.zeros_like(x)
    y[:, :TILE] = correct(x[:, :TILE])
    return y


def crashes_on_3x7(x: np.ndarray) -> np.ndarray:
    if x.shape == (3, 7):
        msg = "the kernel under proof fails on one rung"
        raise RuntimeError(msg)
    return correct(x)


HIDDEN = {
    "softmax_2x": softmax_beta(2.0),
    "uniform_fill": uniform_fill,
    "rows_rolled": rows_rolled,
    "columns_reversed": columns_reversed,
}


def ground(kernel=correct) -> Ground:
    return Ground(
        task=SOFTMAX,
        reference=softmax_reference,
        kernel=kernel,
        hidden=HIDDEN,
        primitives=frozenset({"exp"}),
    )


def spec_of(beta: float = 1.0):
    def spec(inputs, y, cmp):
        return cmp.close(y, definition(inputs["x"], beta))

    return spec


def spec_of_the_gates_case_laws(inputs, y, cmp):
    """Finite, in [0, 1], rows sum to one: true of softmax, and of every temperature."""
    in_range = bool(np.all(np.isfinite(y)) and np.all((y >= 0) & (y <= 1)))
    return in_range and cmp.close(np.sum(y, axis=-1), 1.0)


def model_of(kernel):
    return lambda inputs: kernel(inputs["x"])


def two_dimensional(inputs):
    return inputs["x"].ndim == 2


def fits_one_tile(inputs):
    return inputs["x"].shape[-1] <= TILE


def exp_adds(rng: np.random.Generator) -> bool:
    a, b = rng.normal(size=2)
    return bool(np.isclose(np.exp(a + b), np.exp(a) * np.exp(b), rtol=1e-12))


def exp_is_linear(rng: np.random.Generator) -> bool:
    t = rng.normal()
    return bool(np.isclose(np.exp(t), 1.0 + t, rtol=1e-9))


SOUND_PROOF = ProofCheck(
    accepted=True, escape_hatches=(), statement_has_shape=True, derives_false=False
)
HONEST = Attempt(
    language="fake",
    kernel_id="softmax_correct",
    readings=Readings(spec=spec_of(), model=model_of(correct), hypotheses=two_dimensional),
    axioms=(Axiom("exp_add", frozenset({"exp"}), exp_adds),),
    proof=SOUND_PROOF,
)


def honest(**changes) -> Attempt:
    return replace(HONEST, **changes)


def failing_details(certificate: Certificate) -> dict[Check, str]:
    return {f.check: f.detail for f in certificate.findings if not f.passed}


def test_an_honest_attempt_is_certified():
    certificate = certify(HONEST, ground())
    assert failing_details(certificate) == {}
    assert certificate.certified


def test_a_record_missing_a_check_is_not_a_certificate():
    """Six findings must be present, not merely none failing."""
    five = tuple(Finding(check, True, "") for check in Check if check is not Check.AXIOMS)
    assert not Certificate("fake", "softmax_correct", five).certified


#: Each cheat, the one check that must reject it, and that check's own message.
CHEATS = {
    "checker_rejects_the_proof": (
        lambda: (honest(proof=replace(SOUND_PROOF, accepted=False)), ground()),
        Check.PROOF,
        r"rejected the proof",
    ),
    "proof_ends_in_sorry": (
        lambda: (honest(proof=replace(SOUND_PROOF, escape_hatches=("sorry",))), ground()),
        Check.PROOF,
        r"escape hatch",
    ),
    "spec_fitted_to_a_wrong_kernel": (
        lambda: (
            honest(
                kernel_id="softmax_3x",
                readings=Readings(spec_of(3.0), model_of(softmax_beta(3.0)), two_dimensional),
            ),
            ground(softmax_beta(3.0)),
        ),
        Check.SPEC_SOUNDNESS,
        r"rejects the reference",
    ),
    "spec_made_of_the_gates_case_laws": (
        lambda: (
            honest(readings=Readings(spec_of_the_gates_case_laws, model_of(correct), two_dimensional)),
            ground(),
        ),
        Check.SPEC_COMPLETENESS,
        r"accepts wrong output",
    ),
    "model_idealizes_a_tail_dropping_kernel": (
        lambda: (
            honest(
                kernel_id="softmax_tail_drop",
                readings=Readings(spec_of(), model_of(correct), two_dimensional),
            ),
            ground(tail_dropping),
        ),
        Check.MODEL_FIDELITY,
        r"diverges from the kernel",
    ),
    "hypotheses_exclude_the_failing_cases": (
        lambda: (
            honest(
                kernel_id="softmax_tail_drop",
                readings=Readings(spec_of(), model_of(tail_dropping), fits_one_tile),
            ),
            ground(tail_dropping),
        ),
        Check.NON_VACUITY,
        r"hypotheses exclude",
    ),
    "theorem_of_the_wrong_shape": (
        lambda: (honest(proof=replace(SOUND_PROOF, statement_has_shape=False)), ground()),
        Check.NON_VACUITY,
        r"required shape",
    ),
    "axiom_about_the_model": (
        lambda: (
            honest(axioms=(Axiom("model_is_softmax", frozenset({"exp", "model"}), exp_adds),)),
            ground(),
        ),
        Check.AXIOMS,
        r"beyond the declared primitives",
    ),
    "axiom_that_is_false": (
        lambda: (honest(axioms=(Axiom("exp_is_linear", frozenset({"exp"}), exp_is_linear),)), ground()),
        Check.AXIOMS,
        r"false on a sample",
    ),
    "axiom_nobody_can_test": (
        lambda: (honest(axioms=(Axiom("exp_add", frozenset({"exp"}), None),)), ground()),
        Check.AXIOMS,
        r"has no executable reading",
    ),
    "axioms_that_derive_false": (
        lambda: (honest(proof=replace(SOUND_PROOF, derives_false=True)), ground()),
        Check.AXIOMS,
        r"False is derivable",
    ),
}


@pytest.mark.parametrize("name", sorted(CHEATS))
def test_each_cheat_is_rejected_by_exactly_its_own_check(name: str):
    factory, check, message = CHEATS[name]
    attempt, judged_against = factory()
    details = failing_details(certify(attempt, judged_against))
    assert set(details) == {check}, details
    assert re.search(message, details[check]), details[check]


def test_an_attempt_without_executable_readings_is_not_certified():
    details = failing_details(certify(honest(readings=None), ground()))
    reading_checks = {
        Check.SPEC_SOUNDNESS,
        Check.SPEC_COMPLETENESS,
        Check.MODEL_FIDELITY,
        Check.NON_VACUITY,
    }
    assert set(details) == reading_checks, details
    assert all("no executable reading" in details[c] for c in reading_checks), details


def raising_model(inputs):
    msg = "the agent's model crashed"
    raise RuntimeError(msg)


def writing_model(inputs):
    inputs["x"][...] = 0.0
    return correct(inputs["x"])


def rebinding_model(inputs):
    inputs["x"] = np.zeros_like(inputs["x"])
    return correct(inputs["x"])


def raising_hypotheses(inputs):
    return inputs["missing"].ndim == 2


def raising_axiom(rng):
    return 1 / 0 == 0


#: Agent code misbehaving, the checks that must fail, and the message proving why.
MISBEHAVIOR = {
    "model_raises": (
        Readings(spec_of(), raising_model, two_dimensional),
        (HONEST.axioms[0],),
        {Check.MODEL_FIDELITY},
        r"raised RuntimeError",
    ),
    "model_writes_into_its_inputs": (
        Readings(spec_of(), writing_model, two_dimensional),
        (HONEST.axioms[0],),
        {Check.MODEL_FIDELITY},
        r"read-only",
    ),
    "model_rebinds_an_input": (
        Readings(spec_of(), rebinding_model, two_dimensional),
        (HONEST.axioms[0],),
        {Check.MODEL_FIDELITY},
        r"does not support item assignment",
    ),
    "model_returns_the_wrong_shape": (
        Readings(spec_of(), lambda inputs: correct(inputs["x"]).T.copy(), two_dimensional),
        (HONEST.axioms[0],),
        {Check.MODEL_FIDELITY},
        r"diverges from the kernel",
    ),
    "spec_returns_an_array": (
        Readings(lambda inputs, y, cmp: np.ones(3, dtype=bool), model_of(correct), two_dimensional),
        (HONEST.axioms[0],),
        {Check.SPEC_SOUNDNESS, Check.SPEC_COMPLETENESS},
        r"not a bool",
    ),
    "hypotheses_raise": (
        Readings(spec_of(), model_of(correct), raising_hypotheses),
        (HONEST.axioms[0],),
        {Check.NON_VACUITY},
        r"raised KeyError",
    ),
    "axiom_reading_raises": (
        HONEST.readings,
        (Axiom("exp_add", frozenset({"exp"}), raising_axiom),),
        {Check.AXIOMS},
        r"raised ZeroDivisionError",
    ),
}


@pytest.mark.parametrize("name", sorted(MISBEHAVIOR))
def test_misbehaving_agent_code_fails_only_its_own_check(name: str):
    readings, axioms, checks, message = MISBEHAVIOR[name]
    details = failing_details(certify(honest(readings=readings, axioms=axioms), ground()))
    assert set(details) == checks, details
    assert all(re.search(message, details[c]) for c in checks), details


def test_a_kernel_that_fails_on_a_case_cannot_be_certified():
    details = failing_details(certify(honest(), ground(crashes_on_3x7)))
    assert set(details) == {Check.MODEL_FIDELITY}, details
    assert "failed on" in details[Check.MODEL_FIDELITY]
```

- [ ] **Step 2: Run it red.**

Run: `.venv/bin/python -m pytest tests/unit/formal/test_checks.py -q`
Expected: `ModuleNotFoundError: No module named 'autokernel_pbt.formal.checks'`.

- [ ] **Step 3: Implement `src/autokernel_pbt/formal/checks.py`.**

```python
"""The six checks a certificate must pass, and the certificate record.

A proof the language's checker accepts is necessary and nowhere near sufficient, because the agent
wrote everything the proof is about. Each check below closes one way of getting a proof accepted
without proving adherence, and each is paired, in the tests, with an attempt that cheats in exactly
that way and must be rejected by that check alone.

Agent-supplied readings are data, not harness code: one that raises, returns something other than
a boolean, or writes into its inputs fails its own check and never aborts the others.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from autokernel_pbt.formal.attempt import Attempt, ProofCheck, Readings
from autokernel_pbt.formal.comparisons import Comparisons
from autokernel_pbt.formal.ground import CaseRecord, Ground, realize

#: Samples each axiom's reading is checked on. An axiom false only on a set of measure zero slips
#: through; one false on any positive fraction of its domain is caught with overwhelming odds.
AXIOM_SAMPLES = 64

_NO_READING = "no executable reading, so it fails closed"


class Check(str, Enum):
    """The six checks, in the order they run."""

    PROOF = "proof"
    SPEC_SOUNDNESS = "spec_soundness"
    SPEC_COMPLETENESS = "spec_completeness"
    MODEL_FIDELITY = "model_fidelity"
    NON_VACUITY = "non_vacuity"
    AXIOMS = "axioms"

    # As Verdict and Status do: format() of a str-mixin Enum gives the value on 3.10/3.11 and the
    # name on 3.12+, against a declared requires-python >= 3.10.
    __str__ = str.__str__


@dataclass(frozen=True)
class Finding:
    check: Check
    passed: bool
    detail: str


@dataclass(frozen=True)
class Certificate:
    """What the six checks found for one attempt."""

    language: str
    kernel_id: str
    findings: tuple[Finding, ...]

    @property
    def failed(self) -> frozenset[Check]:
        return frozenset(f.check for f in self.findings if not f.passed)

    @property
    def certified(self) -> bool:
        # All six must have run, not merely none failed: a record missing a check is not a
        # certificate that passed it.
        return {f.check for f in self.findings} == set(Check) and not self.failed


def certify(attempt: Attempt, ground: Ground) -> Certificate:
    """Judge one attempt against its ground. Raises only if the ground cannot judge."""
    cases = realize(ground)
    findings = (
        _proof(attempt.proof),
        _spec_soundness(attempt.readings, cases),
        _spec_completeness(attempt.readings, cases),
        _model_fidelity(attempt.readings, cases),
        _non_vacuity(attempt, cases),
        _axioms(attempt, ground.primitives),
    )
    return Certificate(attempt.language, attempt.kernel_id, findings)


def _holds(reading: Callable[..., object], *args: object) -> tuple[bool | None, str]:
    """Run a reading that should answer yes or no. ``None`` means it did neither."""
    try:
        answer = reading(*args)
    except Exception as exc:  # noqa: BLE001 - agent code failing is a finding, not an abort
        return None, f"raised {type(exc).__name__}: {exc}"
    if not isinstance(answer, (bool, np.bool_)):
        return None, f"returned {type(answer).__name__}, not a bool"
    return bool(answer), ""


def _proof(proof: ProofCheck) -> Finding:
    if not proof.accepted:
        return Finding(Check.PROOF, False, "the language's checker rejected the proof")
    if proof.escape_hatches:
        hatches = sorted(proof.escape_hatches)
        return Finding(Check.PROOF, False, f"escape hatch(es) in the proof: {hatches}")
    return Finding(Check.PROOF, True, "accepted by the checker, with no escape hatch")


def _spec_soundness(readings: Readings | None, cases: tuple[CaseRecord, ...]) -> Finding:
    if readings is None:
        return Finding(Check.SPEC_SOUNDNESS, False, f"the spec has {_NO_READING}")
    rejected = []
    for case in cases:
        cmp = Comparisons.for_output(case.inputs, case.reference.dtype)
        holds, why = _holds(readings.spec, case.inputs, case.reference, cmp)
        if holds is None:
            return Finding(Check.SPEC_SOUNDNESS, False, f"spec reading {why} on {case.case_id}")
        if not holds:
            rejected.append(case.case_id)
    if rejected:
        detail = (
            f"spec rejects the reference on {len(rejected)} of {len(cases)} cases, "
            f"first {rejected[0]}"
        )
        return Finding(Check.SPEC_SOUNDNESS, False, detail)
    detail = f"spec accepts the reference on all {len(cases)} cases"
    return Finding(Check.SPEC_SOUNDNESS, True, detail)


def _spec_completeness(readings: Readings | None, cases: tuple[CaseRecord, ...]) -> Finding:
    if readings is None:
        return Finding(Check.SPEC_COMPLETENESS, False, f"the spec has {_NO_READING}")
    accepted: dict[str, list[str]] = {}
    judged = 0
    for case in cases:
        for name, output in case.hidden.items():
            cmp = Comparisons.for_output(case.inputs, output.dtype)
            if cmp.close(output, case.reference):
                # Right here — a one-column softmax is 1.0 at any temperature — so there is
                # nothing for the spec to reject on this case.
                continue
            judged += 1
            holds, why = _holds(readings.spec, case.inputs, output, cmp)
            if holds is None:
                detail = f"spec reading {why} on {case.case_id} ({name})"
                return Finding(Check.SPEC_COMPLETENESS, False, detail)
            if holds:
                accepted.setdefault(name, []).append(case.case_id)
    if accepted:
        which = ", ".join(f"{name} on {len(ids)}" for name, ids in sorted(accepted.items()))
        detail = f"spec accepts wrong output from {which} case(s)"
        return Finding(Check.SPEC_COMPLETENESS, False, detail)
    detail = f"spec rejects every hidden kernel on all {judged} cases where it is wrong"
    return Finding(Check.SPEC_COMPLETENESS, True, detail)


def _model_fidelity(readings: Readings | None, cases: tuple[CaseRecord, ...]) -> Finding:
    if readings is None:
        return Finding(Check.MODEL_FIDELITY, False, f"the model has {_NO_READING}")
    diverged = []
    for case in cases:
        if case.kernel is None:
            detail = (
                f"the kernel under proof failed on {case.case_id}; there is nothing to be "
                f"faithful to"
            )
            return Finding(Check.MODEL_FIDELITY, False, detail)
        try:
            output = np.asarray(readings.model(case.inputs))
            cmp = Comparisons.for_output(case.inputs, case.kernel.dtype)
            same = cmp.close(output, case.kernel)
        except Exception as exc:  # noqa: BLE001 - agent code failing is a finding, not an abort
            detail = f"model reading raised {type(exc).__name__}: {exc} on {case.case_id}"
            return Finding(Check.MODEL_FIDELITY, False, detail)
        if not same:
            diverged.append(case.case_id)
    if diverged:
        detail = (
            f"model diverges from the kernel on {len(diverged)} of {len(cases)} cases, "
            f"first {diverged[0]}"
        )
        return Finding(Check.MODEL_FIDELITY, False, detail)
    detail = f"model matches the kernel on all {len(cases)} cases"
    return Finding(Check.MODEL_FIDELITY, True, detail)


def _non_vacuity(attempt: Attempt, cases: tuple[CaseRecord, ...]) -> Finding:
    if not attempt.proof.statement_has_shape:
        detail = (
            "the theorem is not of the required shape: for all inputs, hypotheses imply "
            "spec(inputs, model(inputs))"
        )
        return Finding(Check.NON_VACUITY, False, detail)
    if attempt.readings is None:
        return Finding(Check.NON_VACUITY, False, f"the hypotheses have {_NO_READING}")
    excluded = []
    for case in cases:
        holds, why = _holds(attempt.readings.hypotheses, case.inputs)
        if holds is None:
            return Finding(Check.NON_VACUITY, False, f"hypotheses reading {why} on {case.case_id}")
        if not holds:
            excluded.append(case.case_id)
    if excluded:
        detail = (
            f"hypotheses exclude {len(excluded)} of {len(cases)} cases, first {excluded[0]}: "
            f"the theorem says nothing about them"
        )
        return Finding(Check.NON_VACUITY, False, detail)
    return Finding(Check.NON_VACUITY, True, f"hypotheses hold on all {len(cases)} cases")


def _axioms(attempt: Attempt, primitives: frozenset[str]) -> Finding:
    if attempt.proof.derives_false:
        detail = "False is derivable from the axioms within the bounded attempt"
        return Finding(Check.AXIOMS, False, detail)
    rng = np.random.default_rng(0)
    for axiom in attempt.axioms:
        beyond = axiom.mentions - primitives
        if beyond:
            detail = (
                f"axiom {axiom.name!r} mentions {sorted(beyond)}, beyond the declared "
                f"primitives {sorted(primitives)}"
            )
            return Finding(Check.AXIOMS, False, detail)
        if axiom.holds_on is None:
            return Finding(Check.AXIOMS, False, f"axiom {axiom.name!r} has {_NO_READING}")
        for _ in range(AXIOM_SAMPLES):
            holds, why = _holds(axiom.holds_on, rng)
            if holds is None:
                return Finding(Check.AXIOMS, False, f"axiom {axiom.name!r} reading {why}")
            if not holds:
                return Finding(Check.AXIOMS, False, f"axiom {axiom.name!r} is false on a sample")
    detail = (
        f"{len(attempt.axioms)} axiom(s), each about declared primitives only and true on "
        f"{AXIOM_SAMPLES} samples"
    )
    return Finding(Check.AXIOMS, True, detail)
```

- [ ] **Step 4: Green.**

Run: `.venv/bin/python -m pytest tests/unit/formal -q` — expect all passed (3 + 6 + 22).
Run: `.venv/bin/python -m pytest tests/spec/test_0009_formal_core.py -q` — expect 3 passed: the spec test is green.

- [ ] **Step 5: Saboteur-check the checks.** Neuter each check in turn — replace its function body with `return Finding(Check.<X>, True, "neutered")` — and confirm the tests that fail are exactly those naming that check: neutering `_proof` fails only the two `PROOF` cheats; `_spec_soundness`, its cheat plus `spec_returns_an_array` and the no-readings test; `_spec_completeness`, its cheat plus the same two; `_model_fidelity`, its cheat, four `MISBEHAVIOR` cases, the no-readings test and the failing-kernel test; `_non_vacuity`, its two cheats, `hypotheses_raise` and the no-readings test; `_axioms`, its four cheats and `axiom_reading_raises`. The honest attempt must stay certified in every case. Restore after each.

- [ ] **Step 6: Full verification.**

```bash
.venv/bin/python -m pytest -m "not gpu" -q
.venv/bin/ruff check src tests
.venv/bin/python -m pytest -m "not gpu" -q --cov=autokernel_pbt --cov-fail-under=95 2>&1 | tail -3
```
Expected: green and silent apart from the existing Triton skip; ruff clean; coverage at or above 95%.

- [ ] **Step 7: Commit, and mark the feature implemented.** In `specs/README.md` change the 0009 row's status to `implemented`, then:

```bash
git add src/autokernel_pbt/formal/checks.py tests/unit/formal/test_checks.py specs/README.md
scripts/git_commit_clean.sh -m "feat: add the six certificate checks, each the sole catcher of one cheat" -m "An accepted proof certifies nothing when the agent wrote the spec, the model and the theorem, so a certificate requires six checks: the proof itself, the spec against the reference and against wrong kernels the agent never sees, the model against the kernel's own executions, the hypotheses against the real inputs, and every axiom against its intended reading." -m "Each check is pinned by an attempt that cheats in exactly its way and is rejected by it alone, with its own message; neutering any check fails precisely its own cheats. Agent code that raises, answers with something other than a boolean or writes into its inputs fails its own check and never aborts the rest."
git branch --show-current
```

---

## Self-review against the spec

- §4 artifacts → `Attempt`, `Readings`, `Axiom`, `ProofCheck` (Task 2). Refutations are the pilot's (0011): no check consumes one yet, so none is built.
- §4 executable readings, fail closed → `readings: None` path in every reading check (Task 3); harness-owned comparisons (Task 1).
- §5 C1–C6 → `_proof` … `_axioms`, each with its sole-catcher cheats (Task 3); C2/C3 calibration is exercised by the honest attempt (passes both) and two cheats (fail one each).
- §6 hidden set → `Ground.hidden`, refused when empty or indistinguishable (Task 2). The study's real hidden set — `saboteurs.py` rows and the probe kernels — belongs to 0011.
- §10 → `src/autokernel_pbt/formal/`, CPU-only, main suite. Adapters (0010) and the agent driver (0011) are out of this plan by design.
