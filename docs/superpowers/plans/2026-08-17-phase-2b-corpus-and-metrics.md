# Phase 2b Corpus and Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An agent-authored, blinded, gated mutation corpus scored on both backends, and the three metrics that make the paper's headline claims measurable.

**Architecture:** Additive. The corpus is data plus a gate; the metrics are a pure function of the two Parquet tables `driver.read_run` already produces. Nothing in the oracle layer, the driver, or the storage changes.

**Tech Stack:** Python 3.10+, NumPy, Triton, PyArrow, pytest, ruff.

**Design doc:** `docs/superpowers/specs/2026-08-17-phase-2b-corpus-and-metrics-design.md` — read §3 (blinding) and §4 (the gate) before starting; they are what the numbers rest on.

---

## Scope note: there is a natural checkpoint

Tasks 0–8 deliver a complete, testable result on CPU: a gated corpus, the three metrics, and a
report. Tasks 9–10 add the Triton twins and the cross-backend comparison. The design chose both
backends together, and this plan does that — but if the corpus needs revision (it will, see the
gate), stopping at Task 8 leaves working software rather than a half-migrated one.

## File Structure

| File | Responsibility |
|------|---------------|
| `docs/protocol/mutant-authoring.md` | **The blinded prompt, verbatim.** This is the experimental method, not documentation — it must be version-controlled and quoted in the paper. |
| `src/autokernel_pbt/corpus/mutant.py` | `Mutant` — kernel id, intended fault class, taxonomy row, backend, callable. |
| `src/autokernel_pbt/corpus/gate.py` | The admission gate and its rejection reasons. |
| `src/autokernel_pbt/corpus/registry.py` | The corpus: mutants and correct-but-different kernels, per task. |
| `kernels/mutants/numpy_mutants.py` | The agent-authored NumPy mutants. |
| `kernels/mutants/triton_mutants.py` | Their Triton twins. |
| `kernels/mutants/correct_variants.py` | Correct-but-different implementations, for the FP denominator. |
| `src/autokernel_pbt/metrics/rates.py` | Detection rate, FP rate, cases-to-first-failure, from artifacts alone. |
| `src/autokernel_pbt/metrics/report.py` | The fault-class × arm × backend table. |
| `specs/features/0008-corpus-and-metrics/` | Spec and acceptance criteria (Task 0). |

**Repo conventions.** Read `CLAUDE.md`.

- **Never `git commit`.** Use `scripts/git_commit_clean.sh`, then **verify `git branch --show-current`** — a detached HEAD orphaned three commits in this repo once.
- `filterwarnings = ["error"]`; `ruff check src tests kernels scripts` must pass; coverage floor 95%.
- Every new assertion must be the **unique** catcher for at least one saboteur.
- Bad **data** → `INCONCLUSIVE`; bad **call** → raise.

---

### Task 0: Feature 0008 spec and acceptance criteria (red)

**Files:** create `specs/features/0008-corpus-and-metrics/{spec.md,acceptance.yaml}`,
`tests/spec/test_0008_corpus_and_metrics.py`; modify `specs/README.md`.

- [ ] **Step 1: Write the spec**, summarising the design doc's §1–§7 in the house format
      (Problem / Scope / Non-goals / Acceptance), with feature id `0008`.

- [ ] **Step 2: Write `acceptance.yaml`** with these criteria, all CPU-runnable:

```yaml
feature_id: "0008"
feature_name: corpus-and-metrics
version: 1

criteria:
  - id: A_CORRECT_MUTANT_IS_REJECTED
    description: a candidate that matches its reference everywhere is refused admission
    check:
      type: unit_test
      test: tests/unit/corpus/test_gate.py::test_a_candidate_that_is_not_broken_is_rejected

  - id: A_CATASTROPHIC_MUTANT_IS_REJECTED
    description: a candidate that is unjudgeable everywhere is refused admission
    check:
      type: unit_test
      test: tests/unit/corpus/test_gate.py::test_a_candidate_that_never_runs_is_rejected

  - id: A_KERNEL_WRONG_EVERYWHERE_IS_ADMITTED
    description: being wrong on every group is an ordinary bug, not grounds for rejection
    check:
      type: unit_test
      test: tests/unit/corpus/test_gate.py::test_a_candidate_wrong_on_every_group_is_admitted

  - id: REJECTIONS_CARRY_A_REASON
    description: a refused candidate records why, because the rejection rate is a finding
    check:
      type: unit_test
      test: tests/unit/corpus/test_gate.py::test_a_rejection_records_its_reason

  - id: DETECTION_IS_KEYED_BY_GROUP
    description: the rate counts case groups, not results
    check:
      type: unit_test
      test: tests/unit/metrics/test_rates.py::test_detection_rate_counts_groups_not_results

  - id: TOLERANCE_FREE_DETECTION_IS_SEPARATE
    description: the headline claim has its own numerator
    check:
      type: unit_test
      test: tests/unit/metrics/test_rates.py::test_tolerance_free_detection_has_its_own_numerator

  - id: METRICS_COME_FROM_ARTIFACTS_ALONE
    description: rates are computable from the two tables with no oracle in the loop
    check:
      type: unit_test
      test: tests/unit/metrics/test_rates.py::test_rates_are_computed_from_the_tables_alone

  - id: THE_REPORT_STATES_THE_DEFLATION
    description: the ladder's known understatement appears beside the numbers
    check:
      type: unit_test
      test: tests/unit/metrics/test_report.py::test_the_report_states_the_ladder_deflation

  - id: INTENDED_CLASS_IS_LABELLED_AS_INTENDED
    description: the fault class is recorded as intended-by-construction, never as verified
    check:
      type: unit_test
      test: tests/unit/corpus/test_mutant.py::test_fault_class_is_recorded_as_intended
```

- [ ] **Step 3: Write the spec test**, copying `tests/spec/test_0007_triton_backend.py` exactly
      and substituting `0008` and the new acceptance path. It already carries the duplicate-target
      and file-only-target checks.

- [ ] **Step 4: Register the feature** in `specs/README.md` beneath the 0007 row, status
      `in progress`.

- [ ] **Step 5: Run it red.**
      Run: `pytest tests/spec/test_0008_corpus_and_metrics.py -v`
      Expected: wellformed PASSES, the other two FAIL — no named test file exists yet.

- [ ] **Step 6: Commit.**

```bash
git add specs/features/0008-corpus-and-metrics tests/spec/test_0008_corpus_and_metrics.py specs/README.md
scripts/git_commit_clean.sh -m "spec: add feature 0008, the corpus and the metrics" -m "Every criterion is CPU-runnable. The gate criteria are stated as behaviour rather than as a checklist, because the gate is what decides whether a detection rate means anything."
git branch --show-current
```

---

### Task 1: The blinded authoring protocol

**This task is the experiment's method.** It produces no code. Written first because every mutant
authored before it exists is authored under unrecorded conditions and cannot be used.

**Files:** create `docs/protocol/mutant-authoring.md`.

- [ ] **Step 1: Write the protocol.**

```markdown
# Protocol: authoring a mutant

This file is the experimental method, not documentation. It is version-controlled so the paper
can quote it, and so a reader can judge what the authoring agent could and could not see.

## What the agent receives

Exactly two things:

1. The ISSTA subcategory description for the fault class, quoted verbatim from Table 2 of
   `reference/PBT-property-based-testing/papers/2605.19652.pdf`.
2. The correct reference implementation for the task, from `src/autokernel_pbt/props/tasks.py`.

## What the agent must NOT receive

- `kernels/tasks/<id>/acceptance.yaml`, or any property name
- `src/autokernel_pbt/props/properties.py`
- Any tolerance, threshold, or `residual_ratio` detail
- Any prior mutant, or any detection result

A mutant written with the property checklist in hand tells you only that the checklist matches
itself. This list is the difference between a measurement and a tautology.

## The prompt

> Here is a correct NumPy implementation of `<task>`:
>
> ```python
> <reference source>
> ```
>
> Here is a description of a class of real bug found in GPU tile programs:
>
> > <verbatim ISSTA subcategory description>
>
> Write a modified version of the implementation that exhibits that class of bug. Keep the same
> function signature. The result should look like plausible code someone might write, not an
> obviously broken stub. Return only the function.

## What is recorded per candidate

- the task, the subcategory, and the verbatim description given
- the returned source
- the model and date
- the gate verdict and, if rejected, the reason

## What is NOT claimed

The fault class is **intended**, established by what the prompt asked for. Nothing verifies that
the returned kernel exhibits that class rather than another. Automatic defect classification is a
research problem of its own, and a weak classifier would mislabel exactly the cases that matter.
Any per-class number carries this caveat.
```

- [ ] **Step 2: Commit.**

```bash
git add docs/protocol
scripts/git_commit_clean.sh -m "docs: record the blinded mutant-authoring protocol" -m "The experimental method, version-controlled so the paper can quote it and a reader can judge what the authoring agent could see. Written before any mutant exists, because a mutant authored under unrecorded conditions cannot be used." -m "The blinding list is the difference between a measurement and a tautology: a mutant written with the property checklist in hand tells you only that the checklist matches itself."
git branch --show-current
```

---

### Task 2: `Mutant`

**Files:** create `src/autokernel_pbt/corpus/__init__.py`, `src/autokernel_pbt/corpus/mutant.py`,
`tests/unit/corpus/__init__.py`, `tests/unit/corpus/test_mutant.py`.

- [ ] **Step 1: Write the failing test.**

```python
"""Mutant identity and labelling tests."""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.corpus.mutant import Mutant


def _fn(x):
    return x * 2.0


def _mutant(**overrides) -> Mutant:
    defaults = {
        "kernel_id": "softmax_dtype_semantics",
        "task_id": "softmax",
        "intended_class": "type_and_operator/data_type_semantics",
        "taxonomy_quote": "Loss of numeric meaning or precision from implicit casts",
        "backend": "numpy",
        "fn": _fn,
    }
    defaults.update(overrides)
    return Mutant(**defaults)


def test_fault_class_is_recorded_as_intended():
    """The criterion INTENDED_CLASS_IS_LABELLED_AS_INTENDED.

    Nothing verifies that a returned kernel exhibits the class the prompt asked for.
    The attribute is named `intended_class`, not `fault_class`, so a reader of the
    code cannot mistake a construction for a verification -- and any table built from
    it inherits the caveat by its own column name.
    """
    m = _mutant()
    assert m.intended_class == "type_and_operator/data_type_semantics"
    assert not hasattr(m, "fault_class")
    assert not hasattr(m, "verified_class")


def test_the_taxonomy_quote_is_carried_verbatim():
    # The paper's own words, so a reader can check the mutant against the class it
    # claims without re-deriving what the class meant.
    assert "implicit casts" in _mutant().taxonomy_quote


def test_a_mutant_without_a_taxonomy_quote_is_rejected():
    # A mutant with no provenance is untraceable to the corpus it claims to sample.
    with pytest.raises(ValueError, match="taxonomy_quote"):
        _mutant(taxonomy_quote="")


def test_backend_must_be_known():
    with pytest.raises(ValueError, match="unknown backend"):
        _mutant(backend="cuda_cpp")


def test_the_callable_is_reachable():
    assert _mutant().fn(np.ones(2)).tolist() == [2.0, 2.0]
```

- [ ] **Step 2: Run it red.** Expected: `ModuleNotFoundError: autokernel_pbt.corpus`.

- [ ] **Step 3: Implement.**

```python
"""One member of the mutation corpus.

The attribute is `intended_class`, never `fault_class`. Nothing in this project
verifies that an agent-authored kernel exhibits the class its prompt asked for --
automatic defect classification is a research problem of its own, and a weak
classifier would mislabel exactly the cases that matter. Naming the attribute for
what it actually is means a table built from it inherits the caveat by its own column
name, rather than depending on a footnote nobody reads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

KNOWN_BACKENDS = ("numpy", "triton")


@dataclass(frozen=True)
class Mutant:
    """A deliberately-broken kernel, traceable to the taxonomy row it samples."""

    kernel_id: str
    task_id: str
    #: The subcategory the authoring prompt asked for. INTENDED, not verified.
    intended_class: str
    #: The paper's own words for that subcategory, so a reader can check the mutant
    #: against the class it claims without re-deriving what the class meant.
    taxonomy_quote: str
    backend: str
    fn: Callable[..., Any] = field(compare=False)

    def __post_init__(self) -> None:
        if not self.taxonomy_quote.strip():
            msg = (
                f"mutant {self.kernel_id!r} carries no taxonomy_quote; a mutant with no "
                f"provenance is untraceable to the corpus it claims to sample"
            )
            raise ValueError(msg)
        if self.backend not in KNOWN_BACKENDS:
            msg = f"unknown backend {self.backend!r}; expected one of {KNOWN_BACKENDS}"
            raise ValueError(msg)
```

- [ ] **Step 4: Green, then saboteur-check.** Rename `intended_class` to `fault_class` and confirm
      exactly `test_fault_class_is_recorded_as_intended` fails. Restore.

- [ ] **Step 5: Commit.**

```bash
git add src/autokernel_pbt/corpus tests/unit/corpus
scripts/git_commit_clean.sh -m "feat: add the Mutant record" -m "The attribute is intended_class, never fault_class. Nothing verifies that an agent-authored kernel exhibits the class its prompt asked for, so naming it for what it is means any table built from it inherits the caveat by its own column name rather than depending on a footnote."
git branch --show-current
```

---

### Task 3: The validation gate

The part the numbers rest on. See design §4.

**Files:** create `src/autokernel_pbt/corpus/gate.py`, `tests/unit/corpus/test_gate.py`.

- [ ] **Step 1: Write the failing test.**

```python
"""Admission-gate tests.

An agent-authored mutant cannot be taken at its word, and each way it can be wrong
corrupts a different number invisibly. These tests pin the two admission criteria and
the recording of rejections.
"""

from __future__ import annotations

import numpy as np

from autokernel_pbt.corpus.gate import Rejection, admit
from autokernel_pbt.props.backends.base import OUTPUT_NAME, ExecutionResult, Status
from autokernel_pbt.props.case import Case


def _case(cid: str, gid: str) -> Case:
    return Case(
        case_id=cid, group_id=gid, relation="base", task_id="t",
        dtype="float32", shape=(2, 3),
        tensors={"x": np.ones((2, 3), dtype=np.float32)},
    )


def _row(cid, gid, y, status=Status.OK) -> ExecutionResult:
    outputs = {} if y is None else {OUTPUT_NAME: np.asarray(y, dtype=np.float32)}
    return ExecutionResult(case=_case(cid, gid), outputs=outputs, status=status)


def _ref(**kw):
    return np.ones((2, 3), dtype=np.float32)


def test_a_candidate_that_is_not_broken_is_rejected():
    """The criterion A_CORRECT_MUTANT_IS_REJECTED.

    A mutant that is secretly correct enters the detection denominator as a bug
    nobody can catch. Every arm's rate drops for free and the corpus looks harder
    than it is -- the most dangerous of the three failure modes, because nothing
    downstream looks wrong.
    """
    rows = [_row("c0", "g0", np.ones((2, 3))), _row("c1", "g1", np.ones((2, 3)))]
    verdict = admit(rows, reference_fn=_ref)
    assert isinstance(verdict, Rejection)
    assert "not broken" in verdict.reason


def test_a_candidate_that_never_runs_is_rejected():
    """The criterion A_CATASTROPHIC_MUTANT_IS_REJECTED."""
    rows = [
        _row("c0", "g0", None, status=Status.LAUNCH_ERROR),
        _row("c1", "g1", None, status=Status.COMPILE_ERROR),
    ]
    verdict = admit(rows, reference_fn=_ref)
    assert isinstance(verdict, Rejection)
    assert "judgeable" in verdict.reason


def test_a_candidate_wrong_on_every_group_is_admitted():
    """The criterion A_KERNEL_WRONG_EVERYWHERE_IS_ADMITTED.

    An earlier draft of the gate also demanded agreement somewhere, on the theory
    that a kernel wrong everywhere is suspicious. A kernel wrong on every group is an
    ordinary bug that should score a detection rate of 1.0; that criterion would have
    rejected valid mutants for being too easy to catch.
    """
    rows = [_row("c0", "g0", np.full((2, 3), 9.0)), _row("c1", "g1", np.full((2, 3), 9.0))]
    assert admit(rows, reference_fn=_ref) is True


def test_a_partially_broken_candidate_is_admitted():
    rows = [_row("c0", "g0", np.ones((2, 3))), _row("c1", "g1", np.full((2, 3), 9.0))]
    assert admit(rows, reference_fn=_ref) is True


def test_a_rejection_records_its_reason():
    """The criterion REJECTIONS_CARRY_A_REASON.

    The rejection rate is a finding in its own right: it says what proportion of an
    agent's attempts at a named fault class are not that fault. Dropping refused
    candidates silently would discard it.
    """
    rows = [_row("c0", "g0", np.ones((2, 3)))]
    verdict = admit(rows, reference_fn=_ref)
    assert isinstance(verdict, Rejection)
    assert verdict.reason
    assert verdict.groups_broken == 0
    assert verdict.groups_judgeable == 1


def test_a_mixed_candidate_that_crashes_somewhere_is_still_admitted():
    # Crashing on some cases is normal for a real bug; only crashing EVERYWHERE is
    # disqualifying, because that is what makes every arm abstain.
    rows = [
        _row("c0", "g0", None, status=Status.LAUNCH_ERROR),
        _row("c1", "g1", np.full((2, 3), 9.0)),
    ]
    assert admit(rows, reference_fn=_ref) is True
```

- [ ] **Step 2: Run it red.**

- [ ] **Step 3: Implement.**

```python
"""The admission gate for agent-authored mutants.

An agent asked for a specific fault class returns something plausible; whether it is
actually that fault, actually broken, or actually runnable is not guaranteed. Each
way it can be wrong corrupts a different number, and none of them announces itself:

* a mutant that is secretly **correct** enters the detection denominator as a bug
  nobody can catch, lowering every arm's rate for free;
* one broken in a **different class** than intended corrupts per-class rates while
  the total stays plausible;
* one broken **catastrophically** makes every arm INCONCLUSIVE, at which point the
  driver refuses the run and nothing is recorded at all.

The gate addresses the first and third. It deliberately does NOT address the second --
see `Mutant.intended_class`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from autokernel_pbt.props.backends.base import OUTPUT_NAME, ExecutionResult, Status, kernel_inputs
from autokernel_pbt.props.tolerance import DEFAULT_THRESH, ExactDtypeError, residual_ratio


@dataclass(frozen=True)
class Rejection:
    """Why a candidate was refused, kept rather than discarded.

    The rejection rate is a finding: it says what proportion of an agent's attempts
    at a named fault class are not that fault, which is a fact about code-generating
    models and costs nothing extra to collect.
    """

    reason: str
    groups_broken: int
    groups_judgeable: int


def admit(rows: list[ExecutionResult], *, reference_fn: Callable[..., Any]) -> bool | Rejection:
    """`True` if the candidate belongs in the corpus, else a `Rejection`.

    Two criteria, and no more:

    * **broken somewhere** -- it differs from the reference beyond tolerance on at
      least one case group;
    * **judgeable somewhere** -- at least one group ran to `Status.OK`, so the arms
      have something to judge rather than abstaining everywhere.

    Notably absent: any requirement that the candidate AGREE with the reference
    somewhere. A kernel wrong on every group is an ordinary bug that should score a
    detection rate of 1.0, and demanding agreement would reject valid mutants for
    being too easy to catch.
    """
    judgeable: set[str] = set()
    broken: set[str] = set()

    for row in rows:
        if row.status != Status.OK or OUTPUT_NAME not in row.outputs:
            continue
        judgeable.add(row.case.group_id)
        got = np.atleast_1d(row.outputs[OUTPUT_NAME])
        with np.errstate(all="ignore"):
            expected = np.atleast_1d(np.asarray(reference_fn(**kernel_inputs(row.case))))
        if got.shape != expected.shape:
            broken.add(row.case.group_id)
            continue
        try:
            ratio = residual_ratio(got, expected, dtype=got.dtype, n=got.shape[-1])
        except ExactDtypeError:
            # An exact-dtype output cannot be compared by test ratio. Treat it as
            # unbroken here rather than guessing: admitting it on a technicality
            # would put an unverified candidate in the denominator.
            continue
        if not np.isfinite(ratio) or ratio >= DEFAULT_THRESH:
            broken.add(row.case.group_id)

    if not judgeable:
        return Rejection(
            reason="not judgeable on any group: every case failed to produce an output",
            groups_broken=len(broken),
            groups_judgeable=0,
        )
    if not broken:
        return Rejection(
            reason="not broken on any group: it matches the reference within tolerance",
            groups_broken=0,
            groups_judgeable=len(judgeable),
        )
    return True
```

- [ ] **Step 4: Green, then saboteur-check.** Remove the `if not broken` branch and confirm
      exactly `test_a_candidate_that_is_not_broken_is_rejected` and
      `test_a_rejection_records_its_reason` fail. Restore. Then remove `if not judgeable` and
      confirm exactly `test_a_candidate_that_never_runs_is_rejected` fails. Restore.

- [ ] **Step 5: Commit.**

```bash
git add src/autokernel_pbt/corpus/gate.py tests/unit/corpus/test_gate.py
scripts/git_commit_clean.sh -m "feat: add the mutant admission gate" -m "An agent asked for a fault class returns something plausible; whether it is actually broken or actually runnable is not guaranteed, and each way it can be wrong corrupts a different number without announcing itself. A secretly-correct mutant is the worst: it enters the detection denominator as a bug nobody can catch, so every arm's rate drops for free and nothing downstream looks wrong." -m "Two criteria and no more: broken somewhere, judgeable somewhere. Notably absent is any requirement to AGREE with the reference somewhere -- a kernel wrong on every group is an ordinary bug that should score 1.0, and demanding agreement would reject valid mutants for being too easy to catch." -m "Rejections are recorded with their reason rather than dropped, because the rejection rate says what proportion of an agent's attempts at a named class are not that class."
git branch --show-current
```

---

### Task 4: Author the NumPy mutants, blinded

**Follow `docs/protocol/mutant-authoring.md` exactly.** The authoring agent must not be shown
`acceptance.yaml`, `properties.py`, or any tolerance.

CPU-reachable subcategories, from the design's taxonomy table:

| Subcategory | Bugs | Tasks it applies to |
|---|---|---|
| Operator implementation | 80 | relu, softmax, layernorm |
| Data type semantics | 58 | softmax, layernorm |
| Indexing and stride | 35 | softmax, layernorm |
| Control flow / boundary predicates | 16 | softmax, layernorm |
| Special value handling | 9 | softmax, layernorm |

**Files:** create `kernels/mutants/__init__.py`, `kernels/mutants/numpy_mutants.py`,
`tests/unit/corpus/test_numpy_mutants.py`.

- [ ] **Step 1: For each (task, subcategory) pair, run the protocol prompt** and record the
      returned source verbatim into `numpy_mutants.py`, one function per mutant, each carrying a
      docstring with the subcategory and the verbatim taxonomy quote it was authored from.

- [ ] **Step 2: Gate every candidate.** For each, generate the task's ladder, execute with
      `NumpyBackend`, and call `admit`. Record the verdict.

- [ ] **Step 3: Write the gate-result test**, which must assert the corpus is non-trivial and
      that rejections were kept:

```python
def test_every_registered_mutant_passed_the_gate(repo_root):
    """A mutant in the registry has been shown broken and judgeable, not assumed so."""
    from autokernel_pbt.corpus.gate import admit
    from autokernel_pbt.corpus.registry import NUMPY_MUTANTS
    from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
    from autokernel_pbt.props.generator import Generator
    from autokernel_pbt.props.tasks import REFERENCES, TASKS

    assert NUMPY_MUTANTS, "the corpus is empty"
    backend = NumpyBackend()
    for m in NUMPY_MUTANTS:
        task = TASKS[m.task_id]
        rows = [
            backend.run(m.fn, case)
            for group in Generator(task.domain, seed=0).generate(len(task.domain.shapes))
            for case in group.cases
        ]
        assert admit(rows, reference_fn=REFERENCES[m.task_id]) is True, m.kernel_id


def test_the_corpus_covers_more_than_one_fault_class():
    # A corpus concentrated in one class cannot support per-class reporting.
    from autokernel_pbt.corpus.registry import NUMPY_MUTANTS

    assert len({m.intended_class for m in NUMPY_MUTANTS}) >= 3
```

- [ ] **Step 4: Record the rejections** in `docs/measurements/2026-08-17-mutant-gate-outcomes.md`:
      one row per candidate with task, subcategory, verdict and reason, plus the rejection rate.
      This is a finding, not bookkeeping.

- [ ] **Step 5: Commit.**

```bash
git add kernels/mutants src/autokernel_pbt/corpus/registry.py tests/unit/corpus docs/measurements
scripts/git_commit_clean.sh -m "feat: add the agent-authored NumPy mutation corpus" -m "Authored under docs/protocol/mutant-authoring.md, blinded to acceptance.yaml, the property set and every tolerance. Each mutant carries the verbatim taxonomy quote it was written from, so a reader can check it against the class it claims." -m "Every registered mutant passed the admission gate, and the candidates that did not are recorded with their reasons in docs/measurements -- the rejection rate is a finding about what agent-authored bugs look like, not bookkeeping."
git branch --show-current
```

---

### Task 5: Correct-but-different kernels

The false-positive denominator. A correct kernel bit-identical to the reference tests nothing,
because every tolerance-bearing arm is handed a residual of exactly zero.

**Files:** create `kernels/mutants/correct_variants.py`, extend
`tests/unit/corpus/test_numpy_mutants.py`.

- [ ] **Step 1: Write two correct variants per task** — e.g. softmax with and without float64
      widening; layernorm computing variance as `mean(x²) − mean(x)²` rather than
      `mean((x−mean)²)`, which is algebraically identical and numerically different.

- [ ] **Step 2: Write the test** asserting each variant is correct *and* not bit-identical:

```python
def test_every_correct_variant_is_correct_but_not_bit_identical():
    """Both halves matter.

    Correct, or it belongs in the mutant corpus. Not bit-identical, or every
    tolerance-bearing arm is handed a residual of exactly zero and the
    false-positive measurement is vacuous -- which is the whole reason these exist.
    """
    from autokernel_pbt.corpus.registry import CORRECT_VARIANTS
    from autokernel_pbt.props.generator import Generator
    from autokernel_pbt.props.tasks import REFERENCES, TASKS
    import numpy as np

    assert CORRECT_VARIANTS
    for v in CORRECT_VARIANTS:
        task = TASKS[v.task_id]
        differed = False
        for group in Generator(task.domain, seed=0).generate(len(task.domain.shapes)):
            for case in group.cases:
                got = v.fn(**{"x": case.tensors["x"]})
                exp = REFERENCES[v.task_id](x=case.tensors["x"])
                assert np.allclose(got, exp, rtol=1e-3, atol=1e-5), v.kernel_id
                differed |= not np.array_equal(got, exp)
        assert differed, f"{v.kernel_id} is bit-identical; it measures nothing"
```

- [ ] **Step 3: Green, lint, commit.**

```bash
git add kernels/mutants/correct_variants.py src/autokernel_pbt/corpus/registry.py tests
scripts/git_commit_clean.sh -m "feat: add correct-but-different kernels for the false-positive denominator" -m "A correct kernel bit-identical to the reference measures nothing: every tolerance-bearing arm is handed a residual of exactly zero. These differ in the last few ulps while remaining correct, which is the realistic false-positive risk -- allclose's 5-of-9 on layernorm is already a measured instance of exactly that."
git branch --show-current
```

---

### Task 6: Metrics

Pure functions of the two Parquet tables. No oracle, no kernel, no generator in the loop — that
is what makes a recorded run a reusable dataset.

**Files:** create `src/autokernel_pbt/metrics/__init__.py`,
`src/autokernel_pbt/metrics/rates.py`, `tests/unit/metrics/{__init__.py,test_rates.py}`.

- [ ] **Step 1: Write the failing test.**

```python
"""Metric definitions, computed from artifacts alone."""

from __future__ import annotations

from autokernel_pbt.metrics.rates import arm_rates
from autokernel_pbt.props.scores import ArmScores
from autokernel_pbt.props.verdict import PropertyResult, Verdict


def _r(prop, verdict, group, tolerance_free=False):
    return PropertyResult(prop, 1, tolerance_free, verdict, group_id=group)


def test_detection_rate_counts_groups_not_results():
    """The criterion DETECTION_IS_KEYED_BY_GROUP.

    Measured on this corpus: per-result and per-group rates differ 0.222 against
    0.778 for the same 14 detections. The group is the unit at which arms are
    comparable, because arms emit different numbers of results per group.
    """
    arm = ArmScores(arm="declarative", elapsed_s=0.0, results=[
        _r("a", Verdict.FAIL, "g0"), _r("b", Verdict.PASS, "g0"),
        _r("a", Verdict.PASS, "g1"), _r("b", Verdict.PASS, "g1"),
    ])
    rates = arm_rates(arm)
    assert rates.groups_scored == 2
    assert rates.groups_failed == 1
    assert rates.detection_rate == 0.5


def test_tolerance_free_detection_has_its_own_numerator():
    """The criterion TOLERANCE_FREE_DETECTION_IS_SEPARATE.

    "Bugs found without a tolerance argument" is the project's sharpest claim, so it
    cannot be inferred from the overall rate -- a group failed only by a
    tolerance-bearing property must not count toward it.
    """
    arm = ArmScores(arm="declarative", elapsed_s=0.0, results=[
        _r("tolerance_free_prop", Verdict.FAIL, "g0", tolerance_free=True),
        _r("ratio_prop", Verdict.FAIL, "g1", tolerance_free=False),
    ])
    rates = arm_rates(arm)
    assert rates.detection_rate == 1.0
    assert rates.tolerance_free_detection_rate == 0.5


def test_cases_to_first_failure_is_the_first_failing_group_index():
    arm = ArmScores(arm="reference", elapsed_s=0.0, results=[
        _r("p", Verdict.PASS, "g0"), _r("p", Verdict.PASS, "g1"),
        _r("p", Verdict.FAIL, "g2"),
    ])
    assert arm_rates(arm).cases_to_first_failure == 2


def test_cases_to_first_failure_is_none_when_nothing_failed():
    arm = ArmScores(arm="reference", elapsed_s=0.0, results=[_r("p", Verdict.PASS, "g0")])
    assert arm_rates(arm).cases_to_first_failure is None


def test_inconclusive_groups_are_not_counted_as_detections():
    # A group nobody could judge is not a caught bug. Counting it would inflate every
    # arm's rate by the crash rate, which is the quantity the metric isolates.
    arm = ArmScores(arm="reference", elapsed_s=0.0, results=[
        _r("p", Verdict.INCONCLUSIVE, "g0"), _r("p", Verdict.FAIL, "g1"),
    ])
    rates = arm_rates(arm)
    assert rates.groups_failed == 1
    assert rates.groups_inconclusive == 1


def test_rates_are_computed_from_the_tables_alone(tmp_path, repo_root):
    """The criterion METRICS_COME_FROM_ARTIFACTS_ALONE.

    The recorded run is a reusable dataset only if a rate can be computed from it
    months later with no oracle, kernel or generator available.
    """
    from autokernel_pbt.metrics.rates import rates_from_run
    from autokernel_pbt.props.driver import run_task
    from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference
    import numpy as np

    def unnormalized(x):
        s = x - np.max(x, axis=-1, keepdims=True)
        return np.exp(s).astype(x.dtype)

    run_dir = tmp_path / "run"
    run_task(task=SOFTMAX, kernel=unnormalized, reference_fn=softmax_reference,
             run_dir=run_dir, repo_root=repo_root,
             n_groups=len(SOFTMAX.domain.shapes), seed=42, kernel_id="unnormalized",
             kernel_is_broken=True)
    table = rates_from_run(run_dir)
    assert set(table) == {"allclose", "reference", "declarative", "hybrid"}
    # The measured ladder deflation: 7 of 9 groups, the other two being the
    # single-column rungs where an unnormalized softmax is genuinely correct.
    assert table["declarative"].detection_rate == 7 / 9
```

- [ ] **Step 2: Run red, then implement `rates.py`** with an `ArmRates` dataclass carrying
      `groups_scored`, `groups_failed`, `groups_inconclusive`, `detection_rate`,
      `tolerance_free_detection_rate`, `cases_to_first_failure`; `arm_rates(ArmScores)`; and
      `rates_from_run(run_dir)` built on `driver.read_run`.

- [ ] **Step 3: Green, saboteur-check** (count results instead of groups → exactly
      `test_detection_rate_counts_groups_not_results` fails; count all FAILs toward the
      tolerance-free numerator → exactly `test_tolerance_free_detection_has_its_own_numerator`
      fails).

- [ ] **Step 4: Commit.**

```bash
git add src/autokernel_pbt/metrics tests/unit/metrics
scripts/git_commit_clean.sh -m "feat: add the metric definitions" -m "Pure functions of the two Parquet tables, with no oracle, kernel or generator in the loop -- that is what makes a recorded run a reusable dataset rather than a thing that must be re-derived." -m "The unit is the case group, settled and measured: per-result and per-group rates differ 0.222 against 0.778 for the same 14 detections, because arms emit different numbers of results per group. Tolerance-free detection carries its own numerator, since the project's sharpest claim cannot be inferred from the overall rate."
git branch --show-current
```

---

### Task 7: The report

**Files:** create `src/autokernel_pbt/metrics/report.py`, `tests/unit/metrics/test_report.py`.

- [ ] **Step 1: Write the failing test.**

```python
def test_the_report_states_the_ladder_deflation():
    """The criterion THE_REPORT_STATES_THE_DEFLATION.

    Open obligation 3: the degenerate rungs make every absolute rate understate by a
    measured constant. Arm-vs-arm stays unbiased; the absolute number does not. A
    reader who sees only the table would take the rate at face value, so the caveat
    travels WITH the numbers rather than in a document they may never open.
    """
    from autokernel_pbt.metrics.report import render

    text = render({}, backend="numpy")
    assert "deflat" in text.lower()
    assert "0.778" in text or "7/9" in text


def test_the_report_labels_the_class_column_as_intended():
    # The class is established by the authoring prompt and verified by nothing.
    from autokernel_pbt.metrics.report import render

    assert "intended" in render({}, backend="numpy").lower()
```

- [ ] **Step 2: Implement `render`** producing a Markdown fault-class × arm table with a header
      block stating the deflation constant, the intended-class caveat, and the per-class corpus
      size.

- [ ] **Step 3: Green, lint, commit.**

---

### Task 8: Run the CPU corpus and record the result

- [ ] **Step 1: Drive every mutant and every correct variant** through `driver.run_task` on
      `NumpyBackend`, one run directory each, `kernel_is_broken` set correctly — `True` for
      mutants, `False` for correct variants, never left `None`, since collapsing "not stated" into
      "stated correct" enlarges the false-positive denominator.

- [ ] **Step 2: Render the report** and commit it to
      `docs/measurements/2026-08-17-cpu-corpus-results.md`, with the threats section written
      before the numbers are read.

- [ ] **Step 3: Commit.** This is the checkpoint: a complete CPU result.

---

### Task 9: The Triton twins

**Resolves design §9 open question 2.** Each NumPy mutant gets a Triton twin modelling the *same*
fault class. Where a faithful port is impossible — the class is inexpressible in one substrate —
**record the divergence explicitly rather than substituting a different bug**, because
per-backend detection would otherwise be comparing different defects and the cross-backend
comparison would be meaningless.

**Files:** create `kernels/mutants/triton_mutants.py`, `tests/gpu/test_mutant_twins.py`.

- [ ] **Step 1: Author each twin**, under the same blinded protocol, with the NumPy mutant's
      *fault class* as the brief — not its source, so the twin is not a transliteration.

- [ ] **Step 2: Gate every twin on device**, `gpu`-marked.

- [ ] **Step 3: Record any class that has no faithful twin**, with the reason, in the results doc.

- [ ] **Step 4: Commit.**

---

### Task 10: Cross-backend run and close-out

- [ ] **Step 1: Run the corpus on the GPU** using the Lambda runbook, bring the runs home,
      score offline on CPU.
- [ ] **Step 2: Render the cross-backend report.**
- [ ] **Step 3: Update `specs/README.md` (0008 → implemented) and `CLAUDE.md` open obligations.**
- [ ] **Step 4: Verify the gate:** `ruff check src tests kernels scripts`,
      `pytest -m "not gpu" -q --cov=autokernel_pbt --cov-fail-under=95`, `pytest tests/spec/ -v`.
- [ ] **Step 5: Commit.**

---

## Definition of Done

- [ ] All nine feature 0008 criteria resolve to collectable CPU tests; `tests/spec/` green
- [ ] `pytest -m "not gpu"` passes, no failures, no warnings
- [ ] `ruff check src tests kernels scripts` passes; coverage ≥ 95%
- [ ] Every registered mutant has passed the gate, verified by a test that re-runs it
- [ ] Every rejected candidate is recorded with its reason, and the rejection rate is reported
- [ ] Every correct variant is correct and **not** bit-identical to its reference
- [ ] Detection rate is keyed by case group; tolerance-free detection has its own numerator
- [ ] Rates are computable from the two tables with no oracle in the loop
- [ ] The report states the ladder deflation and the intended-class caveat beside the numbers
- [ ] Any fault class without a faithful Triton twin is recorded, not substituted

## Explicitly Out of Scope

- Metric 4, downstream kernel quality — needs the full agentic loop
- Automatic fault-class verification — the class is intended-by-construction
- Shrinking algorithms; tier-2 properties and `compute-sanitizer` (Phase 3b)
- NKI, and any third backend including Apple Silicon
- Adopting `log2(tile) + n_tiles` as the reference arm's normalization
