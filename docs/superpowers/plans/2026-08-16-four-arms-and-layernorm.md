# Four Arms and the Normalization Rung — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the instrument to four oracle arms, add the normalization rung of the ladder with its authoring cost measured, and make a recorded case group regenerable from a spec.

**Architecture:** Additive. Phase 1.5's storage, driver, and score table are accepted as they landed — `run_dir` holds one kernel, `kernel_id` labels it, and `corpus_fingerprint` makes cross-run joins fail loudly. Nothing here rewrites them.

**Tech Stack:** Python 3.10+, NumPy, safetensors, PyArrow/Parquet, pytest, ruff.

**Design doc:** `docs/superpowers/specs/2026-08-16-phase-2a-instrument-design.md` — read §0 first; §3.1–3.2 are superseded and §2 is what this plan implements.
**Predecessor plan:** `docs/superpowers/plans/2026-08-16-phase-1-5-measurable-runs.md`

---

## What this plan does and does not touch

Phase 1.5 discharged open obligations 1, 2 and 4. What remains from `CLAUDE.md`, and how this plan treats each:

| Obligation | Treatment |
|---|---|
| 1. Declarative/hybrid arms have no acceptance criteria | **In scope** — Task 5 |
| 2. `elapsed_s` is order-biased | **Partly** — Task 4 randomizes arm order so the bias stops being systematic. Repeated timing is Phase 2b's. |
| 3. Partial abstention is undetectable | **Out of scope** — it is a semantics question about what abstention means, not a gap in the instrument |
| 4. `HybridOracle` not wired into `run_task` | **In scope** — Task 4 |
| 5. Degenerate ladder shapes deflate detection | **Out of scope, inherited** — layernorm joins the same ladder; Task 8 records that it inherits the deflation |

## File Structure

| File | Responsibility |
|------|---------------|
| `src/autokernel_pbt/props/oracle.py` | Gains `AllcloseOracle`, the fourth arm |
| `src/autokernel_pbt/props/spec.py` | `CaseSpec` — the reducible representation a future shrinker delta-debugs |
| `src/autokernel_pbt/props/case.py` | `CaseGroup` gains an optional `spec` |
| `src/autokernel_pbt/props/generator.py` | Emits `CaseSpec`; gains `group_from_spec` |
| `src/autokernel_pbt/props/driver.py` | Drives four arms in randomized order |
| `src/autokernel_pbt/props/tasks.py` | Gains `layernorm` and its reference |
| `src/autokernel_pbt/props/properties.py` | Gains `RowsHaveZeroMean`, `RowsHaveUnitVariance` |
| `kernels/tasks/layernorm/acceptance.yaml` | Layernorm's declarative contract |
| `specs/features/0006-four-arms/` | Spec and acceptance criteria (Task 0) |
| `docs/measurements/2026-08-16-layernorm-authoring-cost.md` | Metric 3, opened before layernorm exists |

**Repo conventions that apply to every task.** Read `CLAUDE.md` first.

- **Never run `git commit`.** Use `scripts/git_commit_clean.sh -m "subject" -m "body"`, then verify `git branch --show-current` is non-empty.
- `filterwarnings = ["error"]` — any warning is a test failure.
- `ruff check src tests` must pass; CI enforces it and a 95% coverage floor.
- Every new assertion must be the **unique** catcher for at least one saboteur. Pair each with the exact expected message via `pytest.raises(..., match=)`, and verify by deleting each assertion in turn that precisely its own cases fail.
- Bad **data** → `INCONCLUSIVE`; bad **call** → raise.

---

### Task 0: Feature 0006 spec and acceptance criteria (red)

Feature id 0005 is taken by phase 1.5's "measurable runs". This is **0006**.

**Files:**
- Create: `specs/features/0006-four-arms/spec.md`
- Create: `specs/features/0006-four-arms/acceptance.yaml`
- Create: `tests/spec/test_0006_four_arms.py`
- Modify: `specs/README.md`

- [ ] **Step 1: Write the spec**

Create `specs/features/0006-four-arms/spec.md`:

```markdown
# Feature 0006: Four arms and the normalization rung

## Problem

Three gaps remain between the instrument phase 1.5 delivered and the experiment it exists to
run.

The reference arm was deliberately strengthened into a LAPACK-style normalized test ratio so
it would not be a strawman, and phase 1 measured it as catching everything
`allclose(rtol=1e-5)` catches. That measurement is currently something a reader must take on
trust, because plain `allclose` is not among the arms.

The declarative and hybrid arms — the two the project's claim actually rests on — have no
acceptance criteria, and `HybridOracle` is not driven at all.

The ladder stops at softmax, so the normalization rung named in the parent design is missing,
along with the property class that comes with it.

## Scope

1. **A fourth arm** — plain `allclose`, numpy's defaults, unmodified.
2. **All four arms driven**, in randomized order so `elapsed_s` is not systematically biased
   toward whichever arm ran second.
3. **Acceptance criteria** for the declarative, hybrid and allclose arms.
4. **layernorm** — reference, property set, contract, with its authoring cost measured before
   it is written.
5. **A reducible case spec** — a group is regenerable from `(seed, task_id, group_index,
   shape, transforms)`. No shrinking algorithm.

## Non-goals

- The mutation corpus and the four metrics (phase 2b)
- Any shrinking algorithm — only the representation
- Repeated timing for a fair cost-per-bug denominator (phase 2b)
- Deciding what partial abstention means
- CUDA / Triton / NKI backends and tier-2 telemetry (phase 3)
- Any change to phase 1.5's storage, driver join, or score table

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
```

- [ ] **Step 2: Write the acceptance criteria**

Create `specs/features/0006-four-arms/acceptance.yaml`:

```yaml
feature_id: "0006"
feature_name: four-arms
version: 1

criteria:
  - id: ALLCLOSE_ARM_IS_UNMODIFIED
    description: the fourth arm uses numpy's default tolerances, not tuned ones
    check:
      type: unit_test
      test: tests/unit/props/test_oracle.py::test_allclose_arm_uses_numpy_defaults

  - id: ALLCLOSE_ARM_DETECTS
    description: the fourth arm fails an output its reference rejects
    check:
      type: unit_test
      test: tests/unit/props/test_oracle.py::test_allclose_arm_fails_a_diverging_output

  - id: DECLARATIVE_ARM_DETECTS
    description: the declarative arm fails a kernel that violates a declared property
    check:
      type: unit_test
      test: tests/unit/props/test_oracle.py::test_declarative_oracle_fails_a_violating_kernel

  - id: HYBRID_ARM_COMPOSES
    description: the hybrid arm reports both its declarative and reference components
    check:
      type: unit_test
      test: tests/unit/props/test_oracle.py::test_hybrid_oracle_reports_both_components

  - id: FOUR_ARMS_SCORE_ONE_RUN
    description: one recorded run is scored by all four arms
    check:
      type: unit_test
      test: tests/integration/test_four_arms.py::test_four_arms_score_one_recorded_run

  - id: LAUNCH_ERROR_IS_NEVER_A_CAUGHT_BUG
    description: a failed execution is inconclusive in every arm, never a failure
    check:
      type: unit_test
      test: tests/integration/test_four_arms.py::test_launch_error_is_inconclusive_in_every_arm

  - id: ARM_ORDER_IS_RANDOMIZED
    description: arm evaluation order varies with the seed, so timing bias is not systematic
    check:
      type: unit_test
      test: tests/unit/props/test_driver.py::test_arm_order_varies_with_seed

  - id: CASE_SPEC_REGENERATES
    description: a case group is byte-identical when rebuilt from its spec
    check:
      type: unit_test
      test: tests/unit/props/test_spec.py::test_group_from_spec_is_byte_identical

  - id: LAYERNORM_CONTRACT_BUILDS_AN_ORACLE
    description: layernorm's acceptance.yaml constructs a declarative oracle
    check:
      type: unit_test
      test: tests/unit/props/test_tasks.py::test_layernorm_contract_builds_an_oracle

  - id: LAYERNORM_PROPERTIES_DETECT
    description: the layernorm property set fails a kernel that skips the mean subtraction
    check:
      type: unit_test
      test: tests/unit/props/test_properties.py::test_rows_have_zero_mean_fails_an_unsubtracted_output
```

- [ ] **Step 3: Write the spec test**

Create `tests/spec/test_0006_four_arms.py`:

```python
"""Spec-derived acceptance tests (feature 0006).

These assert traceability: every criterion in acceptance.yaml must name a test that
actually exists and is collectable. This is the mechanism the SDD ADR asks for.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ACCEPTANCE = "specs/features/0006-four-arms/acceptance.yaml"

# The check vocabulary already used by features 0001, 0002, 0004 and 0005. Pinning it
# means a typo'd type (e.g. "unit_tests") fails loudly instead of silently filtering
# the criterion out of every traceability check below.
KNOWN_CHECK_TYPES = {
    "unit_test",
    "cli_help",
    "json_schema",
    "field_present",
    "field_equals",
    "config_equals",
}


def _criteria(repo_root: Path) -> list[dict]:
    data = yaml.safe_load((repo_root / ACCEPTANCE).read_text())
    return data["criteria"]


@pytest.mark.spec
def test_0006_acceptance_file_is_wellformed(repo_root: Path):
    data = yaml.safe_load((repo_root / ACCEPTANCE).read_text())
    assert data["feature_id"] == "0006"
    ids = [c["id"] for c in data["criteria"]]
    assert ids, "acceptance.yaml declares no criteria"
    assert len(ids) == len(set(ids)), f"duplicate criterion ids: {ids}"
    unknown = [
        f"{c['id']} -> {c['check']['type']}"
        for c in data["criteria"]
        if c["check"]["type"] not in KNOWN_CHECK_TYPES
    ]
    assert not unknown, f"unknown check types (typo?): {unknown}"


@pytest.mark.spec
def test_0006_every_criterion_names_an_existing_file(repo_root: Path):
    missing = []
    for criterion in _criteria(repo_root):
        check = criterion["check"]
        if check["type"] != "unit_test":
            continue
        path = check["test"].split("::")[0]
        if not (repo_root / path).exists():
            missing.append(f"{criterion['id']} -> {path}")
    assert not missing, f"criteria reference missing test files: {missing}"


@pytest.mark.spec
def test_0006_every_criterion_is_collectable(repo_root: Path):
    """A criterion pointing at a non-existent test node is untraceable, so it fails."""
    node_ids = [
        c["check"]["test"] for c in _criteria(repo_root) if c["check"]["type"] == "unit_test"
    ]
    # Without this guard an empty list would degrade the command to a bare collect over
    # `testpaths`, which exits 0 — reporting success while tracing nothing.
    assert node_ids, "no unit_test criteria to collect"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *node_ids],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=120,
        # The non-zero exit *is* the signal under test; the assertion below reports it
        # with pytest's own stdout, which check=True would discard into a traceback.
        check=False,
    )
    assert proc.returncode == 0, (
        f"pytest could not collect all criteria:\n{proc.stdout}\n{proc.stderr}"
    )
```

- [ ] **Step 4: Register the feature**

In `specs/README.md`, add a row beneath the 0005 row:

```markdown
| [0006](./features/0006-four-arms/spec.md) | Four arms and the normalization rung | in progress |
```

- [ ] **Step 5: Run the spec test to verify it fails (red)**

Run: `pytest tests/spec/test_0006_four_arms.py -v`
Expected: `test_0006_acceptance_file_is_wellformed` PASSES; the other two FAIL, because `tests/integration/test_four_arms.py`, `tests/unit/props/test_spec.py` and several named nodes do not exist. This red state is the point of the task.

- [ ] **Step 6: Commit**

```bash
git add specs/features/0006-four-arms tests/spec/test_0006_four_arms.py specs/README.md
scripts/git_commit_clean.sh -m "spec: add feature 0006, four arms and the normalization rung" -m "Phase 1.5 deferred acceptance criteria for the declarative and hybrid arms to the measurement work, on the grounds that it would exercise them in anger. This is that work, and the two arms the project's claim rests on get criteria before they are driven."
```

---

### Task 1: AllcloseOracle, the fourth arm

The field's naive default, unmodified. Its value is that it is *not* tuned.

**Files:**
- Modify: `src/autokernel_pbt/props/oracle.py`
- Test: `tests/unit/props/test_oracle.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/props/test_oracle.py`, adding `AllcloseOracle` to the module's imports:

```python
def test_allclose_arm_uses_numpy_defaults():
    """The criterion ALLCLOSE_ARM_IS_UNMODIFIED.

    This arm exists so a reader can judge the reference arm's log2(n)-normalized
    ratio against what the field actually does. A tuned allclose would measure
    nothing.
    """
    oracle = AllcloseOracle(lambda **kw: np.zeros((2, 2), dtype=np.float32))
    assert oracle.rtol == 1e-5
    assert oracle.atol == 1e-8


def test_allclose_arm_passes_a_matching_output():
    x = np.ones((2, 2), dtype=np.float32)
    row = _row(x, np.ones((2, 2), dtype=np.float32))
    oracle = AllcloseOracle(lambda **kw: np.ones((2, 2), dtype=np.float32))
    assert oracle.evaluate([row])[0].verdict is Verdict.PASS


def test_allclose_arm_fails_a_diverging_output():
    """The criterion ALLCLOSE_ARM_DETECTS."""
    x = np.ones((2, 2), dtype=np.float32)
    row = _row(x, np.full((2, 2), 1.1, dtype=np.float32))
    oracle = AllcloseOracle(lambda **kw: np.ones((2, 2), dtype=np.float32))
    assert oracle.evaluate([row])[0].verdict is Verdict.FAIL


def test_allclose_arm_is_inconclusive_on_a_failed_execution():
    # A launch error is not evidence about correctness. Booking it as FAIL would
    # manufacture a caught bug in the headline metric.
    row = ExecutionResult(
        case=_case(np.ones((2, 2), dtype=np.float32)),
        outputs={},
        status=Status.LAUNCH_ERROR,
        error="boom",
    )
    oracle = AllcloseOracle(lambda **kw: np.ones((2, 2), dtype=np.float32))
    assert oracle.evaluate([row])[0].verdict is Verdict.INCONCLUSIVE


def test_allclose_arm_is_inconclusive_when_the_reference_is_non_finite():
    # Symmetric with ReferenceOracle: a broken reference is a harness defect, not a
    # finding about the kernel. Left unchecked, allclose against NaN is False, which
    # would book a correct kernel as a caught bug.
    x = np.ones((2, 2), dtype=np.float32)
    row = _row(x, np.ones((2, 2), dtype=np.float32))
    oracle = AllcloseOracle(lambda **kw: np.full((2, 2), np.nan, dtype=np.float32))
    result = oracle.evaluate([row])[0]
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "reference" in result.detail


def test_allclose_arm_fails_a_shape_disagreement():
    # np.allclose broadcasts, so (2,2) against (2,3) raises rather than returning
    # False. Checked explicitly so the arm reports a finding instead of aborting.
    x = np.ones((2, 2), dtype=np.float32)
    row = _row(x, np.ones((2, 3), dtype=np.float32))
    oracle = AllcloseOracle(lambda **kw: np.ones((2, 2), dtype=np.float32))
    assert oracle.evaluate([row])[0].verdict is Verdict.FAIL


def test_allclose_arm_is_not_tolerance_free():
    x = np.ones((2, 2), dtype=np.float32)
    row = _row(x, np.ones((2, 2), dtype=np.float32))
    oracle = AllcloseOracle(lambda **kw: np.ones((2, 2), dtype=np.float32))
    assert oracle.evaluate([row])[0].tolerance_free is False


def test_allclose_arm_attributes_every_result_to_its_case():
    # Every PropertyResult carries exactly one of case_id/group_id; this arm is
    # per-row, so it always attributes to the case.
    x = np.ones((2, 2), dtype=np.float32)
    row = _row(x, np.ones((2, 2), dtype=np.float32))
    oracle = AllcloseOracle(lambda **kw: np.ones((2, 2), dtype=np.float32))
    result = oracle.evaluate([row])[0]
    assert result.case_id == row.case.case_id
    assert result.group_id == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_oracle.py -k allclose -v`
Expected: FAIL with `NameError: name 'AllcloseOracle' is not defined`

- [ ] **Step 3: Write the implementation**

In `src/autokernel_pbt/props/oracle.py`, add the constant beside `REFERENCE_PROPERTY`:

```python
ALLCLOSE_PROPERTY = "allclose_matches_reference"
```

and the class after `ReferenceOracle`:

```python
class AllcloseOracle:
    """The field's default oracle, deliberately unmodified.

    `torch.allclose(candidate, reference)` against an eager implementation is what
    the kernel literature actually uses, and the parent design's §2 critique is
    aimed at it. `ReferenceOracle` here was strengthened into a LAPACK-style
    normalized test ratio precisely so it would not be a strawman — but that
    strengthening is only credible if a reader can see what the unstrengthened
    version scores on the same executions.

    So this arm is not tuned, and must not become tuned. `rtol` and `atol` are
    numpy's documented defaults. Improving it would delete the comparison it exists
    to provide, which is why a test pins the two constants.

    Harness defects are handled exactly as `ReferenceOracle` handles them, and for
    the same reasons: an exception propagates, because it cannot be confused with a
    finding about the kernel; a non-finite reference output is INCONCLUSIVE naming
    the reference as the culprit, because `allclose` against NaN is False and would
    otherwise book a correct kernel as a caught bug; and a failed execution is
    INCONCLUSIVE, because a kernel that never ran is not a detection.
    """

    name = "allclose"

    def __init__(
        self,
        reference_fn: Callable[..., np.ndarray],
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> None:
        self.reference_fn = reference_fn
        self.rtol = rtol
        self.atol = atol

    def evaluate(self, rows: list[ExecutionResult]) -> list[PropertyResult]:
        _require_rows(self.name, rows)
        return [self._check(row) for row in rows]

    def _result(self, verdict: Verdict, detail: str, case_id: str) -> PropertyResult:
        """Every result this arm emits, built in one place.

        Centralized so ``case_id`` cannot be forgotten on one branch out of five.
        """
        return PropertyResult(
            property_name=ALLCLOSE_PROPERTY,
            tier=TIER_PORTABLE,
            # A tolerance is the entire mechanism; this arm is the definition of
            # tolerance-dependent detection.
            tolerance_free=False,
            verdict=verdict,
            detail=detail,
            case_id=case_id,
        )

    def _check(self, row: ExecutionResult) -> PropertyResult:
        case_id = row.case.case_id
        if not _usable(row):
            return self._result(Verdict.INCONCLUSIVE, _unusable_detail(row), case_id)

        # Suppressed for the reason ReferenceOracle suppresses it: this project sets
        # filterwarnings=["error"], so a numpy overflow inside the reference would
        # raise under test config and pass silently under production config, in a
        # project whose central claim is that arms score identical executions.
        with np.errstate(all="ignore"):
            expected = np.asarray(self.reference_fn(**kernel_inputs(row.case)))

        if not np.all(np.isfinite(expected)):
            return self._result(
                Verdict.INCONCLUSIVE,
                f"the reference, not the kernel, produced a non-finite output for {case_id}",
                case_id,
            )

        got = row.outputs["y"]
        # np.allclose broadcasts, so a (2,2) against a (2,3) raises rather than
        # returning False. Checked explicitly so a shape bug is a finding about the
        # kernel rather than an abort partway through a scoring pass.
        if got.shape != expected.shape:
            return self._result(
                Verdict.FAIL,
                f"shape {got.shape} does not match reference shape {expected.shape}",
                case_id,
            )
        if np.allclose(got, expected, rtol=self.rtol, atol=self.atol):
            return self._result(Verdict.PASS, "", case_id)
        return self._result(
            Verdict.FAIL,
            f"allclose(rtol={self.rtol:g}, atol={self.atol:g}) rejected the output",
            case_id,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/test_oracle.py -v`
Expected: PASS, all green.

- [ ] **Step 5: Verify the arm can fail, by breaking it**

A passing test is a hypothesis. Temporarily change `rtol` to `1e9` and confirm `test_allclose_arm_fails_a_diverging_output` fails; restore it.

Run: `pytest tests/unit/props/test_oracle.py -k allclose -v`
Expected: with `rtol=1e9`, `test_allclose_arm_fails_a_diverging_output` FAILS and `test_allclose_arm_uses_numpy_defaults` FAILS. Restore and confirm green.

- [ ] **Step 6: Lint and commit**

```bash
ruff check src tests
git add src/autokernel_pbt/props/oracle.py tests/unit/props/test_oracle.py
scripts/git_commit_clean.sh -m "feat: add the allclose arm" -m "The reference arm was strengthened into a normalized test ratio so it would not be a strawman, but that strengthening is only credible if a reader can see what the unstrengthened version scores on the same executions. Numpy's defaults, untuned, with the two constants pinned by a test so they stay that way." -m "A non-finite reference is INCONCLUSIVE rather than FAIL: allclose against NaN is False, which would otherwise manufacture a caught bug out of a broken reference."
```

---

### Task 2: Acceptance criteria for the declarative and hybrid arms

Closes **open obligation 1**. The two arms the paper's claim rests on assert nothing about detection today.

**Files:**
- Test: `tests/unit/props/test_oracle.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/props/test_oracle.py`:

```python
def test_declarative_oracle_fails_a_violating_kernel():
    """The criterion DECLARATIVE_ARM_DETECTS.

    Phase 1 asserted this arm records a tolerance_free flag but never that it
    *detects* anything. A property set that cannot fail is not an oracle, and the
    project's own finding is that "the property passed" is uninformative without a
    demonstrated ability to fail.
    """
    x = np.ones((2, 4), dtype=np.float32)
    # Rows sum to 2.0, not 1.0: a softmax whose denominator is wrong.
    row = _row(x, np.full((2, 4), 0.5, dtype=np.float32))
    results = DeclarativeOracle([RowsSumToOne()]).evaluate([row])
    assert summary(results) is Verdict.FAIL
    assert any(r.property_name == "rows_sum_to_one" for r in results)


def test_declarative_oracle_passes_a_correct_kernel():
    # The other half of the pair: an arm that always FAILs is as useless as one that
    # always PASSes, and only the two tests together exclude both.
    x = np.ones((2, 4), dtype=np.float32)
    row = _row(x, np.full((2, 4), 0.25, dtype=np.float32))
    assert summary(DeclarativeOracle([RowsSumToOne()]).evaluate([row])) is Verdict.PASS


def test_hybrid_oracle_reports_both_components():
    """The criterion HYBRID_ARM_COMPOSES.

    The hybrid arm's whole claim is composition, so a result set carrying only one
    component's properties would be the arm silently degrading to that component —
    and the measured difference between arms would then be an artefact.
    """
    x = np.ones((2, 4), dtype=np.float32)
    row = _row(x, np.full((2, 4), 0.25, dtype=np.float32))
    hybrid = HybridOracle(
        declarative=DeclarativeOracle([RowsSumToOne()]),
        reference=ReferenceOracle(lambda **kw: np.full((2, 4), 0.25, dtype=np.float32)),
    )
    names = {r.property_name for r in hybrid.evaluate([row])}
    assert "rows_sum_to_one" in names
    assert REFERENCE_PROPERTY in names
```

Ensure `DeclarativeOracle`, `HybridOracle`, `ReferenceOracle`, `REFERENCE_PROPERTY`, `RowsSumToOne` and `summary` are imported in that module.

- [ ] **Step 2: Run the tests**

Run: `pytest tests/unit/props/test_oracle.py -k "declarative_oracle or hybrid_oracle" -v`
Expected: PASS. These pin existing behaviour that was never asserted. If one fails, that is a real defect in the arm — fix the arm, not the test.

- [ ] **Step 3: Confirm each test is a unique catcher**

Delete `RowsSumToOne` from the property list in `test_declarative_oracle_fails_a_violating_kernel` and confirm only that test fails. Restore. Then make `HybridOracle.evaluate` return only its declarative results and confirm only `test_hybrid_oracle_reports_both_components` fails. Restore.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/props/test_oracle.py
scripts/git_commit_clean.sh -m "test: pin detection for the declarative and hybrid arms" -m "Phase 1 asserted the declarative arm records a tolerance_free flag but never that it detects anything, and the hybrid arm had no criterion at all. An arm that silently degraded to one of its components would have looked identical, and the measured difference between arms would have been an artefact of that."
```

---

### Task 3: CaseSpec, the reducible case representation

Ships the representation a future shrinker delta-debugs, and no algorithm. `NOTES.md` §5.3 records the spirv-fuzz result: reduction over a transform *sequence* is free, but only if the sequence was recorded, and that is an up-front decision.

**Files:**
- Create: `src/autokernel_pbt/props/spec.py`
- Modify: `src/autokernel_pbt/props/case.py`
- Modify: `src/autokernel_pbt/props/generator.py`
- Test: `tests/unit/props/test_spec.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_spec.py`:

```python
"""CaseSpec round-trip and regeneration tests."""

import numpy as np
import pytest

from autokernel_pbt.props.domain import InputDomain, TensorSpec
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.spec import CaseSpec

DOMAIN = InputDomain(
    task_id="softmax",
    tensors=(TensorSpec(name="x", dtype="float32"),),
    shapes=((2, 4), (3, 5)),
    relations=("shift_rows",),
)


def test_spec_round_trips_through_json():
    spec = CaseSpec(
        seed=7, task_id="softmax", group_index=3, shape=(2, 4), transforms=("shift_rows",)
    )
    assert CaseSpec.from_json(spec.to_json()) == spec


def test_spec_normalizes_shape_to_ints():
    # np.int64 dims survive construction otherwise and only fail later at json.dumps.
    spec = CaseSpec(
        seed=1, task_id="t", group_index=0, shape=(np.int64(2), np.int64(4)), transforms=()
    )
    assert spec.shape == (2, 4)
    assert all(type(d) is int for d in spec.shape)


def test_spec_rejects_a_negative_group_index():
    with pytest.raises(ValueError, match="group_index must be non-negative"):
        CaseSpec(seed=1, task_id="t", group_index=-1, shape=(2,), transforms=())


def test_spec_rejects_duplicate_transforms():
    # CaseGroup rejects duplicate relations. Catching it here names the spec that is
    # wrong; catching it there names a group id that gives no clue why.
    with pytest.raises(ValueError, match="duplicate transforms"):
        CaseSpec(
            seed=1,
            task_id="t",
            group_index=0,
            shape=(2,),
            transforms=("shift_rows", "shift_rows"),
        )


def test_generator_stamps_a_spec_on_every_group():
    groups = Generator(DOMAIN, seed=11).generate(2)
    assert groups[0].spec == CaseSpec(
        seed=11, task_id="softmax", group_index=0, shape=(2, 4), transforms=("shift_rows",)
    )


def test_group_from_spec_is_byte_identical():
    """The criterion CASE_SPEC_REGENERATES: a spec is a complete recipe."""
    generator = Generator(DOMAIN, seed=11)
    original = generator.generate(3)[2]
    rebuilt = generator.group_from_spec(original.spec)
    assert rebuilt.group_id == original.group_id
    for a, b in zip(original.cases, rebuilt.cases, strict=True):
        assert a.case_id == b.case_id
        assert a.relation == b.relation
        for name, array in a.tensors.items():
            assert np.array_equal(array, b.tensors[name]), name
            assert array.dtype == b.tensors[name].dtype


def test_group_from_spec_honours_a_reduced_transform_list():
    """Dropping a transform is the unit move of a shrinker; the base is unchanged."""
    generator = Generator(DOMAIN, seed=11)
    full = generator.generate(1)[0]
    reduced = generator.group_from_spec(full.spec.without_transform("shift_rows"))
    assert {c.relation for c in reduced.cases} == {"base"}
    assert np.array_equal(reduced.base.tensors["x"], full.base.tensors["x"])


def test_without_transform_rejects_an_absent_name():
    spec = CaseSpec(
        seed=1, task_id="t", group_index=0, shape=(2,), transforms=("shift_rows",)
    )
    with pytest.raises(ValueError, match="does not carry transform 'permute_last_axis'"):
        spec.without_transform("permute_last_axis")


def test_group_from_spec_rejects_a_spec_for_another_task():
    generator = Generator(DOMAIN, seed=11)
    other = CaseSpec(seed=11, task_id="relu", group_index=0, shape=(2, 4), transforms=())
    with pytest.raises(ValueError, match="spec is for task 'relu'"):
        generator.group_from_spec(other)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.spec'`

- [ ] **Step 3: Write CaseSpec**

Create `src/autokernel_pbt/props/spec.py`:

```python
"""The reducible description of one case group.

A `CaseSpec` is everything needed to rebuild a group *given the run's domain*: the
seed, which group it is, the shape it was assigned, and the ordered transforms
applied to its base case. The domain is not duplicated here — a spec that carried
its own copy could disagree with the one the run was generated under.

WHY THIS EXISTS NOW, with no shrinker to use it. `NOTES.md` §5.3 records the
spirv-fuzz result: if metamorphic transformations are small and independent, plain
delta debugging over the *transformation sequence* gives reduction for free — and it
is an architectural decision that must be made up front, not retrofitted. Once a
corpus is recorded against cases that cannot be described, shrinking means
re-executing on hardware to explore, which is exactly the cost the record/replay
architecture exists to avoid.

Shrinking a *tensor* is deliberately not supported. The reduction is over
`transforms`, which is a short list of names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaseSpec:
    """A regeneration recipe for one case group, relative to the run's domain."""

    seed: int
    task_id: str
    group_index: int
    shape: tuple[int, ...]
    transforms: tuple[str, ...]

    def __post_init__(self) -> None:
        # Normalize unconditionally, as InputDomain and Case do. A guard on
        # `isinstance(..., tuple)` would let an already-tuple shape skip int()
        # coercion, so np.int64 dims would survive construction and only fail later
        # when the spec is JSON-encoded.
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))
        object.__setattr__(self, "transforms", tuple(self.transforms))
        if self.group_index < 0:
            msg = f"group_index must be non-negative, got {self.group_index}"
            raise ValueError(msg)
        if len(set(self.transforms)) != len(self.transforms):
            msg = f"spec has duplicate transforms: {list(self.transforms)}"
            raise ValueError(msg)

    def without_transform(self, name: str) -> CaseSpec:
        """This spec with one transform dropped — the unit move of a future shrinker."""
        if name not in self.transforms:
            msg = (
                f"spec for group {self.group_index} does not carry transform {name!r}; "
                f"it has {list(self.transforms)}"
            )
            raise ValueError(msg)
        return CaseSpec(
            seed=self.seed,
            task_id=self.task_id,
            group_index=self.group_index,
            shape=self.shape,
            transforms=tuple(t for t in self.transforms if t != name),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "task_id": self.task_id,
            "group_index": self.group_index,
            "shape": list(self.shape),
            "transforms": list(self.transforms),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseSpec:
        return cls(
            seed=data["seed"],
            task_id=data["task_id"],
            group_index=data["group_index"],
            shape=tuple(data["shape"]),
            transforms=tuple(data["transforms"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> CaseSpec:
        return cls.from_dict(json.loads(text))
```

- [ ] **Step 4: Add the field to CaseGroup**

In `src/autokernel_pbt/props/case.py`, add the import:

```python
from autokernel_pbt.props.spec import CaseSpec
```

and the field after `cases` in the `CaseGroup` dataclass:

```python
    # Optional so a hand-built group in a test need not invent a recipe it will never
    # use. Every group the Generator produces carries one; a group without a spec
    # simply cannot be regenerated or shrunk, which a test group never needs to be.
    spec: CaseSpec | None = None
```

- [ ] **Step 5: Route generation through the spec**

In `src/autokernel_pbt/props/generator.py`, add the import:

```python
from autokernel_pbt.props.spec import CaseSpec
```

Replace `generate`'s group-building loop, and add the two methods:

```python
    def generate(self, n_groups: int) -> list[CaseGroup]:
        """Produce ``n_groups`` case groups.

        Group *i* is stable under changes to ``n_groups``, to the tensor set, and to
        relation ordering. It changes only when ``seed``, ``i``, that group's own
        specs, or ``shapes`` change — a ``shapes`` edit remaps index to shape, which
        is a visible semantic change to what the domain means rather than an
        invisible value shift.

        Every group is built by ``group_from_spec``, never alongside it. Two code
        paths producing "the same" group is the drift that would make a regenerated
        case differ from the recorded one by a bit, which nothing would catch until a
        shrink reported a case the run never executed.
        """
        if n_groups < 0:
            msg = f"n_groups must be non-negative, got {n_groups}"
            raise ValueError(msg)
        warning = self._unexercised_shapes_warning(n_groups)
        if warning is not None:
            warnings.warn(warning, stacklevel=2)
        return [self.group_from_spec(self._spec_for(i)) for i in range(n_groups)]

    def _spec_for(self, index: int) -> CaseSpec:
        """The recipe for group ``index`` under this generator's domain and seed."""
        return CaseSpec(
            seed=self.seed,
            task_id=self.domain.task_id,
            group_index=index,
            # Shape-first: cycle through every shape before repeating any.
            shape=self.domain.shapes[index % len(self.domain.shapes)],
            transforms=tuple(self.domain.relations),
        )

    def group_from_spec(self, spec: CaseSpec) -> CaseGroup:
        """Rebuild one case group from its recipe.

        Byte-identical to the original because the rng is a pure function of
        ``(seed, group_index)`` and the transforms are applied in recorded order. A
        spec with a *reduced* transform list rebuilds the same base case with fewer
        partners, which is the unit move of a future shrinker.
        """
        if spec.task_id != self.domain.task_id:
            msg = (
                f"spec is for task {spec.task_id!r} but this generator carries a domain "
                f"for {self.domain.task_id!r}"
            )
            raise ValueError(msg)
        # One independent stream per group: group i's bytes depend only on (seed, i)
        # and the specs it actually reads — never on how many groups were requested,
        # nor on unrelated tensors or relations. The list-key form is preferred over
        # rng.spawn() because it is a pure function of (seed, index), so one group can
        # be regenerated standalone — which is precisely what this method does.
        rng = np.random.default_rng([spec.seed, spec.group_index])
        group_id = f"{spec.task_id}-g{spec.group_index:05d}"
        base = Case(
            case_id=f"{group_id}-base",
            group_id=group_id,
            relation=BASE_RELATION,
            task_id=spec.task_id,
            dtype=self.domain.tensors[0].dtype,
            shape=spec.shape,
            tensors={t.name: _sample(t, spec.shape, rng) for t in self.domain.tensors},
        )
        cases = [base]
        for relation_name in spec.transforms:
            cases.append(self._relation(relation_name).derive(base, rng))
        return CaseGroup(group_id=group_id, cases=tuple(cases), spec=spec)
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/unit/props/test_spec.py tests/unit/props/test_generator.py tests/unit/props/test_case.py -v`
Expected: PASS. The existing generator tests must not change — routing through `group_from_spec` is a refactor, and any behavioural difference is a bug.

Run: `pytest -m "not gpu" -q`
Expected: PASS, no failures and no warnings.

- [ ] **Step 7: Lint and commit**

```bash
ruff check src tests
git add src/autokernel_pbt/props/spec.py src/autokernel_pbt/props/case.py src/autokernel_pbt/props/generator.py tests/unit/props/test_spec.py
scripts/git_commit_clean.sh -m "feat: describe every case group by a reducible spec" -m "Ships the representation a shrinker delta-debugs and no algorithm. Reduction over a transform sequence is free only if the sequence was recorded, and retrofitting that after a corpus exists means re-executing on hardware to explore -- the cost record/replay exists to avoid." -m "generate() now builds each group by calling group_from_spec rather than alongside it, because two paths producing the same group is the drift that would make a regenerated case differ by a bit, which nothing would catch until a shrink reported a case the run never executed."
```

---

### Task 4: Drive four arms, in randomized order

Wires `AllcloseOracle` and `HybridOracle` into the driver, closing **open obligation 4**, and randomizes arm order so `elapsed_s` stops being systematically biased toward whichever arm ran second — a partial answer to **obligation 2**.

**Files:**
- Modify: `src/autokernel_pbt/props/driver.py`
- Test: `tests/unit/props/test_driver.py`, `tests/integration/test_four_arms.py`

- [ ] **Step 1: Read the current driver**

Run: `sed -n '249,340p' src/autokernel_pbt/props/driver.py`

`run_task` currently builds two arms and scores them in a fixed order. Note how it constructs them and how `ArmScores` records `elapsed_s`, because this task changes only which arms exist and the order they run in — not the join, the group keying, or the table.

- [ ] **Step 2: Write the failing tests**

Append to `tests/unit/props/test_driver.py`:

```python
def test_arm_order_varies_with_seed():
    """The criterion ARM_ORDER_IS_RANDOMIZED.

    elapsed_s is order-biased: whichever arm runs second inherits warm caches. A
    fixed order makes that bias systematic, so a cost-per-bug comparison would
    measure position rather than cost. Randomizing does not make a single run's
    timing fair — only repetition does, which is phase 2b's — but it stops the bias
    favouring the same arm every time.
    """
    orders = {tuple(arm_order(seed)) for seed in range(20)}
    assert len(orders) > 1, "arm order is identical across seeds"


def test_arm_order_is_a_permutation_of_every_arm():
    # A shuffle that dropped an arm would silently reduce the experiment.
    assert sorted(arm_order(0)) == sorted(ARM_NAMES)


def test_arm_order_is_deterministic_for_a_seed():
    # The run must be reproducible: same seed, same order, same timings to compare.
    assert arm_order(3) == arm_order(3)
```

Create `tests/integration/test_four_arms.py`:

```python
"""End-to-end: one recorded run scored by all four arms."""

import numpy as np
import pytest

from autokernel_pbt.props.driver import ARM_NAMES, read_run, run_task
from autokernel_pbt.props.tasks import REFERENCES, TASKS
from autokernel_pbt.props.verdict import Verdict


def good_softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def unnormalized_softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(shifted)


def exploding_kernel(x):
    msg = "kernel exploded"
    raise RuntimeError(msg)


def _run(tmp_path, repo_root, kernel, kernel_id):
    run_task(
        task=TASKS["softmax"],
        kernel=kernel,
        reference_fn=REFERENCES["softmax"],
        run_dir=tmp_path,
        repo_root=repo_root,
        n_groups=3,
        seed=5,
        kernel_id=kernel_id,
    )
    return read_run(tmp_path)


def test_four_arms_score_one_recorded_run(tmp_path, repo_root):
    """The criterion FOUR_ARMS_SCORE_ONE_RUN."""
    _, arms = _run(tmp_path, repo_root, good_softmax, "good")
    assert sorted(a.arm for a in arms) == sorted(ARM_NAMES)


def test_every_arm_catches_an_unnormalized_softmax(tmp_path, repo_root):
    _, arms = _run(tmp_path, repo_root, unnormalized_softmax, "broken")
    for arm in arms:
        verdicts = [r.verdict for r in arm.results]
        assert Verdict.FAIL in verdicts, f"{arm.arm} did not catch an unnormalized softmax"


def test_launch_error_is_inconclusive_in_every_arm(tmp_path, repo_root):
    """The criterion LAUNCH_ERROR_IS_NEVER_A_CAUGHT_BUG.

    A kernel that never ran is not a detection. Booking it as one would inflate
    every arm's rate by the crash rate, which is the metric's whole point.
    """
    _, arms = _run(tmp_path, repo_root, exploding_kernel, "boom")
    for arm in arms:
        verdicts = {r.verdict for r in arm.results}
        assert Verdict.FAIL not in verdicts, f"{arm.arm} scored a launch error as a caught bug"
        assert Verdict.INCONCLUSIVE in verdicts


def test_allclose_and_reference_agree_on_a_correct_kernel(tmp_path, repo_root):
    # The comparison the fourth arm exists to enable: on a correct kernel the
    # strengthened ratio and the field's default must both pass, or the reference
    # arm's calibration is off.
    _, arms = _run(tmp_path, repo_root, good_softmax, "good")
    by_arm = {a.arm: a for a in arms}
    for name in ("allclose", "reference"):
        assert Verdict.FAIL not in {r.verdict for r in by_arm[name].results}


def test_every_score_row_is_keyed_by_group(tmp_path, repo_root):
    # Phase 1.5's invariant, re-asserted for the two new arms: the case group is the
    # unit at which arms are comparable, and a row without one is unjoinable.
    _, arms = _run(tmp_path, repo_root, good_softmax, "good")
    for arm in arms:
        assert all(r.group_id for r in arm.results), f"{arm.arm} emitted an unkeyed row"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/props/test_driver.py -k arm_order tests/integration/test_four_arms.py -v`
Expected: FAIL with `ImportError: cannot import name 'arm_order'` and `cannot import name 'ARM_NAMES'`

- [ ] **Step 4: Write the implementation**

In `src/autokernel_pbt/props/driver.py`, add the imports for `AllcloseOracle` and `HybridOracle` alongside the existing oracle imports, then add above `run_task`:

```python
#: The four arms, in canonical order: the field's default, the strengthened
#: reference, the declarative set, and their composition.
ARM_NAMES = ("allclose", "reference", "declarative", "hybrid")


def arm_order(seed: int) -> list[str]:
    """The order to evaluate arms in for this run.

    `elapsed_s` is order-biased — whichever arm runs second inherits warm caches and
    a warm interpreter. Under a fixed order that bias is *systematic*, so any
    cost-per-bug comparison between arms would partly measure position.

    Randomizing per run does not make a single run's timing fair; only repeated
    timing does, and that belongs to the metrics phase. What it does is stop the
    bias favouring the same arm every time, so that averaging across runs converges
    on something meaningful instead of on the bias.

    Derived from the run's own seed rather than from entropy, because a run must be
    reproducible: the same seed must replay to the same order and therefore to
    comparable timings.
    """
    rng = np.random.default_rng([seed, 0xA6])
    return [ARM_NAMES[i] for i in rng.permutation(len(ARM_NAMES))]
```

In `run_task`, replace the two-line arm construction. The current code is:

```python
    declarative = oracle_from_contract(load_contract(contract_path(task, repo_root)))
    arms: list[Oracle] = [ReferenceOracle(reference_fn), declarative]
```

Replace it with:

```python
    declarative = oracle_from_contract(load_contract(contract_path(task, repo_root)))
    reference = ReferenceOracle(reference_fn)
    # allclose and reference share one reference implementation, so any difference
    # between them is the comparison method and nothing else — which is the whole
    # point of carrying the field's default alongside the strengthened ratio.
    by_name: dict[str, Oracle] = {
        "allclose": AllcloseOracle(reference_fn),
        "reference": reference,
        "declarative": declarative,
        "hybrid": HybridOracle(declarative=declarative, reference=reference),
    }
    arms: list[Oracle] = [by_name[name] for name in arm_order(seed)]
```

**The scoring loop below is unchanged.** It iterates `for oracle in arms` and keys each `ArmScores` by `oracle.name` rather than by a literal, so four arms flow through it exactly as two did. `AllcloseOracle.name` is `"allclose"` and `HybridOracle.name` is `"hybrid"`, matching `ARM_NAMES`. Do not touch the timing bracket, `_keyed_by_group`, or `_verify_join`.

- [ ] **Step 5: Update run_task's docstring, which this task half-answers**

`run_task`'s docstring currently says a usable cost-per-bug figure "needs repeated timing with randomized arm order and a reported spread, and that belongs to the metrics phase." Half of that is now done, and leaving the sentence would misdescribe the code. Replace that paragraph with:

```
    A usable cost-per-bug figure needs repeated timing with a reported spread. Arm order
    is randomized per run as of feature 0006 — see ``arm_order`` — so the bias is no
    longer systematic, but a *single* run's value is still not a fair denominator, and
    the repetition belongs to the metrics phase where the number is consumed. Note the
    measured magnitudes below sit near the clock's noise floor at ~0.5 ms per arm, so
    randomization buys correctness of method rather than precision.
```

Leave the measured ratios (0.73/0.70, and the independent 0.80/0.67) in place. They are the evidence for the bias and randomizing does not invalidate them.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/unit/props/test_driver.py tests/integration/test_four_arms.py -v`
Expected: PASS, all green.

- [ ] **Step 7: Confirm the coverage check still holds for four arms**

Phase 1.5's driver refuses an arm that is `INCONCLUSIVE` on every group. `HybridOracle`'s reference component is deliberately conditional, so confirm the hybrid arm is not tripping that check on the ladder corpus.

Run: `pytest -m "not gpu" -q`
Expected: PASS. If the hybrid arm trips the whole-table coverage check, that is the interaction `CLAUDE.md` obligation 4 warned about — resolve it by scoping the check to the arm's own claimed coverage, and record the decision in the commit body.

- [ ] **Step 8: Lint and commit**

```bash
ruff check src tests
git add src/autokernel_pbt/props/driver.py tests/unit/props/test_driver.py tests/integration/test_four_arms.py
scripts/git_commit_clean.sh -m "feat: drive all four arms in randomized order" -m "allclose and hybrid join the driver, so the two arms the project's claim rests on are finally exercised end to end rather than only in unit tests." -m "Arm order is a permutation of the run's seed. elapsed_s is order-biased -- whichever arm runs second inherits warm caches -- and under a fixed order that bias is systematic, so a cost-per-bug comparison would partly measure position. Randomizing does not make one run's timing fair; it stops the bias favouring the same arm every time, so averaging across runs converges on cost rather than on position."
```

---

### Task 5: Open the authoring-cost record, before layernorm exists

**Order is the point of this task.** Metric 3 is measured, not reconstructed. A cell filled from memory at the end is the failure mode this record exists to prevent.

**Files:**
- Create: `docs/measurements/2026-08-16-layernorm-authoring-cost.md`

- [ ] **Step 1: Create the record**

Create `docs/measurements/2026-08-16-layernorm-authoring-cost.md`:

```markdown
# Authoring cost: onboarding layernorm

**Metric:** parent design §7 metric 3 — effort to onboard a new task under each oracle strategy.

**Protocol.** Opened before any layernorm code existed, and filled as the work happened. `n = 1`,
which is weak, and any report of it must say so. It is pre-registered and honest, which
reconstructing effort afterwards is not.

The two arms are costed separately because that *is* the comparison: the reference arm needs a
trusted implementation, the declarative arm needs a property set, and the open question is
which is cheaper to author for a new op.

| Measure | Reference arm | Declarative arm |
|---|---|---|
| Wall-clock minutes | | |
| Lines of code authored | | |
| Lines auto-drafted and kept unchanged | | |
| Lines auto-drafted then corrected | | |
| Token cost, if agent-authored | | |
| Defects found in review | | |

**Leave a cell blank rather than estimate it.** A blank cell is data; a remembered number is
not.

## Notes

Record anything that made one arm harder than the other — a property that was hard to state, a
reference that was hard to trust, a relation that turned out vacuous, a tolerance that had to
be derived rather than guessed.

## Threats to this measurement

- `n = 1`, one op, one author.
- The author had already read the softmax property set, so the declarative arm benefits from
  transfer that a genuinely new task would not get.
- Wall-clock includes review rounds, which the repo's adversarial review standard makes
  unusually heavy compared with typical practice.
```

- [ ] **Step 2: Commit**

```bash
git add docs/measurements
scripts/git_commit_clean.sh -m "docs: open the layernorm authoring-cost record" -m "Committed before any layernorm code exists, because metric 3 measured after the fact is a remembered number rather than a measurement. The threats section is written now too, while there is no result to be tempted to protect."
```

---

### Task 6: The layernorm reference

**Files:**
- Modify: `src/autokernel_pbt/props/tasks.py`
- Test: `tests/unit/props/test_tasks.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/props/test_tasks.py`:

```python
def test_layernorm_is_registered():
    assert "layernorm" in TASKS
    assert TASKS["layernorm"].domain.task_id == "layernorm"


def test_layernorm_reference_normalizes_each_row():
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    y = REFERENCES["layernorm"](x=x)
    assert np.isclose(y.mean(), 0.0, atol=1e-6)
    # Population variance, not sample: the normalization divides by n, not n-1.
    assert np.isclose(y.var(), 1.0, atol=1e-4)


def test_layernorm_reference_preserves_dtype():
    x = np.ones((2, 4), dtype=np.float32)
    assert REFERENCES["layernorm"](x=x).dtype == np.float32


def test_layernorm_reference_survives_a_constant_row():
    # Zero variance would divide by zero. eps is what makes this defined, and a
    # constant row is reachable from the ladder's (1, 1) and (17, 1) rungs.
    x = np.full((1, 4), 3.0, dtype=np.float32)
    y = REFERENCES["layernorm"](x=x)
    assert np.all(np.isfinite(y))
    assert np.allclose(y, 0.0)


def test_layernorm_reference_is_shift_invariant():
    # The law ShiftRows exercises. Unlike softmax's, this one is exact in real
    # arithmetic: subtracting the row mean removes any per-row constant.
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    shifted = x + np.float32(100.0)
    assert np.allclose(REFERENCES["layernorm"](x=x), REFERENCES["layernorm"](x=shifted), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_tasks.py -k layernorm -v`
Expected: FAIL with `KeyError: 'layernorm'`

- [ ] **Step 3: Note the start time, then write the implementation**

Record the start time in the measurement record. In `src/autokernel_pbt/props/tasks.py`, add after `softmax_reference`:

```python
#: Added to the row variance before the square root. 1e-5 is PyTorch's `LayerNorm`
#: default, chosen so the reference matches the implementation a kernel author is
#: most likely to have been targeting.
LAYERNORM_EPS = 1e-5


def layernorm_reference(x: np.ndarray) -> np.ndarray:
    """Row-wise layer normalization over the last axis, without affine parameters.

    No learnable scale or shift. Those are a separate op fused in practice, and
    including them would mean handing the declarative arm weights it plays no part
    in choosing; the normalization itself is what has interesting properties.

    Variance is the *population* variance (divide by n), matching PyTorch and every
    kernel implementation of it. The sample variance would put the reference a factor
    of sqrt(n/(n-1)) from every correct kernel — at the ladder's (3, 7) rung that is
    8%, far above any tolerance, so every kernel would be booked as a caught bug.

    Accumulation is in float64 and the result cast back, as `softmax_reference` does:
    the reference is the trusted side and is allowed a wider intermediate, while the
    arms measure the residual against the kernel's own dtype.

    `eps` is inside the square root, which is where PyTorch puts it. Outside, a
    constant row gives 0/eps rather than 0/sqrt(eps) — a different value, for a case
    the ladder reaches at two of its nine rungs.
    """
    wide = np.asarray(x, dtype=np.float64)
    mean = np.mean(wide, axis=-1, keepdims=True)
    centered = wide - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return (centered / np.sqrt(variance + LAYERNORM_EPS)).astype(x.dtype)
```

Add the task after `SOFTMAX`:

```python
#: Row-wise layer normalization. The normalization rung: it introduces mean-zero and
#: unit-variance, which are structural facts about the output that need no reference
#: implementation to check, and a division whose denominator can be driven to zero —
#: a second numerical-stability story independent of softmax's overflow one.
#:
#: `shift_rows` is carried because layernorm is *exactly* shift invariant in real
#: arithmetic: subtracting the row mean removes any per-row constant. Unlike softmax,
#: whose shift invariance breaks only when exp overflows and which therefore needs
#: overflow-scale shifts to be non-vacuous, this is a genuine algebraic law at any
#: scale. The relation's dtype-derived default scale is kept anyway, because a kernel
#: that computes its mean in reduced precision degrades exactly there.
LAYERNORM = Task(
    task_id="layernorm",
    domain=InputDomain(
        task_id="layernorm",
        tensors=(TensorSpec(name="x", dtype="float32", distribution="normal"),),
        shapes=_LADDER_SHAPES,
        relations=(ShiftRows.name,),
    ),
)
```

Update both registries:

```python
TASKS: dict[str, Task] = {task.task_id: task for task in (RELU, SOFTMAX, LAYERNORM)}

REFERENCES = {
    RELU.task_id: relu_reference,
    SOFTMAX.task_id: softmax_reference,
    LAYERNORM.task_id: layernorm_reference,
}
```

- [ ] **Step 4: Run the tests and fill the reference-arm column**

Run: `pytest tests/unit/props/test_tasks.py -v`
Expected: PASS.

Fill the **Reference arm** column of the measurement record now, while the work is fresh. Leave the declarative column blank; Task 7 fills it.

- [ ] **Step 5: Lint and commit**

```bash
ruff check src tests
git add src/autokernel_pbt/props/tasks.py tests/unit/props/test_tasks.py docs/measurements
scripts/git_commit_clean.sh -m "feat: add the layernorm reference, the normalization rung" -m "Population variance, not sample. The sample variance would sit a factor of sqrt(n/(n-1)) from every correct kernel -- 8% at the ladder's (3,7) rung -- which dwarfs any tolerance and would book every kernel as a caught bug." -m "eps goes inside the square root, where PyTorch puts it. Outside, a constant row gives 0/eps rather than 0/sqrt(eps), and the ladder reaches constant rows at two of its nine rungs."
```

---

### Task 7: The layernorm property set and contract

**Files:**
- Modify: `src/autokernel_pbt/props/properties.py`
- Create: `kernels/tasks/layernorm/acceptance.yaml`
- Modify: `docs/measurements/2026-08-16-layernorm-authoring-cost.md`
- Test: `tests/unit/props/test_properties.py`, `tests/unit/props/test_tasks.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/props/test_properties.py`:

```python
def test_rows_have_zero_mean_passes_a_normalized_output():
    x = np.ones((2, 4), dtype=np.float32)
    y = np.array([[-1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    assert RowsHaveZeroMean().check(_row(x, y)).verdict is Verdict.PASS


def test_rows_have_zero_mean_fails_an_unsubtracted_output():
    """The criterion LAYERNORM_PROPERTIES_DETECT.

    A kernel that computes the variance but forgets to subtract the mean is the most
    common layernorm defect, and it needs no reference implementation to catch.
    """
    x = np.ones((2, 4), dtype=np.float32)
    y = np.array([[1.0, 3.0, 1.0, 3.0], [3.0, 1.0, 3.0, 1.0]], dtype=np.float32)
    assert RowsHaveZeroMean().check(_row(x, y)).verdict is Verdict.FAIL


def test_rows_have_unit_variance_passes_a_normalized_output():
    x = np.ones((2, 4), dtype=np.float32)
    y = np.array([[-1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    assert RowsHaveUnitVariance().check(_row(x, y)).verdict is Verdict.PASS


def test_rows_have_unit_variance_fails_an_unscaled_output():
    # A kernel that centered but never divided: passes the mean check, fails this one.
    x = np.ones((2, 4), dtype=np.float32)
    y = np.array([[-3.0, 3.0, -3.0, 3.0], [3.0, -3.0, 3.0, -3.0]], dtype=np.float32)
    assert RowsHaveUnitVariance().check(_row(x, y)).verdict is Verdict.FAIL


def test_unit_variance_abstains_on_an_all_zero_output():
    # Layernorm of a constant row is identically zero, not unit variance, because eps
    # dominates the denominator. That is correct, and the ladder reaches it at the
    # (1,1) and (17,1) rungs — judging it would fail every correct kernel there.
    x = np.ones((2, 4), dtype=np.float32)
    y = np.zeros((2, 4), dtype=np.float32)
    assert RowsHaveUnitVariance().check(_row(x, y)).verdict is Verdict.INCONCLUSIVE


def test_zero_mean_is_not_tolerance_free():
    # A float sum of centered values is not exactly zero, so this needs a threshold.
    # Claiming otherwise would inflate the tolerance-free count the headline rests on.
    assert RowsHaveZeroMean().tolerance_free is False


def test_unit_variance_is_not_tolerance_free():
    assert RowsHaveUnitVariance().tolerance_free is False


def test_zero_mean_is_inconclusive_on_a_failed_execution():
    row = ExecutionResult(
        case=_case(np.ones((2, 4), dtype=np.float32)),
        outputs={},
        status=Status.LAUNCH_ERROR,
        error="boom",
    )
    assert RowsHaveZeroMean().check(row).verdict is Verdict.INCONCLUSIVE


def test_layernorm_properties_pass_the_real_reference():
    # The properties must be true of a correct implementation across the whole
    # ladder, or they are false alarms rather than an oracle. This is the check that
    # would have caught a vacuous or over-tight threshold.
    from autokernel_pbt.props.generator import Generator
    from autokernel_pbt.props.tasks import REFERENCES, TASKS

    for group in Generator(TASKS["layernorm"].domain, seed=0).generate(9):
        for case in group.cases:
            y = REFERENCES["layernorm"](x=case.tensors["x"])
            row = ExecutionResult(case=case, outputs={"y": y})
            assert RowsHaveZeroMean().check(row).verdict is not Verdict.FAIL
            assert RowsHaveUnitVariance().check(row).verdict is not Verdict.FAIL
```

Append to `tests/unit/props/test_tasks.py`:

```python
def test_layernorm_contract_builds_an_oracle(repo_root):
    """The criterion LAYERNORM_CONTRACT_BUILDS_AN_ORACLE."""
    from autokernel_pbt.props.contract import load_contract, oracle_from_contract

    contract = load_contract(repo_root / "kernels/tasks/layernorm/acceptance.yaml")
    oracle = oracle_from_contract(contract)
    # DeclarativeOracle splits its set in two; there is no single `.properties`.
    names = {p.name for p in (*oracle.case_properties, *oracle.group_properties)}
    assert "rows_have_zero_mean" in names
    assert "rows_have_unit_variance" in names
    # shift_invariance is a group property, so a test reading only case_properties
    # would pass while the metamorphic half of the contract was silently dropped.
    assert "shift_invariance" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/props/test_properties.py -k "zero_mean or unit_variance or layernorm" -v`
Expected: FAIL with `NameError: name 'RowsHaveZeroMean' is not defined`

- [ ] **Step 3: Write the properties**

In `src/autokernel_pbt/props/properties.py`, add after `RowsSumToOne`:

```python
class RowsHaveZeroMean:
    """Every row of a layernorm output sums to approximately zero.

    Structural: it follows from the definition and needs no reference
    implementation, which is what makes it a declarative-arm property. It catches
    the most common layernorm defect — a kernel that computes the variance but
    forgets to subtract the mean, or subtracts one computed over the wrong axis.

    NOT tolerance-free, and the distinction matters to the headline claim. A float
    sum of n centered values is not exactly zero; it is bounded by roughly
    n * eps * max|y|. Declaring this tolerance-free would inflate the count the
    project's sharpest claim rests on, so it carries a threshold and says so.
    """

    name = "rows_have_zero_mean"
    tier = TIER_PORTABLE
    tolerance_free = False

    def check(self, row: ExecutionResult) -> PropertyResult:
        if not _usable(row):
            return _result(
                self, Verdict.INCONCLUSIVE, _unusable_detail(row), case_id=row.case.case_id
            )
        y = row.outputs["y"]
        unjudgeable = _unjudgeable(y)
        if unjudgeable:
            return _result(self, Verdict.INCONCLUSIVE, unjudgeable, case_id=row.case.case_id)
        means = np.mean(np.asarray(y, dtype=np.float64), axis=-1)
        # Scaled by row width and the output's own magnitude: a wide row accumulates
        # more rounding, and a row of large values does so faster. The max(..., 1.0)
        # floor keeps an all-zero output from demanding an exactly-zero mean.
        eps = float(np.finfo(y.dtype).eps)
        bound = DEFAULT_THRESH * eps * y.shape[-1] * max(float(np.max(np.abs(y))), 1.0)
        worst = float(np.max(np.abs(means)))
        if worst <= bound:
            return _result(self, Verdict.PASS, "", case_id=row.case.case_id)
        return _result(
            self,
            Verdict.FAIL,
            f"largest row mean {worst:.6e} exceeds {bound:.6e}",
            case_id=row.case.case_id,
        )


class RowsHaveUnitVariance:
    """Every row of a layernorm output has approximately unit population variance.

    The complement of `RowsHaveZeroMean`: together they pin the output to the
    normalized family without naming a single expected value, which is the
    declarative arm's whole shape. A kernel that centers but never divides passes
    the mean check and fails this one.

    An all-zero output ABSTAINS rather than failing. Layernorm of a constant row is
    identically zero — variance zero, not one — because `eps` dominates the
    denominator. That is correct behaviour, and the ladder reaches it at the (1, 1)
    and (17, 1) rungs. Scoring those rows would fail every correct kernel on two of
    nine rungs, manufacturing a false-positive rate out of the corpus.

    That abstention is a second instance of the ladder's known detection deflation
    (`CLAUDE.md` obligation 5), and any absolute rate reported for layernorm must
    subtract it exactly as softmax's does.
    """

    name = "rows_have_unit_variance"
    tier = TIER_PORTABLE
    tolerance_free = False

    def check(self, row: ExecutionResult) -> PropertyResult:
        if not _usable(row):
            return _result(
                self, Verdict.INCONCLUSIVE, _unusable_detail(row), case_id=row.case.case_id
            )
        y = row.outputs["y"]
        unjudgeable = _unjudgeable(y)
        if unjudgeable:
            return _result(self, Verdict.INCONCLUSIVE, unjudgeable, case_id=row.case.case_id)
        variances = np.var(np.asarray(y, dtype=np.float64), axis=-1)
        # A row that is identically zero came from a constant input; see the
        # docstring. Judging it would fail every correct kernel on those rungs.
        judged = variances[~np.isclose(variances, 0.0, atol=1e-12)]
        if judged.size == 0:
            return _result(
                self,
                Verdict.INCONCLUSIVE,
                "every row had zero variance, which a constant input makes correct",
                case_id=row.case.case_id,
            )
        eps = float(np.finfo(y.dtype).eps)
        bound = DEFAULT_THRESH * eps * y.shape[-1]
        worst = float(np.max(np.abs(judged - 1.0)))
        if worst <= bound:
            return _result(self, Verdict.PASS, "", case_id=row.case.case_id)
        return _result(
            self,
            Verdict.FAIL,
            f"largest row variance deviation {worst:.6e} exceeds {bound:.6e}",
            case_id=row.case.case_id,
        )
```

Register both in the module's property registry alongside the existing entries, so `acceptance.yaml` can name them by string.

- [ ] **Step 4: Write the contract**

Create `kernels/tasks/layernorm/acceptance.yaml`:

```yaml
task_id: layernorm
version: 1

criteria:
  - id: OUTPUTS_ARE_FINITE
    description: no NaN or infinity reaches the output
    check:
      type: property
      property: outputs_are_finite

  - id: ROWS_HAVE_ZERO_MEAN
    description: each normalized row sums to approximately zero
    check:
      type: property
      property: rows_have_zero_mean

  - id: ROWS_HAVE_UNIT_VARIANCE
    description: each normalized row has approximately unit population variance
    check:
      type: property
      property: rows_have_unit_variance

  - id: SHIFT_INVARIANCE
    description: adding a per-row constant to the input leaves the output unchanged
    check:
      type: property
      property: shift_invariance
```

- [ ] **Step 5: Verify the properties can fail, then fill the record**

Run: `pytest tests/unit/props/test_properties.py -v`
Expected: PASS.

Confirm each property is a unique catcher: an output that is centered but unscaled must fail `RowsHaveUnitVariance` and pass `RowsHaveZeroMean`; an output offset by a constant must fail `RowsHaveZeroMean`. The two tests above assert exactly that pairing.

Run: `pytest -m "not gpu" -q`
Expected: PASS, no failures and no warnings.

Fill the **Declarative arm** column and the Notes section of the measurement record — in particular whether stating the properties was harder or easier than writing the reference, and whether any threshold had to be derived rather than guessed.

- [ ] **Step 6: Lint and commit**

```bash
ruff check src tests
git add src/autokernel_pbt/props/properties.py kernels/tasks/layernorm docs/measurements tests
scripts/git_commit_clean.sh -m "feat: add the layernorm property set and contract" -m "Zero mean and unit variance together pin the output to the normalized family without naming a single expected value, which is the declarative arm's whole shape. Neither is tolerance-free: a float sum of n centered values is not exactly zero, and claiming otherwise would inflate the count the project's sharpest claim rests on." -m "The variance property abstains on an all-zero output rather than failing it. Layernorm of a constant row is identically zero because eps dominates the denominator -- correct behaviour that the ladder reaches at two of its nine rungs, so judging it would manufacture a false-positive rate out of the corpus."
```

---

### Task 8: Close out the feature

**Files:**
- Modify: `specs/README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Mark the feature implemented**

In `specs/README.md`, change the 0006 row's status to `implemented`.

- [ ] **Step 2: Update the open obligations**

In `CLAUDE.md`, rewrite the "Open obligations" section to reflect what this plan closed and what it did not:

```markdown
## Open obligations

Obligations 1, 2 and 4 were discharged by phase 1.5. Acceptance criteria for the declarative
and hybrid arms, and wiring `HybridOracle` into the driver, were discharged by feature 0006.
What remains:

1. **`elapsed_s` is recorded but not yet fair.** Arm order is now randomized per run, so the
   bias is no longer systematic, but a *single* run's value still must not be quoted as a
   cost-per-bug denominator. The metrics phase needs repeated timing.
2. **Partial abstention is undetectable.** The driver refuses an arm that is INCONCLUSIVE on
   *every* group, but an arm that abstains on some cannot be told from one that honestly could
   not judge them — abstention is a legitimate answer, so only the degenerate case is decidable.
   `RowsHaveUnitVariance` is now a deliberate instance of legitimate abstention.
3. **The ladder deflates absolute detection, in two places now.** Degenerate shapes `(1,1)` and
   `(17,1)` make softmax identically 1.0 (measured end to end: 7/9 = 0.778 for both arms
   against `unnormalized_softmax`), and layernorm's variance property abstains on the same
   rungs for the same reason. It deflates every arm equally, so arm-vs-arm stays unbiased, but
   any absolute rate is understated by that constant and the paper must say so.
4. **Authoring cost for layernorm is `n = 1`**
   (`docs/measurements/2026-08-16-layernorm-authoring-cost.md`), with its threats recorded.
   Extend it before the number is reported.
5. `harness/correctness.py` still carries the five-stage skeleton the parent design §11 says
   the property layer replaces. It is load-bearing for features 0001 and 0002 and their
   acceptance criteria, so retiring it means retiring a feature — a scope decision, not cleanup.
```

- [ ] **Step 3: Verify the whole gate**

Run: `ruff check src tests`
Expected: `All checks passed!`

Run: `pytest -m "not gpu" -q --cov=autokernel_pbt --cov-report=term --cov-fail-under=95`
Expected: PASS, coverage at or above 95%.

Run: `pytest tests/spec/ -v`
Expected: PASS — every 0006 criterion resolves to a collectable test.

- [ ] **Step 4: Commit**

```bash
git add specs/README.md CLAUDE.md
scripts/git_commit_clean.sh -m "docs: close out feature 0006" -m "Four arms are driven, the declarative and hybrid arms have criteria, layernorm is on the ladder with its authoring cost measured, and a case group is regenerable from its spec. What remains is timing fairness, abstention semantics, the ladder's known deflation -- now in two places -- and the correctness skeleton whose retirement is a scope decision."
```

---

## Definition of Done

- [ ] All ten feature 0006 criteria resolve to collectable tests, and `tests/spec/` is green
- [ ] `pytest -m "not gpu"` passes with no failures and no warnings
- [ ] `ruff check src tests` passes
- [ ] Coverage is at or above 95%
- [ ] `run_task` drives four arms, in an order that varies with the seed
- [ ] A launch error is `INCONCLUSIVE` in every one of the four arms, never `FAIL`
- [ ] Every score row from every arm is keyed by case group
- [ ] A case group is regenerable byte-identically from its recorded spec
- [ ] The layernorm property set passes the real reference across all nine ladder rungs
- [ ] The layernorm authoring-cost record is filled for both arms, or explicitly blank where unmeasured

## Explicitly Out of Scope

- The mutation corpus and the four metrics (phase 2b)
- Any shrinking **algorithm** — only the `CaseSpec` representation
- Repeated timing for a fair cost-per-bug denominator (phase 2b)
- Deciding what partial abstention means
- Any change to phase 1.5's storage layout, driver join, or score table
- CUDA / Triton / NKI backends and tier-2 telemetry (phase 3)
- Attention, GEMM, fused kernels, rowsum; KernelBench integration
- Retiring `harness/correctness.py`
