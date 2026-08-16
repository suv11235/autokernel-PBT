# Phase 1.5: Measurable Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the four research metrics computable from recorded artifacts, and compose the pipeline once in `src/` instead of twice in test files.

**Architecture:** Three gaps close in dependency order. First the execution table learns *which kernel* produced a row and whether that kernel was known-broken — without which no detection rate can be computed. Then `PropertyResult` gets a persistence path, so verdicts survive the process that produced them. Then a single `run_task` driver in `src/` composes generate → execute → persist → score, replacing the two divergent copies in test files and discharging the "one corpus, both arms" obligation the fairness criterion documents but cannot enforce.

**Tech Stack:** Python 3.10+, NumPy, PyArrow/Parquet, safetensors, pytest.

**Why now:** Both schema changes are cheap against an empty corpus and expensive after the first paid hardware run, which is exactly what the design's risk table warns about ("Treat Phase B schema as the highest-risk interface; over-capture rather than under-capture"). Nothing has been recorded yet.

**Prior context:** `docs/superpowers/specs/2026-08-14-kernel-property-oracle-layer-design.md`, and `CLAUDE.md` for the module contracts. Phase 1 is complete: 386 tests, 9/9 acceptance criteria, `ruff check src tests` and a 95% coverage gate both enforced in CI.

---

## File Structure

| File | Change | Responsibility |
|------|--------|----------------|
| `src/autokernel_pbt/props/table.py` | Modify | Gains `kernel_id`/`kernel_is_broken` columns on the row schema |
| `src/autokernel_pbt/props/backends/base.py` | Modify | `ExecutionResult` carries the kernel identity the backend was given |
| `src/autokernel_pbt/props/scores.py` | Create | Persists `list[PropertyResult]` beside the execution table |
| `src/autokernel_pbt/props/driver.py` | Create | `run_task` — the single composition of the pipeline |
| `specs/features/0005-measurable-runs/spec.md` | Create | Feature spec |
| `specs/features/0005-measurable-runs/acceptance.yaml` | Create | Machine-checkable criteria |
| `tests/spec/test_0005_measurable_runs.py` | Create | Traceability, mirroring `test_0004_property_layer.py` |
| `tests/unit/props/test_scores.py` | Create | Score-table round trip |
| `tests/integration/test_driver.py` | Create | Driver end-to-end |

Task 0 writes the spec first and starts red, per `specs/README.md` and `docs/adr/0001-sdd-tdd.md`.

---

## Conventions (from CLAUDE.md — do not rediscover)

- `scripts/git_commit_clean.sh` only. **`git commit` and `git commit --amend` are forbidden.**
- Verify `git branch --show-current` is non-empty after every commit.
- `filterwarnings = ["error"]` — any warning is a test failure.
- `ruff check src tests` must pass; coverage must stay ≥95%.
- Every `PropertyResult` carries **exactly one** of `case_id`/`group_id`.
- Bad **data** → `INCONCLUSIVE`; bad **call** → raise. The line is whether a re-run costs hardware time.
- Each assertion must be the **unique** catcher for at least one saboteur. Verify by deleting each in turn.

---

### Task 0: Spec and acceptance criteria (red)

**Files:**
- Create: `specs/features/0005-measurable-runs/spec.md`
- Create: `specs/features/0005-measurable-runs/acceptance.yaml`
- Create: `tests/spec/test_0005_measurable_runs.py`
- Modify: `specs/README.md` (feature index)

- [ ] **Step 1: Write the spec**

Create `specs/features/0005-measurable-runs/spec.md`:

```markdown
# Feature 0005: Measurable runs

## Problem

Phase 1 built every layer but recorded too little to measure. Four of the five research metrics
cannot be computed from the persisted artifacts:

- **Bug-catching power** and **false-positive rate** need to join a row to *which kernel produced
  it* and *whether that kernel was known-broken*. The execution table records neither.
- **Cost-per-bug** needs oracle evaluation timed. Only kernel wall time is recorded.
- The **tolerance-free split** is correct in memory but `PropertyResult` is never persisted.

Separately, the pipeline is composed only in test code, twice, and the two copies already differ.

## Scope

1. `ExecutionResult` and the Parquet schema carry `kernel_id` and `kernel_is_broken`.
2. A score table persists `list[PropertyResult]` beside the execution table, joinable by
   `case_id`/`group_id`, and records which oracle arm produced each result and how long that arm took.
3. A `run_task` driver in `src/` composes generate → execute → persist → score for one task and
   one kernel, and is the only such composition.

## Non-goals

- Metric *computation* — this feature makes the numbers derivable, not derived.
- The mutation corpus (Phase 2).
- Device backends and tier-2 telemetry (Phase 3).

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
```

- [ ] **Step 2: Write the acceptance criteria**

Create `specs/features/0005-measurable-runs/acceptance.yaml`:

```yaml
feature_id: "0005"
feature_name: measurable-runs
version: 1

criteria:
  - id: KERNEL_IDENTITY
    description: a recorded row identifies which kernel produced it and whether it was broken
    check:
      type: unit_test
      test: tests/unit/props/test_table.py::test_kernel_identity_round_trips

  - id: SCORES_PERSIST
    description: property results survive a round trip and rejoin their execution rows
    check:
      type: unit_test
      test: tests/unit/props/test_scores.py::test_scores_round_trip_and_rejoin_their_rows

  - id: ARM_ATTRIBUTION
    description: each persisted result records which oracle arm produced it
    check:
      type: unit_test
      test: tests/unit/props/test_scores.py::test_each_result_records_its_arm

  - id: ORACLE_COST_RECORDED
    description: each arm's evaluation time is recorded so cost-per-bug is derivable
    check:
      type: unit_test
      test: tests/unit/props/test_scores.py::test_arm_elapsed_is_recorded

  - id: ONE_DRIVER
    description: the driver composes the pipeline and both arms score the same recorded corpus
    check:
      type: unit_test
      test: tests/integration/test_driver.py::test_both_arms_score_the_same_recorded_corpus

  - id: DETECTION_IS_DERIVABLE
    description: a detection rate can be computed from the persisted artifacts alone
    check:
      type: unit_test
      test: tests/integration/test_driver.py::test_detection_rate_is_computable_from_disk

  - id: CONTRACT_DRIVES_THE_ARM
    description: the driver builds the declarative arm from the kernel contract, not a hardcoded set
    check:
      type: unit_test
      test: tests/integration/test_driver.py::test_the_declarative_arm_comes_from_the_contract
```

- [ ] **Step 3: Write the spec test**

Create `tests/spec/test_0005_measurable_runs.py`:

```python
"""Spec-derived acceptance tests (feature 0005).

Mirrors tests/spec/test_0004_property_layer.py: every criterion must name a test
that exists and is collectable, so a criterion cannot certify a test nobody wrote.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ACCEPTANCE = "specs/features/0005-measurable-runs/acceptance.yaml"

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
def test_0005_acceptance_file_is_wellformed(repo_root: Path):
    data = yaml.safe_load((repo_root / ACCEPTANCE).read_text())
    assert data["feature_id"] == "0005"
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
def test_0005_every_criterion_names_an_existing_file(repo_root: Path):
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
def test_0005_every_criterion_is_collectable(repo_root: Path):
    """A criterion pointing at a non-existent test node is untraceable, so it fails."""
    node_ids = [
        c["check"]["test"] for c in _criteria(repo_root) if c["check"]["type"] == "unit_test"
    ]
    assert node_ids, "no unit_test criteria to collect"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *node_ids],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"pytest could not collect all criteria:\n{proc.stdout}\n{proc.stderr}"
    )
```

- [ ] **Step 4: Register the feature**

In `specs/README.md`, add beneath the 0004 row:

```markdown
| [0005](./features/0005-measurable-runs/spec.md) | Measurable runs | in progress |
```

- [ ] **Step 5: Run the spec test to verify it fails (red)**

Run: `pytest tests/spec/test_0005_measurable_runs.py -v`
Expected: `test_0005_acceptance_file_is_wellformed` PASSES; the other two FAIL, because
`tests/unit/props/test_scores.py` and `tests/integration/test_driver.py` do not exist and
`test_kernel_identity_round_trips` is not yet in `test_table.py`. This red state is the point.

- [ ] **Step 6: Commit**

```bash
git add specs/features/0005-measurable-runs tests/spec/test_0005_measurable_runs.py specs/README.md
scripts/git_commit_clean.sh -m "spec: add feature 0005 measurable runs" -m "Four of five research metrics are not computable from what phase 1 records. Criteria pin the schema and driver work that makes them derivable."
```

---

### Task 1: Kernel identity on the execution row

Bug-catching power and false-positive rate both require joining a row to which kernel variant ran
and whether it was the known-broken one. Neither is recorded today, and the run directory is the
only kernel identifier.

**Files:**
- Modify: `src/autokernel_pbt/props/backends/base.py`
- Modify: `src/autokernel_pbt/props/table.py`
- Test: `tests/unit/props/test_table.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/props/test_table.py`:

```python
def test_kernel_identity_round_trips(tmp_path):
    """A row must say which kernel produced it and whether that kernel was broken.

    Without both, a detection rate cannot be computed from the table: nothing joins
    a verdict to the ground truth about the kernel under test.
    """
    result = _result("c0")
    result.kernel_id = "softmax_missing_max_subtraction"
    result.kernel_is_broken = True
    ExecutionTable(tmp_path / "run").write([result])

    row = ExecutionTable(tmp_path / "run").read()[0]
    assert row.kernel_id == "softmax_missing_max_subtraction"
    assert row.kernel_is_broken is True


def test_kernel_identity_defaults_are_recorded_not_guessed(tmp_path):
    """An unlabelled kernel round-trips as unlabelled, never as a silent False.

    `kernel_is_broken=None` means "ground truth not stated"; False means "stated
    correct". Collapsing the two would silently enlarge the correct-kernel
    denominator of the false-positive rate.
    """
    ExecutionTable(tmp_path / "run").write([_result("c0")])
    row = ExecutionTable(tmp_path / "run").read()[0]
    assert row.kernel_id == ""
    assert row.kernel_is_broken is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_table.py::test_kernel_identity_round_trips -v`
Expected: FAIL with `AttributeError: 'ExecutionResult' object has no attribute 'kernel_id'`

- [ ] **Step 3: Add the fields to ExecutionResult**

In `src/autokernel_pbt/props/backends/base.py`, add to the `ExecutionResult` dataclass, after
`error`:

```python
    # Ground truth about the kernel that produced this row. kernel_is_broken is
    # tri-state on purpose: None means "not stated", False means "stated correct".
    # Collapsing them would silently enlarge the correct-kernel denominator of the
    # false-positive rate.
    kernel_id: str = ""
    kernel_is_broken: bool | None = None
```

- [ ] **Step 4: Add the columns to the table schema**

In `src/autokernel_pbt/props/table.py`, add to `SCHEMA` after `("error", pa.string())`:

```python
        ("kernel_id", pa.string()),
        ("kernel_is_broken", pa.bool_()),
```

In `_record`, after the existing `record["error"] = ...` line:

```python
        record["kernel_id"] = result.kernel_id
        record["kernel_is_broken"] = result.kernel_is_broken
```

In `read`, where `ExecutionResult(...)` is constructed, add:

```python
                kernel_id=record["kernel_id"],
                kernel_is_broken=record["kernel_is_broken"],
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/props/test_table.py -v`
Expected: PASS, all table tests green including the two new ones.

- [ ] **Step 6: Verify the whole suite and the gates**

Run: `pytest -m "not gpu" -q && ruff check src tests`
Expected: all green except the two known-red 0005 traceability tests; ruff clean.

- [ ] **Step 7: Mutation-verify**

Delete the `kernel_is_broken` line from `_record` and confirm `test_kernel_identity_round_trips`
fails. Then change the `read` reconstruction to `kernel_is_broken=False` and confirm
`test_kernel_identity_defaults_are_recorded_not_guessed` fails. Restore both; confirm
`git status` is clean. Report both counts.

- [ ] **Step 8: Commit**

```bash
git add src/autokernel_pbt/props/backends/base.py src/autokernel_pbt/props/table.py tests/unit/props/test_table.py
scripts/git_commit_clean.sh -m "feat: record kernel identity and ground truth on execution rows" -m "Detection rate and false-positive rate both need to join a row to which kernel produced it and whether that kernel was known-broken. kernel_is_broken is tri-state so an unlabelled run cannot masquerade as a correct one."
```

---

### Task 2: The score table

`PropertyResult` is the only object holding the research output and the only data class in the
layer with no serialization path. It also carries no arm identity and no timing, so cost-per-bug
is not derivable.

**Files:**
- Create: `src/autokernel_pbt/props/scores.py`
- Test: `tests/unit/props/test_scores.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_scores.py`:

```python
"""Score table round-trip tests."""

import pytest

from autokernel_pbt.props.scores import ArmScores, ScoreTable
from autokernel_pbt.props.verdict import TIER_PORTABLE, PropertyResult, Verdict


def _results() -> list[PropertyResult]:
    return [
        PropertyResult(
            property_name="rows_sum_to_one",
            tier=TIER_PORTABLE,
            tolerance_free=False,
            verdict=Verdict.FAIL,
            detail="ratio=44.8",
            case_id="softmax-g00000-base",
        ),
        PropertyResult(
            property_name="shift_invariance",
            tier=TIER_PORTABLE,
            tolerance_free=False,
            verdict=Verdict.PASS,
            group_id="softmax-g00000",
        ),
    ]


def test_scores_round_trip_and_rejoin_their_rows(tmp_path):
    """Verdicts must survive the process that produced them, still joinable.

    A result that cannot be rejoined to its execution row is unusable for any
    per-property or per-group analysis.
    """
    table = ScoreTable(tmp_path / "run")
    table.write([ArmScores(arm="declarative", elapsed_s=0.25, results=_results())])

    arms = ScoreTable(tmp_path / "run").read()
    assert len(arms) == 1
    got = arms[0].results
    assert [r.property_name for r in got] == ["rows_sum_to_one", "shift_invariance"]
    assert got[0].case_id == "softmax-g00000-base"
    assert got[0].group_id == ""
    assert got[1].group_id == "softmax-g00000"
    assert got[1].case_id == ""
    assert got[0].verdict is Verdict.FAIL
    assert got[0].tolerance_free is False
    assert got[0].detail == "ratio=44.8"


def test_each_result_records_its_arm(tmp_path):
    """Which arm produced a verdict is the axis the whole comparison turns on."""
    table = ScoreTable(tmp_path / "run")
    table.write(
        [
            ArmScores(arm="declarative", elapsed_s=0.25, results=_results()),
            ArmScores(arm="reference", elapsed_s=0.75, results=_results()[:1]),
        ]
    )
    arms = {a.arm: a for a in ScoreTable(tmp_path / "run").read()}
    assert set(arms) == {"declarative", "reference"}
    assert len(arms["declarative"].results) == 2
    assert len(arms["reference"].results) == 1


def test_arm_elapsed_is_recorded(tmp_path):
    """Cost-per-bug needs oracle time; kernel wall time does not answer it."""
    ScoreTable(tmp_path / "run").write(
        [ArmScores(arm="reference", elapsed_s=1.5, results=_results()[:1])]
    )
    assert ScoreTable(tmp_path / "run").read()[0].elapsed_s == 1.5


def test_read_on_missing_run_returns_empty(tmp_path):
    assert ScoreTable(tmp_path / "nope").read() == []


def test_an_arm_with_no_results_is_rejected(tmp_path):
    """An arm that scored nothing would enter the denominator having judged nothing."""
    with pytest.raises(ValueError, match="no results"):
        ScoreTable(tmp_path / "run").write([ArmScores(arm="declarative", elapsed_s=0.1, results=[])])


def test_a_result_attributed_to_neither_case_nor_group_is_rejected(tmp_path):
    """Mirrors properties._result: an orphaned result cannot be rejoined."""
    orphan = PropertyResult(
        property_name="x", tier=TIER_PORTABLE, tolerance_free=True, verdict=Verdict.PASS
    )
    with pytest.raises(ValueError, match="exactly one"):
        ScoreTable(tmp_path / "run").write(
            [ArmScores(arm="declarative", elapsed_s=0.1, results=[orphan])]
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_scores.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.scores'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/scores.py`:

```python
"""Persisted oracle verdicts, stored beside the execution table.

The execution table records what a kernel *did*; this records what each arm *made
of it*. Both live under one run directory so a verdict can be rejoined to the bytes
it judged by ``case_id``/``group_id``.

Parquet, not safetensors: these rows carry no tensors and the analysis over them is
columnar aggregation — detection rate by arm, by property, by tolerance_free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from autokernel_pbt.props.verdict import PropertyResult, Verdict

SCORES_FILE = "scores.parquet"
SCORES_TMP = f".{SCORES_FILE}.tmp"

SCHEMA = pa.schema(
    [
        ("arm", pa.string()),
        ("elapsed_s", pa.float64()),
        ("property_name", pa.string()),
        ("tier", pa.int64()),
        ("tolerance_free", pa.bool_()),
        ("verdict", pa.string()),
        ("detail", pa.string()),
        ("case_id", pa.string()),
        ("group_id", pa.string()),
    ]
)


@dataclass(frozen=True)
class ArmScores:
    """One oracle arm's verdicts over one recorded corpus, and what it cost."""

    arm: str
    elapsed_s: float
    results: list[PropertyResult] = field(default_factory=list)


class ScoreTable:
    """Read/write the recorded verdicts for one run."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)

    @property
    def scores_path(self) -> Path:
        return self.run_dir / SCORES_FILE

    def write(self, arms: list[ArmScores]) -> None:
        records = []
        for arm in arms:
            if not arm.results:
                msg = f"arm {arm.arm!r} has no results; it would enter the denominator having judged nothing"
                raise ValueError(msg)
            for result in arm.results:
                if bool(result.case_id) == bool(result.group_id):
                    msg = (
                        f"result {result.property_name!r} must carry exactly one of "
                        f"case_id/group_id to be rejoinable"
                    )
                    raise ValueError(msg)
                records.append(
                    {
                        "arm": arm.arm,
                        "elapsed_s": arm.elapsed_s,
                        "property_name": result.property_name,
                        "tier": result.tier,
                        "tolerance_free": result.tolerance_free,
                        "verdict": str(result.verdict),
                        "detail": result.detail,
                        "case_id": result.case_id,
                        "group_id": result.group_id,
                    }
                )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        staged = self.run_dir / SCORES_TMP
        try:
            pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), staged)
            staged.replace(self.scores_path)
        finally:
            staged.unlink(missing_ok=True)

    def read(self) -> list[ArmScores]:
        if not self.scores_path.exists():
            return []
        by_arm: dict[str, ArmScores] = {}
        for record in pq.read_table(self.scores_path).to_pylist():
            arm = by_arm.setdefault(
                record["arm"], ArmScores(arm=record["arm"], elapsed_s=record["elapsed_s"])
            )
            arm.results.append(
                PropertyResult(
                    property_name=record["property_name"],
                    tier=record["tier"],
                    tolerance_free=record["tolerance_free"],
                    verdict=Verdict(record["verdict"]),
                    detail=record["detail"],
                    case_id=record["case_id"],
                    group_id=record["group_id"],
                )
            )
        return list(by_arm.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/props/test_scores.py -v`
Expected: PASS, 6 passed.

- [ ] **Step 5: Mutation-verify each guard is a unique catcher**

Delete the empty-results guard, then the exactly-one-of guard, then change
`str(result.verdict)` to `result.verdict.name`. After each, run
`pytest tests/unit/props/test_scores.py -q` and record which tests fail. Each guard must fail
**precisely its own case and no others**. Restore after each; confirm `git status` clean.
Report the three counts.

- [ ] **Step 6: Run the whole suite and gates**

Run: `pytest -m "not gpu" -q && ruff check src tests`
Expected: green apart from the 0005 traceability tests; ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/autokernel_pbt/props/scores.py tests/unit/props/test_scores.py
scripts/git_commit_clean.sh -m "feat: persist oracle verdicts beside the execution table" -m "PropertyResult was the only object holding the research output and the only data class with no serialization path. Arm identity and elapsed time are recorded with it, so detection rate by arm and cost-per-bug both become derivable from disk."
```

---

### Task 3: The driver

The pipeline is composed only in test code, twice, and the two copies already differ. The fairness
criterion's docstring hands the "one corpus, both arms" obligation to whoever writes the driver;
this discharges it.

**Files:**
- Create: `src/autokernel_pbt/props/driver.py`
- Test: `tests/integration/test_driver.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_driver.py`:

```python
"""Driver end-to-end tests."""

import numpy as np
import pytest

from autokernel_pbt.props.driver import run_task
from autokernel_pbt.props.scores import ScoreTable
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference
from autokernel_pbt.props.verdict import Verdict


def correct_softmax(x):
    s = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(s)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(x.dtype)


def unnormalized_softmax(x):
    return np.exp(x - np.max(x, axis=-1, keepdims=True)).astype(x.dtype)


@pytest.mark.integration
def test_both_arms_score_the_same_recorded_corpus(tmp_path, repo_root):
    """The obligation the fairness criterion documents but cannot itself enforce.

    A test passing one dict to two arms proves those two arms agree. It cannot prove
    the driver hands every arm the same corpus, because the driver did not exist.
    """
    run_task(
        task=SOFTMAX,
        kernel=correct_softmax,
        reference_fn=softmax_reference,
        repo_root=repo_root,
        run_dir=tmp_path / "run",
        n_groups=6,
        seed=42,
        kernel_id="correct_softmax",
        kernel_is_broken=False,
    )

    rows = ExecutionTable(tmp_path / "run").read()
    arms = ScoreTable(tmp_path / "run").read()
    assert {a.arm for a in arms} == {"reference", "declarative"}

    recorded_cases = {r.case.case_id for r in rows}
    recorded_groups = {r.case.group_id for r in rows}
    for arm in arms:
        judged = {r.case_id for r in arm.results if r.case_id}
        judged_groups = {r.group_id for r in arm.results if r.group_id}
        assert judged <= recorded_cases, f"{arm.arm} judged a case that was never recorded"
        assert judged_groups <= recorded_groups


@pytest.mark.integration
def test_detection_rate_is_computable_from_disk(tmp_path, repo_root):
    """The point of the whole feature: a number, from artifacts, with no re-run."""
    for kernel, kernel_id, broken in [
        (correct_softmax, "correct_softmax", False),
        (unnormalized_softmax, "unnormalized_softmax", True),
    ]:
        run_task(
            task=SOFTMAX,
            kernel=kernel,
            reference_fn=softmax_reference,
        repo_root=repo_root,
            run_dir=tmp_path / kernel_id,
            n_groups=6,
            seed=42,
            kernel_id=kernel_id,
            kernel_is_broken=broken,
        )

    def caught(run_dir):
        arms = ScoreTable(run_dir).read()
        declarative = next(a for a in arms if a.arm == "declarative")
        return any(r.verdict is Verdict.FAIL for r in declarative.results)

    assert caught(tmp_path / "unnormalized_softmax"), "broken kernel not detected"
    assert not caught(tmp_path / "correct_softmax"), "correct kernel produced a false positive"

    rows = ExecutionTable(tmp_path / "unnormalized_softmax").read()
    assert all(r.kernel_is_broken is True for r in rows)
    assert all(r.kernel_id == "unnormalized_softmax" for r in rows)


@pytest.mark.integration
def test_the_kernel_is_executed_exactly_once_per_case(tmp_path, repo_root):
    """Both arms replay one execution; neither re-runs the kernel."""
    calls = {"n": 0}

    def counting(x):
        calls["n"] += 1
        return correct_softmax(x)

    run_task(
        task=SOFTMAX,
        kernel=counting,
        reference_fn=softmax_reference,
        repo_root=repo_root,
        run_dir=tmp_path / "run",
        n_groups=4,
        seed=1,
        kernel_id="counting",
        kernel_is_broken=False,
    )
    assert calls["n"] == len(ExecutionTable(tmp_path / "run").read())


@pytest.mark.integration
def test_oracle_time_is_recorded_for_every_arm(tmp_path):
    run_task(
        task=SOFTMAX,
        kernel=correct_softmax,
        reference_fn=softmax_reference,
        repo_root=repo_root,
        run_dir=tmp_path / "run",
        n_groups=4,
        seed=1,
        kernel_id="correct_softmax",
        kernel_is_broken=False,
    )
    for arm in ScoreTable(tmp_path / "run").read():
        assert arm.elapsed_s > 0.0, f"{arm.arm} recorded no elapsed time"


@pytest.mark.integration
def test_the_declarative_arm_comes_from_the_contract(tmp_path, repo_root):
    """The spec is the oracle — so the driver must read it, not a hardcoded tuple.

    Removing a criterion from a copy of the contract must change what the driver
    scores. Without this, `contract.py` stays exercised only by its own tests and
    the spec-as-oracle claim is untested in production code.
    """
    import shutil

    import yaml

    full = run_and_read_property_names(tmp_path / "full", repo_root)
    assert "rows_sum_to_one" in full

    trimmed_root = tmp_path / "trimmed_root"
    shutil.copytree(repo_root / "kernels", trimmed_root / "kernels")
    path = trimmed_root / "kernels" / "tasks" / "softmax" / "acceptance.yaml"
    data = yaml.safe_load(path.read_text())
    data["criteria"] = [
        c for c in data["criteria"] if c["check"]["property"] != "rows_sum_to_one"
    ]
    path.write_text(yaml.safe_dump(data, sort_keys=False))

    trimmed = run_and_read_property_names(tmp_path / "trimmed", trimmed_root)
    assert "rows_sum_to_one" not in trimmed, "the driver ignored the contract"


def run_and_read_property_names(run_dir, root):
    run_task(
        task=SOFTMAX,
        kernel=correct_softmax,
        reference_fn=softmax_reference,
        repo_root=root,
        run_dir=run_dir,
        n_groups=2,
        seed=3,
        kernel_id="correct_softmax",
        kernel_is_broken=False,
    )
    arms = {a.arm: a for a in ScoreTable(run_dir).read()}
    return {r.property_name for r in arms["declarative"].results}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_driver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.driver'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/driver.py`:

```python
"""The one composition of the pipeline: generate, execute, persist, score.

Phase 1 assembled this twice in test files and the two copies drifted. Every arm
scores rows read back from the persisted table rather than the in-memory results,
so the replay path is the one under test — and the arms provably share a corpus,
which a test passing one dict to two arms cannot establish.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.contract import load_contract, oracle_from_contract
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.oracle import ReferenceOracle
from autokernel_pbt.props.scores import ArmScores, ScoreTable
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import Task

KERNEL_TASKS_DIR = Path("kernels") / "tasks"
CONTRACT_FILENAME = "acceptance.yaml"


def contract_for(task: Task, repo_root: Path) -> Path:
    return repo_root / KERNEL_TASKS_DIR / task.task_id / CONTRACT_FILENAME


def run_task(
    *,
    task: Task,
    kernel: Callable[..., np.ndarray],
    reference_fn: Callable[..., np.ndarray],
    run_dir: Path | str,
    repo_root: Path,
    n_groups: int,
    seed: int,
    kernel_id: str,
    kernel_is_broken: bool | None = None,
) -> None:
    """Record one kernel's executions for one task, then score them with both arms.

    The declarative arm is built from the task's ``acceptance.yaml`` rather than a
    hand-assembled property tuple. That is the point of the contract mechanism —
    writing the spec is writing the oracle — and it is what stops the driver from
    silently applying softmax's laws to a task that does not obey them.
    """
    groups = Generator(task.domain, seed=seed).generate(n_groups)
    backend = NumpyBackend()

    results = []
    for group in groups:
        for case in group.cases:
            result = backend.run(kernel, case)
            result.kernel_id = kernel_id
            result.kernel_is_broken = kernel_is_broken
            results.append(result)
    ExecutionTable(run_dir).write(results)

    # Score the persisted rows, not the in-memory ones: the replay path is what the
    # research claim rests on, so it is the path that must be exercised.
    recorded = ExecutionTable(run_dir).read_groups()

    declarative = oracle_from_contract(load_contract(contract_for(task, repo_root)))
    arms = [
        ("reference", ReferenceOracle(reference_fn=reference_fn)),
        ("declarative", declarative),
    ]
    scored = []
    for name, oracle in arms:
        start = time.perf_counter()
        rows = [r for group_rows in recorded.values() for r in oracle.evaluate(group_rows)]
        elapsed = time.perf_counter() - start
        scored.append(ArmScores(arm=name, elapsed_s=elapsed, results=rows))
    ScoreTable(run_dir).write(scored)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_driver.py -v`
Expected: PASS, 4 passed.

- [ ] **Step 5: Verify the traceability tests flip green**

Run: `pytest tests/spec/ -v`
Expected: all spec tests pass, including the three 0005 ones — every criterion now resolves.

- [ ] **Step 6: Run the whole suite and gates**

Run: `pytest -m "not gpu" -q && ruff check src tests`
Expected: all green, warning-free; ruff clean. Then
`pytest -m "not gpu" -q --cov=autokernel_pbt --cov-fail-under=95` must also pass.

- [ ] **Step 7: Mutation-verify the corpus-sharing claim**

Make the declarative arm score a *freshly generated* corpus instead of the recorded one, and
confirm `test_both_arms_score_the_same_recorded_corpus` fails. Then make `run_task` execute the
kernel a second time for the second arm and confirm
`test_the_kernel_is_executed_exactly_once_per_case` fails. Restore both; confirm `git status`
clean. Report both counts.

- [ ] **Step 8: Update the feature index**

In `specs/README.md`, change the 0005 row status from `in progress` to `implemented`.

- [ ] **Step 9: Commit**

```bash
git add src/autokernel_pbt/props/driver.py tests/integration/test_driver.py specs/README.md
scripts/git_commit_clean.sh -m "feat: compose the pipeline once in a driver" -m "Both arms now score rows read back from the persisted table, so the replay path is the one exercised and the arms provably share a corpus. Discharges the obligation assert_replay_fairness documents but cannot itself enforce."
```

---

## Definition of Done

- [ ] `pytest -m "not gpu"` green and warning-free
- [ ] `ruff check src tests` clean
- [ ] `pytest --cov=autokernel_pbt --cov-fail-under=95` passes
- [ ] All six 0005 criteria resolve; `tests/spec/` fully green
- [ ] A detection rate is computable from `rows.parquet` + `scores.parquet` with no re-execution
- [ ] `src/autokernel_pbt/props/driver.py` is the only composition of the pipeline
- [ ] `contract.py` is imported by production code, not only by its own tests

## Explicitly Out of Scope

- Metric *computation* and reporting — this makes the numbers derivable, not derived
- The mutation corpus and the four metrics themselves (Phase 2)
- CUDA/Triton/NKI backends and tier-2 telemetry (Phase 3)
- Acceptance criteria for the declarative and hybrid arms — worth doing, but they belong with the
  Phase 2 measurement work that will exercise those arms in anger
- Multi-task driving; `run_task` handles one task and one kernel by design
