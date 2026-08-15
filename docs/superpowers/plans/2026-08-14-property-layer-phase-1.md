# Property Layer Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CPU-only core of the property layer — generate a deterministic case set, execute it, persist every execution, and score the recorded executions with two independent oracle strategies over byte-identical inputs.

**Architecture:** Batch-first record/replay. Generation (Phase A), execution (Phase B) and checking (Phase C) are separate stages joined by a persisted execution table. Oracles never drive generation, so competing oracle strategies see identical inputs. Metamorphic properties are supported by emitting *case groups* — related inputs sharing a `group_id`.

**Tech Stack:** Python 3.11+, NumPy (core dep, no torch required), safetensors for tensor payloads, PyArrow/Parquet for row metadata, pytest.

**Design doc:** `docs/superpowers/specs/2026-08-14-kernel-property-oracle-layer-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/autokernel_pbt/props/domain.py` | `TensorSpec`, `InputDomain` — serializable description of a task's input space |
| `src/autokernel_pbt/props/case.py` | `Case`, `CaseGroup` — generated inputs and their group identity |
| `src/autokernel_pbt/props/relations.py` | `Relation` implementations that derive metamorphic partners |
| `src/autokernel_pbt/props/generator.py` | `Generator` — seeded, deterministic case-group production |
| `src/autokernel_pbt/props/backends/base.py` | `Backend` protocol, `ExecutionResult` |
| `src/autokernel_pbt/props/backends/numpy_backend.py` | CPU backend |
| `src/autokernel_pbt/props/table.py` | `ExecutionTable` — persist/load rows (safetensors + Parquet) |
| `src/autokernel_pbt/props/verdict.py` | `Verdict`, `PropertyResult` |
| `src/autokernel_pbt/props/tolerance.py` | LAPACK-style normalized test ratios |
| `src/autokernel_pbt/props/properties.py` | Tier-1 property library |
| `src/autokernel_pbt/props/oracle.py` | `ReferenceOracle`, `DeclarativeOracle`, `HybridOracle` |
| `src/autokernel_pbt/props/tasks.py` | Task registry for the elementwise→reduction ladder |

| `specs/features/0004-property-oracle-layer/spec.md` | Human-readable requirement (Task 0) |
| `specs/features/0004-property-oracle-layer/acceptance.yaml` | Machine-checkable criteria (Task 0) |
| `kernels/tasks/softmax/acceptance.yaml` | Per-kernel property contract (Task 13) |

Tests mirror this under `tests/unit/props/`, with spec tests at `tests/spec/` and one integration test at `tests/integration/test_record_replay.py`.

**Spec-driven note.** This repo requires spec-before-code (`specs/README.md`, `docs/adr/0001-sdd-tdd.md`). Task 0 writes the spec and acceptance criteria first; Tasks 1–12 turn them green. Feature id **0004** is used rather than reusing the retired 0003.

`spec.md` authoring migrates to GitHub spec-kit in a later phase; `acceptance.yaml` stays hand-owned permanently, because spec-kit produces natural-language artifacts and has no machine-checkable acceptance mechanism. Do **not** install spec-kit in this plan.

---

### Task 0: Spec and acceptance criteria (red)

Write the spec first and prove the acceptance criteria are not yet satisfiable. The criteria name
test node ids that Tasks 1–12 will create, so this task's spec test fails until they exist — that
failure *is* the red state.

**Files:**
- Create: `specs/features/0004-property-oracle-layer/spec.md`
- Create: `specs/features/0004-property-oracle-layer/acceptance.yaml`
- Create: `tests/spec/test_0004_property_layer.py`
- Modify: `specs/README.md` (feature index table)

- [ ] **Step 1: Write the spec**

Create `specs/features/0004-property-oracle-layer/spec.md`:

```markdown
# Feature 0004: Property/oracle layer (phase 1, CPU)

## Problem

A kernel is not self-describing. Something must decide whether an execution was correct. That
decision procedure is the **oracle**. The project compares three oracle strategies, so oracle
choice must be a variable that can be changed without changing anything else — including the
inputs the kernel saw.

## Scope

Batch-first record/replay, CPU only:

1. **Generate** — deterministic, seeded case sets described by a serializable `InputDomain`.
   Related inputs are emitted as **case groups** so metamorphic properties are expressible.
2. **Execute** — one backend pass over the whole batch, persisting inputs, outputs and telemetry.
3. **Check** — oracles evaluated offline over the recorded table, never influencing generation.

Three oracle arms: reference (one strong property), declarative (many weak properties), hybrid.

Every property verdict is recorded individually and tagged with its tier and whether it required
a numerical tolerance.

## Non-goals

- Mutation corpus, metrics, and shrinking (phase 2)
- CUDA / Triton / NKI backends and tier-2 telemetry (phase 3)
- bfloat16 (no native NumPy dtype; arrives with the device backends)
- Attention and GEMM tasks; KernelBench integration

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
```

- [ ] **Step 2: Write the acceptance criteria**

Create `specs/features/0004-property-oracle-layer/acceptance.yaml`:

```yaml
feature_id: "0004"
feature_name: property-oracle-layer
version: 1

criteria:
  - id: DETERMINISTIC_GENERATION
    description: the same seed reproduces byte-identical inputs
    check:
      type: unit_test
      test: tests/unit/props/test_generator.py::test_same_seed_gives_identical_tensors

  - id: CASE_GROUPS
    description: related cases share a group id and are retrievable by relation
    check:
      type: unit_test
      test: tests/unit/props/test_case.py::test_group_finds_case_by_relation

  - id: TABLE_ROUND_TRIP
    description: recorded tensors survive persistence bitwise
    check:
      type: unit_test
      test: tests/unit/props/test_table.py::test_round_trip_preserves_tensors_bitwise

  - id: THREE_VALUED_VERDICT
    description: an empty property set is inconclusive, never a pass
    check:
      type: unit_test
      test: tests/unit/props/test_verdict.py::test_empty_results_are_inconclusive_not_pass

  - id: DIMENSIONLESS_TOLERANCE
    description: the reference arm uses a scale-invariant test ratio, not allclose
    check:
      type: unit_test
      test: tests/unit/props/test_tolerance.py::test_ratio_is_dimensionless_across_scale

  - id: PROPERTY_ATTRIBUTION
    description: each verdict records the property name and its tolerance-free flag
    check:
      type: unit_test
      test: tests/unit/props/test_oracle.py::test_declarative_oracle_records_tolerance_free_flag

  - id: REPLAY_FAIRNESS
    description: oracle arms score identical recorded executions
    check:
      type: unit_test
      test: tests/integration/test_record_replay.py::test_both_arms_see_byte_identical_inputs
```

- [ ] **Step 3: Write the spec test**

Create `tests/spec/test_0004_property_layer.py`:

```python
"""Spec-derived acceptance tests (feature 0004).

These assert traceability: every criterion in acceptance.yaml must name a test that
actually exists and is collectable. This is the mechanism the SDD ADR asks for.
"""

import subprocess
import sys

import pytest
import yaml

ACCEPTANCE = "specs/features/0004-property-oracle-layer/acceptance.yaml"


def _criteria(repo_root):
    data = yaml.safe_load((repo_root / ACCEPTANCE).read_text())
    return data["criteria"]


@pytest.mark.spec
def test_0004_acceptance_file_is_wellformed(repo_root):
    data = yaml.safe_load((repo_root / ACCEPTANCE).read_text())
    assert data["feature_id"] == "0004"
    ids = [c["id"] for c in data["criteria"]]
    assert ids, "acceptance.yaml declares no criteria"
    assert len(ids) == len(set(ids)), f"duplicate criterion ids: {ids}"


@pytest.mark.spec
def test_0004_every_criterion_names_an_existing_file(repo_root):
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
def test_0004_every_criterion_is_collectable(repo_root):
    """A criterion pointing at a non-existent test node is untraceable, so it fails."""
    node_ids = [
        c["check"]["test"] for c in _criteria(repo_root) if c["check"]["type"] == "unit_test"
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *node_ids],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert proc.returncode == 0, f"pytest could not collect all criteria:\n{proc.stdout}\n{proc.stderr}"
```

- [ ] **Step 4: Register the feature**

In `specs/README.md`, add a row to the feature index table beneath the 0002 row:

```markdown
| [0004](./features/0004-property-oracle-layer/spec.md) | Property/oracle layer (phase 1) | in progress |
```

- [ ] **Step 5: Run the spec test to verify it fails (red)**

Run: `pytest tests/spec/test_0004_property_layer.py -v`
Expected: `test_0004_acceptance_file_is_wellformed` PASSES; the other two FAIL, because none of
the referenced test files exist yet. This red state is the point of the task.

- [ ] **Step 6: Commit**

```bash
git add specs/features/0004-property-oracle-layer tests/spec/test_0004_property_layer.py specs/README.md
scripts/git_commit_clean.sh -m "spec: add feature 0004 property/oracle layer acceptance criteria"
```

---

### Task 1: Package scaffold and dependencies

**Files:**
- Create: `src/autokernel_pbt/props/__init__.py`
- Create: `tests/unit/props/__init__.py`
- Modify: `pyproject.toml:11-15`

- [ ] **Step 1: Add dependencies**

In `pyproject.toml`, replace the `dependencies` list with:

```toml
dependencies = [
  "numpy>=1.24",
  "pyyaml>=6.0",
  "jsonschema>=4.20",
  "safetensors>=0.4",
  "pyarrow>=15.0",
]
```

- [ ] **Step 2: Create the package**

Create `src/autokernel_pbt/props/__init__.py`:

```python
"""Property-based testing layer: generation, execution, and oracles."""
```

Create an empty `tests/unit/props/__init__.py`:

```python
```

- [ ] **Step 3: Verify install**

Run: `pip install -e ".[dev]" && python -c "import autokernel_pbt.props, safetensors, pyarrow; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/autokernel_pbt/props/__init__.py tests/unit/props/__init__.py
scripts/git_commit_clean.sh -m "feat: scaffold props package with storage deps"
```

---

### Task 2: TensorSpec and InputDomain

An `InputDomain` is the serializable description of a task's input space. It must round-trip
through a dict so a recorded run can be regenerated months later.

**Files:**
- Create: `src/autokernel_pbt/props/domain.py`
- Test: `tests/unit/props/test_domain.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_domain.py`:

```python
"""InputDomain serialization tests."""

import pytest

from autokernel_pbt.props.domain import InputDomain, TensorSpec


def test_tensor_spec_round_trips():
    spec = TensorSpec(name="x", dtype="float32", distribution="normal")
    assert TensorSpec.from_dict(spec.to_dict()) == spec


def test_domain_round_trips():
    domain = InputDomain(
        task_id="softmax",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=((4, 8), (3, 7)),
    )
    assert InputDomain.from_dict(domain.to_dict()) == domain


def test_domain_rejects_empty_shapes():
    with pytest.raises(ValueError, match="at least one shape"):
        InputDomain(task_id="t", tensors=(TensorSpec(name="x", dtype="float32"),), shapes=())


def test_domain_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="unsupported dtype"):
        TensorSpec(name="x", dtype="bfloat16")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_domain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.domain'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/domain.py`:

```python
"""Serializable description of a task's input space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Phase 1 is NumPy-only; bfloat16 has no native NumPy dtype and arrives with the
# device backends in Phase 3.
SUPPORTED_DTYPES = ("float16", "float32", "float64")
SUPPORTED_DISTRIBUTIONS = ("normal", "uniform", "zeros", "ones")


@dataclass(frozen=True)
class TensorSpec:
    """One input tensor's value distribution. Shape comes from the domain."""

    name: str
    dtype: str
    distribution: str = "normal"
    low: float = -1.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype!r}; expected one of {SUPPORTED_DTYPES}")
        if self.distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"unsupported distribution {self.distribution!r}")

    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "distribution": self.distribution,
            "low": self.low,
            "high": self.high,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorSpec:
        return cls(**data)


@dataclass(frozen=True)
class InputDomain:
    """Everything needed to regenerate a task's case set from a seed."""

    task_id: str
    tensors: tuple[TensorSpec, ...]
    shapes: tuple[tuple[int, ...], ...]
    relations: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.shapes:
            raise ValueError("domain needs at least one shape")
        if not self.tensors:
            raise ValueError("domain needs at least one tensor spec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tensors": [t.to_dict() for t in self.tensors],
            "shapes": [list(s) for s in self.shapes],
            "relations": list(self.relations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputDomain:
        return cls(
            task_id=data["task_id"],
            tensors=tuple(TensorSpec.from_dict(t) for t in data["tensors"]),
            shapes=tuple(tuple(s) for s in data["shapes"]),
            relations=tuple(data.get("relations", ())),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/test_domain.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/autokernel_pbt/props/domain.py tests/unit/props/test_domain.py
scripts/git_commit_clean.sh -m "feat: add InputDomain and TensorSpec"
```

---

### Task 3: Case and CaseGroup

`group_id` is the structural requirement that makes metamorphic properties expressible. A group
holds a base case plus zero or more derived partners.

**Files:**
- Create: `src/autokernel_pbt/props/case.py`
- Test: `tests/unit/props/test_case.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_case.py`:

```python
"""Case and CaseGroup tests."""

import numpy as np
import pytest

from autokernel_pbt.props.case import Case, CaseGroup


def _case(case_id: str, relation: str = "base") -> Case:
    return Case(
        case_id=case_id,
        group_id="g0",
        relation=relation,
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.zeros((2, 3), dtype=np.float32)},
    )


def test_group_exposes_base_case():
    group = CaseGroup(group_id="g0", cases=(_case("c0"), _case("c1", "shift")))
    assert group.base.case_id == "c0"


def test_group_finds_case_by_relation():
    group = CaseGroup(group_id="g0", cases=(_case("c0"), _case("c1", "shift")))
    assert group.by_relation("shift").case_id == "c1"


def test_group_by_relation_returns_none_when_absent():
    group = CaseGroup(group_id="g0", cases=(_case("c0"),))
    assert group.by_relation("shift") is None


def test_group_requires_exactly_one_base():
    with pytest.raises(ValueError, match="exactly one base"):
        CaseGroup(group_id="g0", cases=(_case("c0", "shift"),))


def test_group_rejects_mismatched_group_id():
    other = Case(
        case_id="c1",
        group_id="OTHER",
        relation="shift",
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.zeros((2, 3), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="group_id mismatch"):
        CaseGroup(group_id="g0", cases=(_case("c0"), other))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_case.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.case'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/case.py`:

```python
"""Generated inputs and their group identity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

BASE_RELATION = "base"


@dataclass(frozen=True)
class Case:
    """One executable input set."""

    case_id: str
    group_id: str
    relation: str
    task_id: str
    dtype: str
    shape: tuple[int, ...]
    tensors: dict[str, np.ndarray] = field(compare=False)

    def metadata(self) -> dict[str, Any]:
        """Everything except tensor payloads — this is what lands in Parquet."""
        return {
            "case_id": self.case_id,
            "group_id": self.group_id,
            "relation": self.relation,
            "task_id": self.task_id,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class CaseGroup:
    """A base case plus its metamorphic partners, sharing one group_id."""

    group_id: str
    cases: tuple[Case, ...]

    def __post_init__(self) -> None:
        bases = [c for c in self.cases if c.relation == BASE_RELATION]
        if len(bases) != 1:
            raise ValueError(f"group {self.group_id} needs exactly one base case, got {len(bases)}")
        for case in self.cases:
            if case.group_id != self.group_id:
                raise ValueError(
                    f"group_id mismatch: case {case.case_id} has {case.group_id!r}, "
                    f"group is {self.group_id!r}"
                )

    @property
    def base(self) -> Case:
        return next(c for c in self.cases if c.relation == BASE_RELATION)

    def by_relation(self, relation: str) -> Case | None:
        return next((c for c in self.cases if c.relation == relation), None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/test_case.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/autokernel_pbt/props/case.py tests/unit/props/test_case.py
scripts/git_commit_clean.sh -m "feat: add Case and CaseGroup with group identity"
```

---

### Task 4: Relations

A `Relation` derives a metamorphic partner from a base case. Each one names the property that
will consume it.

**Files:**
- Create: `src/autokernel_pbt/props/relations.py`
- Test: `tests/unit/props/test_relations.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_relations.py`:

```python
"""Relation tests."""

import numpy as np

from autokernel_pbt.props.case import Case
from autokernel_pbt.props.relations import RELATIONS, PermuteLastAxis, ShiftRows


def _base() -> Case:
    return Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="softmax",
        dtype="float32",
        shape=(2, 4),
        tensors={"x": np.arange(8, dtype=np.float32).reshape(2, 4)},
    )


def test_shift_rows_adds_per_row_constant():
    rng = np.random.default_rng(0)
    derived = ShiftRows().derive(_base(), rng)
    diff = derived.tensors["x"] - _base().tensors["x"]
    # Every element in a row shifted by the same amount.
    assert np.allclose(diff, diff[:, :1])


def test_shift_rows_sets_relation_and_group():
    derived = ShiftRows().derive(_base(), np.random.default_rng(0))
    assert derived.relation == "shift_rows"
    assert derived.group_id == "g0"
    assert derived.case_id == "c0::shift_rows"


def test_permute_last_axis_is_a_permutation():
    derived = PermuteLastAxis().derive(_base(), np.random.default_rng(0))
    assert np.array_equal(
        np.sort(derived.tensors["x"], axis=-1), np.sort(_base().tensors["x"], axis=-1)
    )


def test_permute_records_its_index_map():
    derived = PermuteLastAxis().derive(_base(), np.random.default_rng(0))
    perm = derived.tensors["__perm__"]
    assert sorted(perm.tolist()) == [0, 1, 2, 3]


def test_relations_registry_is_keyed_by_name():
    for name, factory in RELATIONS.items():
        assert factory().name == name


def test_relations_are_deterministic_for_a_seed():
    a = ShiftRows().derive(_base(), np.random.default_rng(7))
    b = ShiftRows().derive(_base(), np.random.default_rng(7))
    assert np.array_equal(a.tensors["x"], b.tensors["x"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_relations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.relations'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/relations.py`:

```python
"""Metamorphic relations: derive a partner case from a base case."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Protocol

import numpy as np

from autokernel_pbt.props.case import Case


class Relation(Protocol):
    """Derives one metamorphic partner from a base case."""

    name: str

    def derive(self, base: Case, rng: np.random.Generator) -> Case: ...


def _derived(base: Case, relation: str, tensors: dict[str, np.ndarray]) -> Case:
    return replace(
        base,
        case_id=f"{base.case_id}::{relation}",
        relation=relation,
        tensors=tensors,
    )


class ShiftRows:
    """x -> x + c, one constant per row. Consumed by shift-invariance properties."""

    name = "shift_rows"

    def derive(self, base: Case, rng: np.random.Generator) -> Case:
        tensors = dict(base.tensors)
        x = tensors["x"]
        shift = rng.normal(0.0, 1.0, size=(x.shape[0], 1)).astype(x.dtype)
        tensors["x"] = (x + shift).astype(x.dtype)
        return _derived(base, self.name, tensors)


class PermuteLastAxis:
    """Permute the last axis. Consumed by equivariance properties.

    The permutation is stored under ``__perm__`` so the property can undo it.
    """

    name = "permute_last_axis"

    def derive(self, base: Case, rng: np.random.Generator) -> Case:
        tensors = dict(base.tensors)
        x = tensors["x"]
        perm = rng.permutation(x.shape[-1])
        tensors["x"] = np.take(x, perm, axis=-1)
        tensors["__perm__"] = perm.astype(np.int64)
        return _derived(base, self.name, tensors)


RELATIONS: dict[str, Callable[[], Relation]] = {
    ShiftRows.name: ShiftRows,
    PermuteLastAxis.name: PermuteLastAxis,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/test_relations.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/autokernel_pbt/props/relations.py tests/unit/props/test_relations.py
scripts/git_commit_clean.sh -m "feat: add metamorphic relations"
```

---

### Task 5: Generator

Deterministic from a seed. Shape-first sampling: every shape in the domain is visited before any
shape repeats, because boundary shape coverage is the highest-yield, lowest-false-positive
generation strategy.

**Files:**
- Create: `src/autokernel_pbt/props/generator.py`
- Test: `tests/unit/props/test_generator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_generator.py`:

```python
"""Generator determinism and coverage tests."""

import numpy as np

from autokernel_pbt.props.domain import InputDomain, TensorSpec
from autokernel_pbt.props.generator import Generator

DOMAIN = InputDomain(
    task_id="softmax",
    tensors=(TensorSpec(name="x", dtype="float32"),),
    shapes=((2, 4), (3, 5), (1, 7)),
    relations=("shift_rows", "permute_last_axis"),
)


def test_same_seed_gives_identical_tensors():
    a = Generator(DOMAIN, seed=123).generate(4)
    b = Generator(DOMAIN, seed=123).generate(4)
    for ga, gb in zip(a, b):
        assert np.array_equal(ga.base.tensors["x"], gb.base.tensors["x"])


def test_different_seed_gives_different_tensors():
    a = Generator(DOMAIN, seed=1).generate(1)
    b = Generator(DOMAIN, seed=2).generate(1)
    assert not np.array_equal(a[0].base.tensors["x"], b[0].base.tensors["x"])


def test_every_shape_visited_before_repeat():
    groups = Generator(DOMAIN, seed=0).generate(3)
    assert {g.base.shape for g in groups} == set(DOMAIN.shapes)


def test_group_contains_base_plus_each_relation():
    groups = Generator(DOMAIN, seed=0).generate(1)
    relations = {c.relation for c in groups[0].cases}
    assert relations == {"base", "shift_rows", "permute_last_axis"}


def test_group_ids_are_unique():
    groups = Generator(DOMAIN, seed=0).generate(6)
    assert len({g.group_id for g in groups}) == 6


def test_dtype_is_honoured():
    groups = Generator(DOMAIN, seed=0).generate(1)
    assert groups[0].base.tensors["x"].dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.generator'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/generator.py`:

```python
"""Phase A: deterministic, seeded case-group generation."""

from __future__ import annotations

import numpy as np

from autokernel_pbt.props.case import BASE_RELATION, Case, CaseGroup
from autokernel_pbt.props.domain import InputDomain, TensorSpec
from autokernel_pbt.props.relations import RELATIONS


def _sample(spec: TensorSpec, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    dtype = spec.numpy_dtype()
    if spec.distribution == "normal":
        values = rng.normal(0.0, 1.0, size=shape)
    elif spec.distribution == "uniform":
        values = rng.uniform(spec.low, spec.high, size=shape)
    elif spec.distribution == "zeros":
        values = np.zeros(shape)
    elif spec.distribution == "ones":
        values = np.ones(shape)
    else:  # pragma: no cover - guarded by TensorSpec.__post_init__
        raise ValueError(f"unsupported distribution {spec.distribution!r}")
    return values.astype(dtype)


class Generator:
    """Produces case groups deterministically from a domain and a seed."""

    def __init__(self, domain: InputDomain, seed: int) -> None:
        self.domain = domain
        self.seed = seed

    def generate(self, n_groups: int) -> list[CaseGroup]:
        rng = np.random.default_rng(self.seed)
        groups: list[CaseGroup] = []
        for index in range(n_groups):
            # Shape-first: cycle through every shape before repeating any.
            shape = self.domain.shapes[index % len(self.domain.shapes)]
            group_id = f"{self.domain.task_id}-g{index:05d}"
            base = Case(
                case_id=f"{group_id}-base",
                group_id=group_id,
                relation=BASE_RELATION,
                task_id=self.domain.task_id,
                dtype=self.domain.tensors[0].dtype,
                shape=shape,
                tensors={t.name: _sample(t, shape, rng) for t in self.domain.tensors},
            )
            cases = [base]
            for relation_name in self.domain.relations:
                relation = RELATIONS[relation_name]()
                cases.append(relation.derive(base, rng))
            groups.append(CaseGroup(group_id=group_id, cases=tuple(cases)))
        return groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/test_generator.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/autokernel_pbt/props/generator.py tests/unit/props/test_generator.py
scripts/git_commit_clean.sh -m "feat: add deterministic case-group generator"
```

---

### Task 6: Backend protocol and NumPy backend

The backend is the only component that will ever need hardware. Its interface is fixed now so
Phase 3 can add CUDA/NKI without touching anything upstream. Telemetry is captured here even
though the CPU backend has little to report — the field must exist from the start, because
tier-2 properties cannot recover it offline.

**Files:**
- Create: `src/autokernel_pbt/props/backends/__init__.py`
- Create: `src/autokernel_pbt/props/backends/base.py`
- Create: `src/autokernel_pbt/props/backends/numpy_backend.py`
- Test: `tests/unit/props/test_numpy_backend.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_numpy_backend.py`:

```python
"""NumPy backend tests."""

import numpy as np

from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.case import Case


def _case() -> Case:
    return Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="relu",
        dtype="float32",
        shape=(4,),
        tensors={"x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)},
    )


def test_successful_run_reports_ok():
    result = NumpyBackend().run(lambda x: np.maximum(x, 0.0), _case())
    assert result.status == "ok"
    assert np.array_equal(result.outputs["y"], np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32))


def test_exception_is_captured_not_raised():
    def boom(x):
        raise RuntimeError("kernel exploded")

    result = NumpyBackend().run(boom, _case())
    assert result.status == "launch_error"
    assert "kernel exploded" in result.error


def test_telemetry_records_backend_name():
    result = NumpyBackend().run(lambda x: x, _case())
    assert result.telemetry["backend"] == "numpy"


def test_telemetry_records_wall_time():
    result = NumpyBackend().run(lambda x: x, _case())
    assert result.telemetry["wall_ms"] >= 0.0


def test_perm_helper_tensor_is_not_passed_to_kernel():
    case = _case()
    case.tensors["__perm__"] = np.array([0, 1, 2, 3], dtype=np.int64)
    result = NumpyBackend().run(lambda x: x, case)
    assert result.status == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_numpy_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.backends'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/backends/__init__.py`:

```python
"""Execution backends."""
```

Create `src/autokernel_pbt/props/backends/base.py`:

```python
"""Phase B: the execution boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from autokernel_pbt.props.case import Case

# Tensors whose names start with this prefix are generator bookkeeping (e.g. a
# recorded permutation) and are never passed to the kernel.
HELPER_PREFIX = "__"


@dataclass
class ExecutionResult:
    """One kernel execution. Persisted verbatim as an execution-table row."""

    case: Case
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str = ""


class Backend(Protocol):
    name: str

    def run(self, kernel: Callable[..., np.ndarray], case: Case) -> ExecutionResult: ...


def kernel_inputs(case: Case) -> dict[str, np.ndarray]:
    """Tensors the kernel actually receives, excluding generator bookkeeping."""
    return {k: v for k, v in case.tensors.items() if not k.startswith(HELPER_PREFIX)}
```

Create `src/autokernel_pbt/props/backends/numpy_backend.py`:

```python
"""CPU backend. First-class, not a stub: CI runs entirely on it."""

from __future__ import annotations

import time
import traceback
from typing import Callable

import numpy as np

from autokernel_pbt.props.backends.base import ExecutionResult, kernel_inputs
from autokernel_pbt.props.case import Case


class NumpyBackend:
    name = "numpy"

    def run(self, kernel: Callable[..., np.ndarray], case: Case) -> ExecutionResult:
        inputs = kernel_inputs(case)
        start = time.perf_counter()
        try:
            output = kernel(**inputs)
        except Exception as exc:  # noqa: BLE001 - a failing kernel is data, not an error
            return ExecutionResult(
                case=case,
                telemetry={"backend": self.name, "wall_ms": 0.0},
                status="launch_error",
                error=f"{exc}\n{traceback.format_exc()}",
            )
        wall_ms = (time.perf_counter() - start) * 1000.0
        return ExecutionResult(
            case=case,
            outputs={"y": np.asarray(output)},
            telemetry={"backend": self.name, "wall_ms": wall_ms},
            status="ok",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/test_numpy_backend.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/autokernel_pbt/props/backends tests/unit/props/test_numpy_backend.py
scripts/git_commit_clean.sh -m "feat: add backend protocol and NumPy backend"
```

---

### Task 7: ExecutionTable persistence

This is the artifact the whole architecture exists to produce. A round-trip test is the single
most important test in Phase 1: if rows do not survive persistence byte-identically, oracle
comparison is not fair.

**Files:**
- Create: `src/autokernel_pbt/props/table.py`
- Test: `tests/unit/props/test_table.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_table.py`:

```python
"""ExecutionTable round-trip tests."""

import numpy as np

from autokernel_pbt.props.backends.base import ExecutionResult
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.table import ExecutionTable


def _result(case_id: str, group_id: str = "g0", relation: str = "base") -> ExecutionResult:
    case = Case(
        case_id=case_id,
        group_id=group_id,
        relation=relation,
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.full((2, 3), 0.5, dtype=np.float32)},
    )
    return ExecutionResult(
        case=case,
        outputs={"y": np.full((2, 3), 0.25, dtype=np.float32)},
        telemetry={"backend": "numpy", "wall_ms": 1.5},
        status="ok",
    )


def test_round_trip_preserves_tensors_bitwise(tmp_path):
    table = ExecutionTable(tmp_path / "run1")
    table.write([_result("c0")])
    rows = ExecutionTable(tmp_path / "run1").read()
    assert np.array_equal(rows[0].outputs["y"], np.full((2, 3), 0.25, dtype=np.float32))
    assert np.array_equal(rows[0].case.tensors["x"], np.full((2, 3), 0.5, dtype=np.float32))


def test_round_trip_preserves_metadata(tmp_path):
    table = ExecutionTable(tmp_path / "run1")
    table.write([_result("c0")])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.case.case_id == "c0"
    assert row.case.group_id == "g0"
    assert row.case.shape == (2, 3)
    assert row.telemetry["backend"] == "numpy"
    assert row.status == "ok"


def test_grouping_reassembles_case_groups(tmp_path):
    table = ExecutionTable(tmp_path / "run1")
    table.write([_result("c0"), _result("c1", relation="shift_rows")])
    groups = ExecutionTable(tmp_path / "run1").read_groups()
    assert list(groups) == ["g0"]
    assert {r.case.relation for r in groups["g0"]} == {"base", "shift_rows"}


def test_failed_execution_round_trips(tmp_path):
    failed = _result("c0")
    failed.status = "launch_error"
    failed.error = "boom"
    failed.outputs = {}
    ExecutionTable(tmp_path / "run1").write([failed])
    row = ExecutionTable(tmp_path / "run1").read()[0]
    assert row.status == "launch_error"
    assert row.error == "boom"
    assert row.outputs == {}


def test_read_on_missing_run_returns_empty(tmp_path):
    assert ExecutionTable(tmp_path / "nope").read() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_table.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.table'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/table.py`:

```python
"""Persisted execution table: Parquet metadata plus safetensors payloads.

Tensor payloads are far too large for a JSON ledger, and the analysis over rows is
columnar aggregation, so metadata lives in Parquet and tensors in safetensors.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import load_file, save_file

from autokernel_pbt.props.backends.base import ExecutionResult
from autokernel_pbt.props.case import Case

METADATA_FILE = "rows.parquet"
TENSOR_DIR = "tensors"
INPUT_PREFIX = "in."
OUTPUT_PREFIX = "out."


class ExecutionTable:
    """Read/write the recorded executions for one run."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)

    def write(self, results: list[ExecutionResult]) -> None:
        tensor_dir = self.run_dir / TENSOR_DIR
        tensor_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for result in results:
            payload: dict[str, np.ndarray] = {}
            for name, array in result.case.tensors.items():
                payload[f"{INPUT_PREFIX}{name}"] = np.ascontiguousarray(array)
            for name, array in result.outputs.items():
                payload[f"{OUTPUT_PREFIX}{name}"] = np.ascontiguousarray(array)
            save_file(payload, str(tensor_dir / f"{result.case.case_id}.safetensors"))
            record = result.case.metadata()
            record["shape"] = json.dumps(record["shape"])
            record["telemetry"] = json.dumps(result.telemetry)
            record["status"] = result.status
            record["error"] = result.error
            records.append(record)
        pq.write_table(pa.Table.from_pylist(records), self.run_dir / METADATA_FILE)

    def read(self) -> list[ExecutionResult]:
        path = self.run_dir / METADATA_FILE
        if not path.exists():
            return []
        results = []
        for record in pq.read_table(path).to_pylist():
            payload = load_file(
                str(self.run_dir / TENSOR_DIR / f"{record['case_id']}.safetensors")
            )
            inputs = {
                k[len(INPUT_PREFIX) :]: v
                for k, v in payload.items()
                if k.startswith(INPUT_PREFIX)
            }
            outputs = {
                k[len(OUTPUT_PREFIX) :]: v
                for k, v in payload.items()
                if k.startswith(OUTPUT_PREFIX)
            }
            case = Case(
                case_id=record["case_id"],
                group_id=record["group_id"],
                relation=record["relation"],
                task_id=record["task_id"],
                dtype=record["dtype"],
                shape=tuple(json.loads(record["shape"])),
                tensors=inputs,
            )
            results.append(
                ExecutionResult(
                    case=case,
                    outputs=outputs,
                    telemetry=json.loads(record["telemetry"]),
                    status=record["status"],
                    error=record["error"],
                )
            )
        return results

    def read_groups(self) -> OrderedDict[str, list[ExecutionResult]]:
        """Rows reassembled into case groups, preserving write order."""
        groups: OrderedDict[str, list[ExecutionResult]] = OrderedDict()
        for row in self.read():
            groups.setdefault(row.case.group_id, []).append(row)
        return groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/test_table.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/autokernel_pbt/props/table.py tests/unit/props/test_table.py
scripts/git_commit_clean.sh -m "feat: add execution table persistence"
```

---

### Task 8: Verdict and PropertyResult

Three-valued verdicts are mandatory. An `INCONCLUSIVE` result must never be counted as a caught
bug, or the false-positive metric becomes meaningless.

**Files:**
- Create: `src/autokernel_pbt/props/verdict.py`
- Test: `tests/unit/props/test_verdict.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_verdict.py`:

```python
"""Verdict semantics tests."""

from autokernel_pbt.props.verdict import PropertyResult, Verdict, summarize


def _r(name: str, verdict: Verdict, tolerance_free: bool = True) -> PropertyResult:
    return PropertyResult(
        property_name=name, tier=1, tolerance_free=tolerance_free, verdict=verdict
    )


def test_all_pass_summarizes_to_pass():
    assert summarize([_r("a", Verdict.PASS), _r("b", Verdict.PASS)]) is Verdict.PASS


def test_any_fail_summarizes_to_fail():
    results = [_r("a", Verdict.PASS), _r("b", Verdict.FAIL), _r("c", Verdict.INCONCLUSIVE)]
    assert summarize(results) is Verdict.FAIL


def test_inconclusive_without_fail_summarizes_to_inconclusive():
    assert summarize([_r("a", Verdict.PASS), _r("b", Verdict.INCONCLUSIVE)]) is Verdict.INCONCLUSIVE


def test_empty_results_are_inconclusive_not_pass():
    # A property set that checked nothing has not established correctness.
    assert summarize([]) is Verdict.INCONCLUSIVE


def test_result_records_attribution_fields():
    result = _r("softmax_rows_sum_to_one", Verdict.FAIL, tolerance_free=False)
    assert result.property_name == "softmax_rows_sum_to_one"
    assert result.tier == 1
    assert result.tolerance_free is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_verdict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.verdict'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/verdict.py`:

```python
"""Three-valued verdicts and per-property attribution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class PropertyResult:
    """One property's verdict on one case or group.

    ``tier`` and ``tolerance_free`` are recorded per result so detection can be
    reported split by property tier and by whether a tolerance argument was needed.
    """

    property_name: str
    tier: int
    tolerance_free: bool
    verdict: Verdict
    detail: str = ""


def summarize(results: list[PropertyResult]) -> Verdict:
    """FAIL dominates; an empty or wholly inconclusive set is INCONCLUSIVE, never PASS."""
    if any(r.verdict is Verdict.FAIL for r in results):
        return Verdict.FAIL
    if not results or any(r.verdict is Verdict.INCONCLUSIVE for r in results):
        return Verdict.INCONCLUSIVE
    return Verdict.PASS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/test_verdict.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/autokernel_pbt/props/verdict.py tests/unit/props/test_verdict.py
scripts/git_commit_clean.sh -m "feat: add three-valued verdicts"
```

---

### Task 9: Tolerance via normalized test ratios

LAPACK-style: a dimensionless residual with one threshold across every size and precision, rather
than a hand-picked `rtol` per dtype.

**Files:**
- Create: `src/autokernel_pbt/props/tolerance.py`
- Test: `tests/unit/props/test_tolerance.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_tolerance.py`:

```python
"""Test-ratio tolerance tests."""

import numpy as np

from autokernel_pbt.props.tolerance import DEFAULT_THRESH, machine_eps, test_ratio


def test_identical_arrays_give_zero_ratio():
    x = np.ones((4, 4), dtype=np.float32)
    assert test_ratio(x, x) == 0.0


def test_ratio_is_dimensionless_across_scale():
    # Scaling both arrays by 1000 must not change the ratio.
    ref = np.linspace(1.0, 2.0, 16, dtype=np.float64).reshape(4, 4)
    cand = ref + 1e-12
    small = test_ratio(cand, ref)
    large = test_ratio(cand * 1000.0, ref * 1000.0)
    assert np.isclose(small, large, rtol=1e-6)


def test_float32_rounding_stays_under_threshold():
    ref = np.random.default_rng(0).normal(size=(64, 64))
    cand = ref.astype(np.float32).astype(np.float64)
    assert test_ratio(cand, ref, dtype=np.float32) < DEFAULT_THRESH


def test_gross_error_exceeds_threshold():
    ref = np.ones((8, 8), dtype=np.float32)
    cand = ref.copy()
    cand[0, 0] = 5.0
    assert test_ratio(cand, ref) > DEFAULT_THRESH


def test_nan_in_candidate_gives_infinite_ratio():
    ref = np.ones((4,), dtype=np.float32)
    cand = np.array([np.nan, 1.0, 1.0, 1.0], dtype=np.float32)
    assert np.isinf(test_ratio(cand, ref))


def test_zero_reference_does_not_divide_by_zero():
    ref = np.zeros((4,), dtype=np.float32)
    cand = np.zeros((4,), dtype=np.float32)
    assert np.isfinite(test_ratio(cand, ref))


def test_machine_eps_matches_numpy():
    assert machine_eps(np.float32) == np.finfo(np.float32).eps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_tolerance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.tolerance'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/tolerance.py`:

```python
"""Normalized test ratios, after the LAPACK convention (LAWN 41 s7.1.1).

A test ratio is dimensionless: it divides the residual by the scale of the problem
and by the unit roundoff, so a single threshold covers every routine, size and
precision. This replaces per-dtype ``rtol``/``atol`` guesses.
"""

from __future__ import annotations

import numpy as np

# LAPACK uses 30.0 across its entire test suite.
DEFAULT_THRESH = 30.0


def machine_eps(dtype: type | np.dtype) -> float:
    return float(np.finfo(dtype).eps)


def test_ratio(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    dtype: type | np.dtype | None = None,
) -> float:
    """‖candidate - reference‖_inf / (‖reference‖_inf * eps * n), or inf if non-finite.

    ``dtype`` selects the unit roundoff; it defaults to the candidate's dtype. Pass it
    explicitly when comparing a low-precision candidate promoted to float64.
    """
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return float("inf")
    if not np.all(np.isfinite(cand)) or not np.all(np.isfinite(ref)):
        return float("inf")

    eps = machine_eps(dtype if dtype is not None else np.asarray(candidate).dtype)
    residual = float(np.max(np.abs(cand - ref))) if cand.size else 0.0
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    # A zero reference has no scale of its own; fall back to unit scale so the
    # ratio stays finite and still measures absolute deviation in units of eps.
    scale = scale if scale > 0.0 else 1.0
    n = max(cand.shape[-1], 1) if cand.ndim else 1
    return residual / (scale * eps * n)


def within_threshold(ratio: float, thresh: float = DEFAULT_THRESH) -> bool:
    return bool(np.isfinite(ratio) and ratio < thresh)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/test_tolerance.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/autokernel_pbt/props/tolerance.py tests/unit/props/test_tolerance.py
scripts/git_commit_clean.sh -m "feat: add LAPACK-style test ratios"
```

---

### Task 10: Tier-1 property library

Two scopes: case-level properties see one row; group-level properties see a whole case group and
implement the metamorphic relations. Note which are tolerance-free — that tag drives the headline
result.

**Files:**
- Create: `src/autokernel_pbt/props/properties.py`
- Test: `tests/unit/props/test_properties.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_properties.py`:

```python
"""Tier-1 property tests."""

import numpy as np

from autokernel_pbt.props.backends.base import ExecutionResult
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.properties import (
    OutputsAreFinite,
    RowsSumToOne,
    ShiftInvariance,
    ValuesInUnitInterval,
)
from autokernel_pbt.props.verdict import Verdict


def _row(x: np.ndarray, y: np.ndarray, relation: str = "base") -> ExecutionResult:
    case = Case(
        case_id=f"c-{relation}",
        group_id="g0",
        relation=relation,
        task_id="softmax",
        dtype="float32",
        shape=x.shape,
        tensors={"x": x},
    )
    return ExecutionResult(case=case, outputs={"y": y}, telemetry={}, status="ok")


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return (e / e.sum(axis=-1, keepdims=True)).astype(x.dtype)


X = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float32)


def test_rows_sum_to_one_passes_on_correct_output():
    assert RowsSumToOne().check(_row(X, _softmax(X))).verdict is Verdict.PASS


def test_rows_sum_to_one_fails_on_unnormalized_output():
    assert RowsSumToOne().check(_row(X, np.exp(X))).verdict is Verdict.FAIL


def test_values_in_unit_interval_is_tolerance_free():
    assert ValuesInUnitInterval().tolerance_free is True


def test_values_in_unit_interval_fails_on_negative():
    bad = _softmax(X).copy()
    bad[0, 0] = -0.1
    assert ValuesInUnitInterval().check(_row(X, bad)).verdict is Verdict.FAIL


def test_outputs_are_finite_fails_on_nan():
    bad = _softmax(X).copy()
    bad[0, 0] = np.nan
    assert OutputsAreFinite().check(_row(X, bad)).verdict is Verdict.FAIL


def test_case_property_is_inconclusive_on_failed_execution():
    row = _row(X, _softmax(X))
    row.status = "launch_error"
    row.outputs = {}
    assert RowsSumToOne().check(row).verdict is Verdict.INCONCLUSIVE


def test_shift_invariance_passes_on_correct_softmax():
    shifted_x = X + np.array([[10.0], [-5.0]], dtype=np.float32)
    group = [_row(X, _softmax(X)), _row(shifted_x, _softmax(shifted_x), "shift_rows")]
    assert ShiftInvariance().check_group(group).verdict is Verdict.PASS


def test_shift_invariance_fails_on_non_invariant_kernel():
    # A kernel that forgets the max-subtraction trick is still shift invariant, so
    # instead use one that is genuinely not: plain normalization of x.
    def bad(x):
        return (x / x.sum(axis=-1, keepdims=True)).astype(x.dtype)

    shifted_x = X + np.array([[10.0], [-5.0]], dtype=np.float32)
    group = [_row(X, bad(X)), _row(shifted_x, bad(shifted_x), "shift_rows")]
    assert ShiftInvariance().check_group(group).verdict is Verdict.FAIL


def test_shift_invariance_inconclusive_without_partner():
    assert ShiftInvariance().check_group([_row(X, _softmax(X))]).verdict is Verdict.INCONCLUSIVE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_properties.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.properties'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/properties.py`:

```python
"""Tier-1 (portable/semantic) properties.

Tier-1 properties are pure functions of (inputs, outputs) and hold for any correct
implementation on any backend. ``tolerance_free`` marks the ones that need no
numerical tolerance argument at all.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from autokernel_pbt.props.backends.base import ExecutionResult
from autokernel_pbt.props.tolerance import DEFAULT_THRESH, test_ratio, within_threshold
from autokernel_pbt.props.verdict import PropertyResult, Verdict

TIER_PORTABLE = 1


class CaseProperty(Protocol):
    name: str
    tier: int
    tolerance_free: bool

    def check(self, row: ExecutionResult) -> PropertyResult: ...


class GroupProperty(Protocol):
    name: str
    tier: int
    tolerance_free: bool
    requires_relation: str

    def check_group(self, rows: list[ExecutionResult]) -> PropertyResult: ...


def _result(prop, verdict: Verdict, detail: str = "") -> PropertyResult:
    return PropertyResult(
        property_name=prop.name,
        tier=prop.tier,
        tolerance_free=prop.tolerance_free,
        verdict=verdict,
        detail=detail,
    )


def _usable(row: ExecutionResult) -> bool:
    return row.status == "ok" and "y" in row.outputs


class OutputsAreFinite:
    """No NaN or Inf anywhere in the output."""

    name = "outputs_are_finite"
    tier = TIER_PORTABLE
    tolerance_free = True

    def check(self, row: ExecutionResult) -> PropertyResult:
        if not _usable(row):
            return _result(self, Verdict.INCONCLUSIVE, f"status={row.status}")
        if not np.all(np.isfinite(row.outputs["y"])):
            return _result(self, Verdict.FAIL, "output contains NaN or Inf")
        return _result(self, Verdict.PASS)


class ValuesInUnitInterval:
    """Every output value lies in [0, 1]. Structural, no tolerance needed."""

    name = "values_in_unit_interval"
    tier = TIER_PORTABLE
    tolerance_free = True

    def check(self, row: ExecutionResult) -> PropertyResult:
        if not _usable(row):
            return _result(self, Verdict.INCONCLUSIVE, f"status={row.status}")
        y = row.outputs["y"]
        if not np.all(np.isfinite(y)):
            return _result(self, Verdict.INCONCLUSIVE, "non-finite output")
        if float(y.min()) < 0.0 or float(y.max()) > 1.0:
            return _result(self, Verdict.FAIL, f"range [{y.min()}, {y.max()}] outside [0, 1]")
        return _result(self, Verdict.PASS)


class RowsSumToOne:
    """Each row of the output sums to 1. Needs a tolerance."""

    name = "rows_sum_to_one"
    tier = TIER_PORTABLE
    tolerance_free = False

    def check(self, row: ExecutionResult) -> PropertyResult:
        if not _usable(row):
            return _result(self, Verdict.INCONCLUSIVE, f"status={row.status}")
        y = row.outputs["y"]
        if not np.all(np.isfinite(y)):
            return _result(self, Verdict.INCONCLUSIVE, "non-finite output")
        sums = y.sum(axis=-1)
        ratio = test_ratio(sums, np.ones_like(sums), dtype=y.dtype)
        if not within_threshold(ratio):
            return _result(self, Verdict.FAIL, f"row-sum test ratio {ratio:.3g} >= {DEFAULT_THRESH}")
        return _result(self, Verdict.PASS, f"ratio={ratio:.3g}")


class ShiftInvariance:
    """f(x + c) == f(x) for a per-row constant c. Metamorphic: needs the group partner."""

    name = "shift_invariance"
    tier = TIER_PORTABLE
    tolerance_free = False
    requires_relation = "shift_rows"

    def check_group(self, rows: list[ExecutionResult]) -> PropertyResult:
        base = next((r for r in rows if r.case.relation == "base"), None)
        partner = next((r for r in rows if r.case.relation == self.requires_relation), None)
        if base is None or partner is None:
            return _result(self, Verdict.INCONCLUSIVE, "group missing shift_rows partner")
        if not _usable(base) or not _usable(partner):
            return _result(self, Verdict.INCONCLUSIVE, "group contains a failed execution")
        ratio = test_ratio(partner.outputs["y"], base.outputs["y"], dtype=base.outputs["y"].dtype)
        if not within_threshold(ratio):
            return _result(self, Verdict.FAIL, f"shift test ratio {ratio:.3g} >= {DEFAULT_THRESH}")
        return _result(self, Verdict.PASS, f"ratio={ratio:.3g}")


SOFTMAX_CASE_PROPERTIES: tuple[CaseProperty, ...] = (
    OutputsAreFinite(),
    ValuesInUnitInterval(),
    RowsSumToOne(),
)
SOFTMAX_GROUP_PROPERTIES: tuple[GroupProperty, ...] = (ShiftInvariance(),)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/test_properties.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/autokernel_pbt/props/properties.py tests/unit/props/test_properties.py
scripts/git_commit_clean.sh -m "feat: add tier-1 property library"
```

---

### Task 11: Oracles

The three arms. Each takes a recorded case group and returns per-property results — never a bare
boolean, because attribution is what makes the experiment analysable.

**Files:**
- Create: `src/autokernel_pbt/props/oracle.py`
- Test: `tests/unit/props/test_oracle.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_oracle.py`:

```python
"""Oracle arm tests."""

import numpy as np

from autokernel_pbt.props.backends.base import ExecutionResult
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.oracle import DeclarativeOracle, HybridOracle, ReferenceOracle
from autokernel_pbt.props.properties import SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES
from autokernel_pbt.props.verdict import Verdict

X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(x.dtype)


def _row(x: np.ndarray, y: np.ndarray, relation: str = "base") -> ExecutionResult:
    case = Case(
        case_id=f"c-{relation}",
        group_id="g0",
        relation=relation,
        task_id="softmax",
        dtype="float32",
        shape=x.shape,
        tensors={"x": x},
    )
    return ExecutionResult(case=case, outputs={"y": y}, telemetry={}, status="ok")


def _good_group():
    shifted = X + 4.0
    return [_row(X, _softmax(X)), _row(shifted, _softmax(shifted), "shift_rows")]


def _bad_group():
    def bad(x):
        return (np.abs(x) / np.abs(x).sum(axis=-1, keepdims=True)).astype(x.dtype)

    shifted = X + 4.0
    return [_row(X, bad(X)), _row(shifted, bad(shifted), "shift_rows")]


def test_reference_oracle_passes_correct_kernel():
    oracle = ReferenceOracle(reference_fn=_softmax)
    assert oracle.summary(oracle.evaluate(_good_group())) is Verdict.PASS


def test_reference_oracle_fails_wrong_kernel():
    oracle = ReferenceOracle(reference_fn=_softmax)
    assert oracle.summary(oracle.evaluate(_bad_group())) is Verdict.FAIL


def test_reference_oracle_emits_exactly_one_property_per_row():
    oracle = ReferenceOracle(reference_fn=_softmax)
    results = oracle.evaluate(_good_group())
    assert len(results) == 2
    assert {r.property_name for r in results} == {"matches_reference"}


def test_declarative_oracle_passes_correct_kernel():
    oracle = DeclarativeOracle(SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES)
    assert oracle.summary(oracle.evaluate(_good_group())) is Verdict.PASS


def test_declarative_oracle_fails_wrong_kernel():
    oracle = DeclarativeOracle(SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES)
    assert oracle.summary(oracle.evaluate(_bad_group())) is Verdict.FAIL


def test_declarative_oracle_attributes_each_property():
    oracle = DeclarativeOracle(SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES)
    names = {r.property_name for r in oracle.evaluate(_good_group())}
    assert "rows_sum_to_one" in names
    assert "shift_invariance" in names


def test_declarative_oracle_records_tolerance_free_flag():
    oracle = DeclarativeOracle(SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES)
    results = oracle.evaluate(_good_group())
    free = {r.property_name for r in results if r.tolerance_free}
    assert "values_in_unit_interval" in free
    assert "rows_sum_to_one" not in free


def test_hybrid_oracle_includes_results_from_both_arms():
    oracle = HybridOracle(
        DeclarativeOracle(SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES),
        ReferenceOracle(reference_fn=_softmax),
    )
    names = {r.property_name for r in oracle.evaluate(_good_group())}
    assert "matches_reference" in names
    assert "rows_sum_to_one" in names


def test_hybrid_short_circuits_when_declarative_fails():
    # Declarative acts as a cheap pre-filter; the reference arm is not run.
    oracle = HybridOracle(
        DeclarativeOracle(SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES),
        ReferenceOracle(reference_fn=_softmax),
    )
    names = {r.property_name for r in oracle.evaluate(_bad_group())}
    assert "matches_reference" not in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_oracle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.oracle'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/oracle.py`:

```python
"""Phase C: the three oracle arms, evaluated offline over recorded rows.

Oracles never influence generation. They read a recorded case group and return
per-property results, so two arms can be scored over byte-identical executions.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

from autokernel_pbt.props.backends.base import ExecutionResult, kernel_inputs
from autokernel_pbt.props.properties import (
    TIER_PORTABLE,
    CaseProperty,
    GroupProperty,
)
from autokernel_pbt.props.tolerance import DEFAULT_THRESH, test_ratio, within_threshold
from autokernel_pbt.props.verdict import PropertyResult, Verdict, summarize


class Oracle(Protocol):
    name: str

    def evaluate(self, rows: list[ExecutionResult]) -> list[PropertyResult]: ...


class _OracleBase:
    def summary(self, results: list[PropertyResult]) -> Verdict:
        return summarize(results)


class ReferenceOracle(_OracleBase):
    """One strong property: output matches a reference implementation.

    Degenerate property-based testing — maximally strong, maximally brittle.
    """

    name = "reference"

    def __init__(
        self,
        reference_fn: Callable[..., np.ndarray],
        thresh: float = DEFAULT_THRESH,
    ) -> None:
        self.reference_fn = reference_fn
        self.thresh = thresh

    def evaluate(self, rows: list[ExecutionResult]) -> list[PropertyResult]:
        results = []
        for row in rows:
            results.append(self._check_row(row))
        return results

    def _check_row(self, row: ExecutionResult) -> PropertyResult:
        if row.status != "ok" or "y" not in row.outputs:
            return PropertyResult(
                property_name="matches_reference",
                tier=TIER_PORTABLE,
                tolerance_free=False,
                verdict=Verdict.INCONCLUSIVE,
                detail=f"status={row.status}",
            )
        expected = self.reference_fn(**kernel_inputs(row.case))
        got = row.outputs["y"]
        ratio = test_ratio(got, expected, dtype=got.dtype)
        ok = within_threshold(ratio, self.thresh)
        return PropertyResult(
            property_name="matches_reference",
            tier=TIER_PORTABLE,
            tolerance_free=False,
            verdict=Verdict.PASS if ok else Verdict.FAIL,
            detail=f"ratio={ratio:.3g} thresh={self.thresh}",
        )


class DeclarativeOracle(_OracleBase):
    """Many individually-weak properties. Never computes the answer."""

    name = "declarative"

    def __init__(
        self,
        case_properties: Sequence[CaseProperty],
        group_properties: Sequence[GroupProperty] = (),
    ) -> None:
        self.case_properties = tuple(case_properties)
        self.group_properties = tuple(group_properties)

    def evaluate(self, rows: list[ExecutionResult]) -> list[PropertyResult]:
        results: list[PropertyResult] = []
        for row in rows:
            for prop in self.case_properties:
                results.append(prop.check(row))
        for group_prop in self.group_properties:
            results.append(group_prop.check_group(rows))
        return results


class HybridOracle(_OracleBase):
    """Declarative laws as a cheap pre-filter; the reference arm only if they pass."""

    name = "hybrid"

    def __init__(self, declarative: DeclarativeOracle, reference: ReferenceOracle) -> None:
        self.declarative = declarative
        self.reference = reference

    def evaluate(self, rows: list[ExecutionResult]) -> list[PropertyResult]:
        results = self.declarative.evaluate(rows)
        if summarize(results) is Verdict.FAIL:
            return results
        return results + self.reference.evaluate(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/test_oracle.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/autokernel_pbt/props/oracle.py tests/unit/props/test_oracle.py
scripts/git_commit_clean.sh -m "feat: add reference, declarative, and hybrid oracles"
```

---

### Task 12: End-to-end record/replay integration

The test that proves the architecture works: one execution run, replayed through two oracle arms,
and an assertion that both arms saw **byte-identical inputs**. That assertion is the whole point.

**Files:**
- Create: `src/autokernel_pbt/props/tasks.py`
- Create: `tests/integration/test_record_replay.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_record_replay.py`:

```python
"""End-to-end: generate -> execute -> record -> replay two oracles."""

import numpy as np
import pytest

from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.oracle import DeclarativeOracle, ReferenceOracle
from autokernel_pbt.props.properties import SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference
from autokernel_pbt.props.verdict import Verdict


def _record(tmp_path, kernel):
    """Run Phase A and Phase B once, and return a reader over the recorded rows."""
    groups = Generator(SOFTMAX.domain, seed=42).generate(6)
    backend = NumpyBackend()
    results = [backend.run(kernel, case) for group in groups for case in group.cases]
    ExecutionTable(tmp_path / "run").write(results)
    return ExecutionTable(tmp_path / "run")


def _declarative() -> DeclarativeOracle:
    return DeclarativeOracle(SOFTMAX_CASE_PROPERTIES, SOFTMAX_GROUP_PROPERTIES)


@pytest.mark.integration
def test_correct_kernel_passes_both_arms(tmp_path):
    table = _record(tmp_path, softmax_reference)
    declarative = _declarative()
    reference = ReferenceOracle(reference_fn=softmax_reference)
    for rows in table.read_groups().values():
        assert declarative.summary(declarative.evaluate(rows)) is Verdict.PASS
        assert reference.summary(reference.evaluate(rows)) is Verdict.PASS


@pytest.mark.integration
def test_both_arms_see_byte_identical_inputs(tmp_path):
    """The core architectural guarantee: oracle choice cannot perturb the inputs."""
    table = _record(tmp_path, softmax_reference)
    first = table.read()
    second = table.read()
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.case.case_id == b.case.case_id
        assert np.array_equal(a.case.tensors["x"], b.case.tensors["x"])
        assert a.case.tensors["x"].dtype == b.case.tensors["x"].dtype


@pytest.mark.integration
def test_broken_kernel_is_caught_by_both_arms(tmp_path):
    def broken(x):
        # Normalizes magnitudes rather than exponentials: not shift invariant,
        # and does not match the reference.
        a = np.abs(x)
        return (a / a.sum(axis=-1, keepdims=True)).astype(x.dtype)

    table = _record(tmp_path, broken)
    declarative = _declarative()
    reference = ReferenceOracle(reference_fn=softmax_reference)
    groups = list(table.read_groups().values())
    assert any(declarative.summary(declarative.evaluate(r)) is Verdict.FAIL for r in groups)
    assert any(reference.summary(reference.evaluate(r)) is Verdict.FAIL for r in groups)


@pytest.mark.integration
def test_replay_needs_no_re_execution(tmp_path):
    """A second oracle scores the recorded run without touching the backend."""
    table = _record(tmp_path, softmax_reference)
    late_oracle = DeclarativeOracle(SOFTMAX_CASE_PROPERTIES, ())
    for rows in table.read_groups().values():
        assert late_oracle.summary(late_oracle.evaluate(rows)) is Verdict.PASS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_record_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.tasks'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/tasks.py`:

```python
"""Task registry for the development ladder.

The ladder adds one property class per rung: elementwise (pointwise) -> reduction
(order/associativity) -> normalization (numerical stability).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autokernel_pbt.props.domain import InputDomain, TensorSpec


@dataclass(frozen=True)
class Task:
    task_id: str
    domain: InputDomain


def relu_reference(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0).astype(x.dtype)


def softmax_reference(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return (e / e.sum(axis=-1, keepdims=True)).astype(x.dtype)


# Shapes are boundary-flavoured: powers of two, odd remainders, and single rows.
_LADDER_SHAPES = ((4, 8), (3, 7), (1, 16), (5, 1), (8, 33))

RELU = Task(
    task_id="relu",
    domain=InputDomain(
        task_id="relu",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=_LADDER_SHAPES,
        relations=(),
    ),
)

SOFTMAX = Task(
    task_id="softmax",
    domain=InputDomain(
        task_id="softmax",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=_LADDER_SHAPES,
        relations=("shift_rows",),
    ),
)

TASKS = {RELU.task_id: RELU, SOFTMAX.task_id: SOFTMAX}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_record_replay.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Run the whole suite**

Run: `pytest -m "not gpu" -q`
Expected: PASS, all tests green

- [ ] **Step 6: Commit**

```bash
git add src/autokernel_pbt/props/tasks.py tests/integration/test_record_replay.py
scripts/git_commit_clean.sh -m "feat: add task registry and record/replay integration test"
```

---

### Task 13: Kernel acceptance contracts drive the declarative oracle

This is the bridge between spec-driven and property-based development. A kernel's
`acceptance.yaml` names the properties it must satisfy, and the declarative oracle is *built from
that file* rather than hand-assembled in Python. Writing the spec becomes writing the oracle —
which is exactly what the authoring-cost metric will later measure.

**Files:**
- Create: `kernels/tasks/softmax/acceptance.yaml`
- Create: `kernels/tasks/relu/acceptance.yaml`
- Modify: `src/autokernel_pbt/props/properties.py` (append the registry)
- Create: `src/autokernel_pbt/props/contract.py`
- Test: `tests/unit/props/test_contract.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/test_contract.py`:

```python
"""Kernel acceptance contract tests."""

import pytest

from autokernel_pbt.props.contract import (
    UnknownPropertyError,
    load_contract,
    oracle_from_contract,
)


def test_loads_softmax_contract(repo_root):
    contract = load_contract(repo_root / "kernels/tasks/softmax/acceptance.yaml")
    assert contract.task_id == "softmax"
    assert "rows_sum_to_one" in contract.property_names


def test_builds_declarative_oracle_from_contract(repo_root):
    contract = load_contract(repo_root / "kernels/tasks/softmax/acceptance.yaml")
    oracle = oracle_from_contract(contract)
    names = {p.name for p in oracle.case_properties} | {p.name for p in oracle.group_properties}
    assert names == set(contract.property_names)


def test_relu_contract_has_no_group_properties(repo_root):
    oracle = oracle_from_contract(load_contract(repo_root / "kernels/tasks/relu/acceptance.yaml"))
    assert oracle.group_properties == ()


def test_unknown_property_name_is_rejected(tmp_path):
    path = tmp_path / "acceptance.yaml"
    path.write_text(
        "task_id: fake\nversion: 1\ncriteria:\n"
        "  - id: BOGUS\n    description: nope\n"
        "    check:\n      type: property\n      property: no_such_property\n"
    )
    with pytest.raises(UnknownPropertyError, match="no_such_property"):
        oracle_from_contract(load_contract(path))


def test_every_criterion_carries_a_description(repo_root):
    contract = load_contract(repo_root / "kernels/tasks/softmax/acceptance.yaml")
    assert all(c.description for c in contract.criteria)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_contract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.contract'`

- [ ] **Step 3: Add the property registry**

Append to `src/autokernel_pbt/props/properties.py`:

```python
# Registry so acceptance.yaml can name properties by string. Keep in sync when
# adding a property: an unregistered property cannot appear in a kernel contract.
CASE_PROPERTY_REGISTRY: dict[str, type] = {
    OutputsAreFinite.name: OutputsAreFinite,
    ValuesInUnitInterval.name: ValuesInUnitInterval,
    RowsSumToOne.name: RowsSumToOne,
}

GROUP_PROPERTY_REGISTRY: dict[str, type] = {
    ShiftInvariance.name: ShiftInvariance,
}
```

- [ ] **Step 4: Write the contract loader**

Create `src/autokernel_pbt/props/contract.py`:

```python
"""Kernel acceptance contracts: acceptance.yaml -> DeclarativeOracle.

The spec is the oracle. A kernel's acceptance criteria name properties, and the
declarative arm is constructed from that file rather than assembled by hand, so
spec-driven and property-based development share one artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from autokernel_pbt.props.oracle import DeclarativeOracle
from autokernel_pbt.props.properties import CASE_PROPERTY_REGISTRY, GROUP_PROPERTY_REGISTRY

PROPERTY_CHECK = "property"


class UnknownPropertyError(KeyError):
    """A contract named a property that is not in either registry."""


@dataclass(frozen=True)
class Criterion:
    id: str
    description: str
    property_name: str


@dataclass(frozen=True)
class Contract:
    task_id: str
    criteria: tuple[Criterion, ...]

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(c.property_name for c in self.criteria)


def load_contract(path: Path | str) -> Contract:
    data = yaml.safe_load(Path(path).read_text())
    criteria = tuple(
        Criterion(
            id=entry["id"],
            description=entry["description"],
            property_name=entry["check"]["property"],
        )
        for entry in data["criteria"]
        if entry["check"]["type"] == PROPERTY_CHECK
    )
    return Contract(task_id=data["task_id"], criteria=criteria)


def oracle_from_contract(contract: Contract) -> DeclarativeOracle:
    case_props = []
    group_props = []
    for name in contract.property_names:
        if name in CASE_PROPERTY_REGISTRY:
            case_props.append(CASE_PROPERTY_REGISTRY[name]())
        elif name in GROUP_PROPERTY_REGISTRY:
            group_props.append(GROUP_PROPERTY_REGISTRY[name]())
        else:
            raise UnknownPropertyError(
                f"{name!r} is not a registered property "
                f"(task {contract.task_id!r}); add it to a registry in properties.py"
            )
    return DeclarativeOracle(tuple(case_props), tuple(group_props))
```

- [ ] **Step 5: Write the kernel contracts**

Create `kernels/tasks/softmax/acceptance.yaml`:

```yaml
task_id: softmax
version: 1

criteria:
  - id: FINITE_OUTPUT
    description: no NaN or Inf in the output
    check:
      type: property
      property: outputs_are_finite

  - id: PROBABILITY_RANGE
    description: every output value lies in [0, 1]
    check:
      type: property
      property: values_in_unit_interval

  - id: NORMALIZED
    description: each row of the output sums to one
    check:
      type: property
      property: rows_sum_to_one

  - id: SHIFT_INVARIANT
    description: adding a per-row constant to the input leaves the output unchanged
    check:
      type: property
      property: shift_invariance
```

Create `kernels/tasks/relu/acceptance.yaml`:

```yaml
task_id: relu
version: 1

criteria:
  - id: FINITE_OUTPUT
    description: no NaN or Inf in the output
    check:
      type: property
      property: outputs_are_finite
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/unit/props/test_contract.py -v`
Expected: PASS, 5 passed

- [ ] **Step 7: Verify the feature 0004 spec tests are now green**

Run: `pytest tests/spec/test_0004_property_layer.py -v`
Expected: PASS, 3 passed — every acceptance criterion now resolves to a collectable test.

- [ ] **Step 8: Run the whole suite**

Run: `pytest -m "not gpu" -q`
Expected: PASS, all tests green

- [ ] **Step 9: Commit**

```bash
git add kernels/tasks src/autokernel_pbt/props/contract.py src/autokernel_pbt/props/properties.py tests/unit/props/test_contract.py
scripts/git_commit_clean.sh -m "feat: build declarative oracles from kernel acceptance contracts"
```

---

## Definition of Done

- [ ] Feature 0004 spec tests pass — every acceptance criterion resolves to a collectable test
- [ ] A kernel's declarative oracle is constructed from its `acceptance.yaml`, not hand-assembled
- [ ] `pytest -m "not gpu"` passes with no failures
- [ ] `ruff check src tests` passes
- [ ] A recorded run can be replayed through two oracle arms with no backend involvement
- [ ] Every property result carries `tier` and `tolerance_free` for later attribution analysis
- [ ] `Verdict.INCONCLUSIVE` is produced for failed executions, never silently treated as PASS

## Explicitly Out of Scope for Phase 1

These belong to Phases 2 and 3 and must not be built here:

- Mutation corpus and the four metrics (Phase 2)
- Shrinking (Phase 2)
- CUDA/Triton/NKI backends and tier-2 telemetry capture (Phase 3)
- Hypothesis corpus harvesting — the seeded NumPy generator is sufficient until Phase 2 needs
  float edge-case tuning
- Attention and GEMM tasks; KernelBench integration
- **GitHub spec-kit adoption.** Do not install `specify-cli` or create `.specify/` in this plan.
  Task 0 and Task 13 hand-author the spec artifacts. Once the property vocabulary exists, a
  follow-on phase adds a kernel-specific spec-kit extension that generates `spec.md` and drafts
  the `acceptance.yaml` property list; the machine-checkable `acceptance.yaml` contract defined in
  Task 13 remains hand-owned, because spec-kit produces natural-language artifacts only. Pin the
  spec-kit version when that phase lands.
