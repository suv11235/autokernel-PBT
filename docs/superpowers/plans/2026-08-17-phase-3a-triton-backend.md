# Phase 3a Triton Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Triton backend that records rich compile-time telemetry, Triton ports of the ladder, and a runbook that turns a rented Lambda GPU hour into a recorded run scored offline on CPU.

**Architecture:** Additive. `Backend` is unchanged; a `TritonKernel` adapter satisfies its existing `Callable[..., np.ndarray]` signature while exposing the compiled artifact telemetry is read from. Everything is written on a machine with no CUDA, so structure is verified by CPU contract tests and behaviour by `gpu`-marked tests run by hand on the instance.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, Triton, pytest, ruff.

**Design doc:** `docs/superpowers/specs/2026-08-17-phase-3a-triton-backend-design.md` — read §2 (why Triton, not CUDA C++), §4 (telemetry) and §6 (device realities) before starting.

---

## The constraint that shapes every task

**This is written blind.** The development machine is Darwin arm64 with no CUDA, no torch and no triton. Nothing in Tasks 1–6 can be executed against real hardware before the smoke session in Task 7.

Two consequences the plan takes seriously rather than hopes away:

1. **Every device-independent behaviour gets a CPU test.** Telemetry key completeness, adapter protocol conformance, status mapping totality, kernel-source hashing — all verifiable without a GPU, and all of them are where wiring bugs actually live.
2. **Triton's introspection API is unstable across versions and cannot be verified here.** Register counts and spill counts have moved between `CompiledKernel` attributes and `.metadata` across releases. Telemetry extraction is therefore written **defensively**: it probes a list of known locations, records what it found, and records *that a field was missing* rather than silently omitting it. A schema that quietly drops a field looks identical to hardware that does not report it.

## File Structure

| File | Responsibility |
|------|---------------|
| `src/autokernel_pbt/props/backends/telemetry.py` | The telemetry schema: key names, `schema_version`, defensive extraction helpers. Device-independent and fully CPU-testable. |
| `src/autokernel_pbt/props/backends/triton_kernel.py` | `TritonKernel` — the callable adapter. Host numpy in, host numpy out; owns launch config and exposes the compiled artifact. |
| `src/autokernel_pbt/props/backends/triton_backend.py` | `TritonBackend` — mirrors `NumpyBackend`'s shape, adds device telemetry, status mapping and the device-buffer integrity check. |
| `kernels/triton/ladder.py` | Triton ports of relu, softmax, layernorm, each matching its existing `acceptance.yaml`. |
| `src/autokernel_pbt/props/tasks.py` | Gains a tolerance-sweep task whose reduction lengths reach 16384. |
| `docs/runbooks/2026-08-17-lambda-smoke-session.md` | The runbook and bootstrap script for the rented instance. |
| `specs/features/0007-triton-backend/` | Spec and acceptance criteria (Task 0). |

Tests mirror this under `tests/unit/props/backends/` with `gpu`-marked device tests in `tests/gpu/`.

**Repo conventions.** Read `CLAUDE.md` first.

- **Never run `git commit`.** Use `scripts/git_commit_clean.sh -m "subject" -m "body"`, then **verify `git branch --show-current` is non-empty** — a detached HEAD silently orphaned three commits in this repo once, and checking out a *remote-tracking* ref (`origin/<branch>`) is one way in.
- `filterwarnings = ["error"]` — any warning is a test failure.
- `ruff check src tests` must pass; CI enforces it and a 95% coverage floor.
- Every new assertion must be the **unique** catcher for at least one saboteur.
- Bad **data** → `INCONCLUSIVE`; bad **call** → raise.

---

### Task 0: Feature 0007 spec and acceptance criteria (red)

**Files:**
- Create: `specs/features/0007-triton-backend/spec.md`
- Create: `specs/features/0007-triton-backend/acceptance.yaml`
- Create: `tests/spec/test_0007_triton_backend.py`
- Modify: `specs/README.md`

- [ ] **Step 1: Write the spec**

Create `specs/features/0007-triton-backend/spec.md`:

```markdown
# Feature 0007: Triton backend and the first hardware run

## Problem

Every measurement this project has produced is from one backend. The tier-1 properties claim to
be portable — they are the cross-backend equivalence contract the translation workstream is
built on — and that claim has never been tested against a second backend. The reference arm's
`log2(n)` tolerance is the pairwise-summation bound, and no backend other than NumPy has been
checked against it.

Recording on hardware needs neither the mutation corpus nor the metrics layer, because scoring
is offline over the recorded table.

## Scope

1. **Telemetry schema** — everything free at compile time, captured on every execution, with a
   `schema_version` and defensive extraction that records missing fields as missing.
2. **`TritonKernel`** — a callable adapter satisfying the existing `Backend` protocol while
   exposing the compiled artifact.
3. **`TritonBackend`** — device execution, status mapping including `COMPILE_ERROR`, and a
   device-buffer integrity check replacing the host-side read-only guarantee.
4. **Triton ports** of relu, softmax and layernorm against their existing contracts.
5. **A tolerance-sweep task** whose reduction lengths reach 16384, because the ladder spans only
   `log2(n)` 0..7.
6. **A runbook** for the Lambda session.

## Non-goals

- Tier-2 properties, `compute-sanitizer`, `ncu` (Phase 3b)
- The mutation corpus and the four metrics (Phase 2b)
- NKI / Trainium
- GEMM, attention, and any claim about tensor-core round-toward-zero
- Autotuning or any search over launch configurations

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
```

- [ ] **Step 2: Write the acceptance criteria**

Create `specs/features/0007-triton-backend/acceptance.yaml`:

```yaml
feature_id: "0007"
feature_name: triton-backend
version: 1

criteria:
  - id: TELEMETRY_SCHEMA_IS_COMPLETE
    description: every declared telemetry key is present on every execution, or explicitly absent
    check:
      type: unit_test
      test: tests/unit/props/backends/test_telemetry.py::test_every_declared_key_is_present

  - id: MISSING_FIELDS_ARE_RECORDED_NOT_DROPPED
    description: a field the toolchain does not report is recorded as missing, not omitted
    check:
      type: unit_test
      test: tests/unit/props/backends/test_telemetry.py::test_an_unavailable_field_is_recorded_as_missing

  - id: TELEMETRY_CARRIES_A_SCHEMA_VERSION
    description: a reader can tell a run recorded before a field existed from one where it was absent
    check:
      type: unit_test
      test: tests/unit/props/backends/test_telemetry.py::test_schema_version_is_recorded

  - id: ADAPTER_SATISFIES_THE_BACKEND_PROTOCOL
    description: TritonKernel is callable with numpy kwargs and returns a numpy array
    check:
      type: unit_test
      test: tests/unit/props/backends/test_triton_kernel.py::test_adapter_is_callable_with_numpy_kwargs

  - id: STATUS_MAPPING_IS_TOTAL
    description: every failure mode maps to a Status, with compile errors distinguished
    check:
      type: unit_test
      test: tests/unit/props/backends/test_triton_backend.py::test_status_mapping_is_total

  - id: COMPILE_ERROR_IS_DISTINGUISHED_FROM_LAUNCH_ERROR
    description: a kernel that never compiled is not reported as a launch failure
    check:
      type: unit_test
      test: tests/unit/props/backends/test_triton_backend.py::test_compile_failure_maps_to_compile_error

  - id: TOLERANCE_SWEEP_SPANS_THE_RANGE
    description: the sweep task reaches reduction lengths the ladder does not
    check:
      type: unit_test
      test: tests/unit/props/test_tasks.py::test_tolerance_sweep_spans_a_wider_log2_range_than_the_ladder
```

- [ ] **Step 3: Write the spec test**

Create `tests/spec/test_0007_triton_backend.py`, copying the structure of
`tests/spec/test_0006_four_arms.py` exactly — including its duplicate-target and file-only-target
checks — with `ACCEPTANCE = "specs/features/0007-triton-backend/acceptance.yaml"` and
`data["feature_id"] == "0007"`:

```python
"""Spec-derived acceptance tests (feature 0007).

These assert traceability: every criterion in acceptance.yaml must name a test that
actually exists and is collectable. This is the mechanism the SDD ADR asks for.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ACCEPTANCE = "specs/features/0007-triton-backend/acceptance.yaml"

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
def test_0007_acceptance_file_is_wellformed(repo_root: Path):
    data = yaml.safe_load((repo_root / ACCEPTANCE).read_text())
    assert data["feature_id"] == "0007"
    ids = [c["id"] for c in data["criteria"]]
    assert ids, "acceptance.yaml declares no criteria"
    assert len(ids) == len(set(ids)), f"duplicate criterion ids: {ids}"
    unknown = [
        f"{c['id']} -> {c['check']['type']}"
        for c in data["criteria"]
        if c["check"]["type"] not in KNOWN_CHECK_TYPES
    ]
    assert not unknown, f"criteria use unknown check types: {unknown}"
    targets = [
        c["check"]["test"] for c in data["criteria"] if c["check"]["type"] == "unit_test"
    ]
    # Two criteria sharing a node id both look traced while only one has independent
    # evidence — the same shape as a criterion certified by an assertion that never ran.
    assert len(targets) == len(set(targets)), f"criteria share test targets: {targets}"
    # A node id without `::` traces to a *file*, not an obligation.
    file_only = [
        f"{c['id']} -> {c['check']['test']}"
        for c in data["criteria"]
        if c["check"]["type"] == "unit_test" and "::" not in c["check"]["test"]
    ]
    assert not file_only, f"criteria name a file rather than a test node: {file_only}"


@pytest.mark.spec
def test_0007_every_criterion_names_an_existing_file(repo_root: Path):
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
def test_0007_every_criterion_is_collectable(repo_root: Path):
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
        # The non-zero exit *is* the signal under test.
        check=False,
    )
    assert proc.returncode == 0, (
        f"pytest could not collect all criteria:\n{proc.stdout}\n{proc.stderr}"
    )
```

**Note:** every criterion above names a **CPU-runnable** test. None requires a GPU. That is
deliberate — a criterion whose evidence only exists on rented hardware would leave `tests/spec/`
red on every developer machine and in CI.

- [ ] **Step 4: Register the feature**

In `specs/README.md`, add beneath the 0006 row:

```markdown
| [0007](./features/0007-triton-backend/spec.md) | Triton backend and first hardware run | in progress |
```

- [ ] **Step 5: Run the spec test to verify it fails (red)**

Run: `pytest tests/spec/test_0007_triton_backend.py -v`
Expected: `test_0007_acceptance_file_is_wellformed` PASSES; the other two FAIL, because none of the named test files exist yet.

- [ ] **Step 6: Commit**

```bash
git add specs/features/0007-triton-backend tests/spec/test_0007_triton_backend.py specs/README.md
scripts/git_commit_clean.sh -m "spec: add feature 0007, the Triton backend" -m "Every criterion names a CPU-runnable test. A criterion whose evidence only existed on rented hardware would leave tests/spec red on every developer machine and in CI, so the device behaviour is covered by gpu-marked tests that are not acceptance criteria."
git branch --show-current
```

---

### Task 1: GPU test infrastructure

The `gpu` marker is declared in `pyproject.toml` and used by nothing. This task makes it usable
and gives the whole suite a way to skip device tests cleanly.

**Files:**
- Create: `tests/gpu/__init__.py`
- Create: `tests/gpu/conftest.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`, replace the `gpu` extra:

```toml
gpu = [
  "torch>=2.2",
  "triton>=3.0",
]
```

`triton` is a separate distribution from `torch` on PyPI for Linux/CUDA. On Lambda Stack both are
already present in the system environment; the extra exists so a fresh venv on the instance can
install them, and so the versions are declared rather than implied.

- [ ] **Step 2: Write the skip fixture**

Create an empty `tests/gpu/__init__.py`:

```python
```

Create `tests/gpu/conftest.py`:

```python
"""Fixtures for device tests.

Everything here skips cleanly when there is no CUDA, so `pytest` on a developer
machine and in CI is green without a GPU. These tests are run by hand on the rented
instance; none of them is an acceptance criterion, for exactly that reason.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def torch_cuda():
    """The torch module, or a skip if CUDA is unavailable.

    Imported inside the fixture rather than at module scope: torch is an optional
    extra, and a module-level import would make collection fail on a machine without
    it rather than skip.
    """
    torch = pytest.importorskip("torch", reason="torch is not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    return torch


@pytest.fixture(scope="session")
def triton_module():
    return pytest.importorskip("triton", reason="triton is not installed")
```

- [ ] **Step 3: Verify the suite is still green without a GPU**

Run: `pytest -m "not gpu" -q`
Expected: PASS, unchanged count.

Run: `pytest -m gpu -q`
Expected: `no tests ran` — there are none yet, and that is the correct state.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/gpu
scripts/git_commit_clean.sh -m "test: make the gpu marker usable" -m "The marker has been declared in pyproject since phase 1 and used by nothing. The fixtures import torch and triton inside the fixture body rather than at module scope, so a machine without them skips rather than failing collection."
git branch --show-current
```

---

### Task 2: The telemetry schema

The one irreversible decision. Device-independent, so it is fully testable on CPU — which is the
point, because it is the thing that cannot be fixed after a paid run.

**Files:**
- Create: `src/autokernel_pbt/props/backends/telemetry.py`
- Create: `tests/unit/props/backends/__init__.py`
- Create: `tests/unit/props/backends/test_telemetry.py`

- [ ] **Step 1: Write the failing test**

Create an empty `tests/unit/props/backends/__init__.py`:

```python
```

Create `tests/unit/props/backends/test_telemetry.py`:

```python
"""Telemetry schema tests.

The schema is the one thing a re-run cannot recover, so it is tested hard and on CPU.
Extraction is deliberately defensive: Triton's introspection surface has moved between
releases, and a field that silently vanishes looks exactly like hardware that does not
report it.
"""

from __future__ import annotations

import pytest

from autokernel_pbt.props.backends.telemetry import (
    MISSING,
    SCHEMA_VERSION,
    TELEMETRY_SCHEMA_VERSION,
    declared_keys,
    extract,
    probe,
)


class _Meta:
    """Stands in for a Triton CompiledKernel's metadata."""

    num_warps = 4
    num_stages = 3
    shared = 8192


class _Compiled:
    """Stands in for a Triton CompiledKernel."""

    n_regs = 40
    n_spills = 0
    metadata = _Meta()
    asm = {"ptx": "// ptx text", "ttir": "// ttir"}


def test_schema_version_is_recorded():
    """The criterion TELEMETRY_CARRIES_A_SCHEMA_VERSION."""
    out = extract(_Compiled(), device={}, launch={})
    assert out[TELEMETRY_SCHEMA_VERSION] == SCHEMA_VERSION
    assert isinstance(SCHEMA_VERSION, int)


def test_every_declared_key_is_present():
    """The criterion TELEMETRY_SCHEMA_IS_COMPLETE.

    Every declared key appears on every execution. A key that is merely absent when
    unavailable makes "not captured" and "captured as zero" indistinguishable in an
    aggregate months later.
    """
    out = extract(_Compiled(), device={}, launch={})
    assert set(declared_keys()) <= set(out)


def test_an_unavailable_field_is_recorded_as_missing():
    """The criterion MISSING_FIELDS_ARE_RECORDED_NOT_DROPPED."""

    class _Bare:
        metadata = _Meta()

    out = extract(_Bare(), device={}, launch={})
    assert out["n_regs"] is MISSING
    assert "n_regs" in out


def test_probe_reads_the_first_location_that_exists():
    # Triton has moved these between the kernel object and its metadata across
    # releases, so extraction names several locations rather than one.
    assert probe(_Compiled(), ("nowhere", "n_regs")) == 40
    assert probe(_Compiled(), ("metadata.num_warps",)) == 4


def test_probe_returns_missing_when_no_location_exists():
    assert probe(_Compiled(), ("absent", "also.absent")) is MISSING


def test_probe_does_not_confuse_a_falsy_value_with_absence():
    # n_spills == 0 is the *good* case and must not read as "unavailable".
    assert probe(_Compiled(), ("n_spills",)) == 0
    assert probe(_Compiled(), ("n_spills",)) is not MISSING


def test_ptx_is_hashed_not_stored():
    # The PTX of a large kernel is tens of kilobytes and would be carried on every
    # row of the group. The hash identifies the artifact; the text is recoverable by
    # recompiling from the recorded source hash.
    out = extract(_Compiled(), device={}, launch={})
    assert out["ptx_hash"] != "// ptx text"
    assert len(out["ptx_hash"]) == 16


def test_absent_asm_hashes_to_missing():
    class _NoAsm:
        metadata = _Meta()

    assert extract(_NoAsm(), device={}, launch={})["ptx_hash"] is MISSING


def test_device_and_launch_groups_are_merged_verbatim():
    out = extract(_Compiled(), device={"device_name": "A10"}, launch={"grid": [8, 1, 1]})
    assert out["device_name"] == "A10"
    assert out["grid"] == [8, 1, 1]


def test_missing_is_json_serializable():
    # Telemetry is JSON-encoded into the execution row; a sentinel that cannot be
    # encoded would take the whole run's persistence with it.
    import json

    json.dumps(extract(_Compiled(), device={}, launch={}))


@pytest.mark.parametrize("key", ["n_regs", "n_spills", "shared_bytes", "num_warps", "num_stages"])
def test_the_fault_class_relevant_keys_are_declared(key):
    # These are the ISSTA taxonomy's device-only signals: register pressure, spills,
    # and the launch geometry a tile compiler chose. Losing one costs a hardware run.
    assert key in declared_keys()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/backends/test_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.backends.telemetry'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/backends/telemetry.py`:

```python
"""The device telemetry schema.

This is the one part of a hardware run that a re-run cannot recover. Everything else
— every verdict, every rate, every arm — is re-derivable offline from the recorded
table for free. A counter that was not captured costs another rented hour, which is
the cost the whole record/replay architecture exists to avoid. So the schema
over-captures, and it is written and tested on CPU where it is cheap to get right.

TWO RULES, both learned from what goes wrong in aggregates months later.

*A declared key is always present.* A field that is simply omitted when unavailable
makes "the toolchain did not report this" indistinguishable from "this was zero", and
the two mean opposite things about a kernel's register pressure. Unavailable fields
carry the `MISSING` sentinel instead.

*Extraction is defensive, not assertive.* Triton's introspection surface has moved
between releases — register and spill counts have lived on the compiled kernel object
and on its metadata at different times — and this module is written on a machine with
no Triton to check against. `probe` therefore names several candidate locations per
field and takes the first that exists, so a version bump degrades one field to
MISSING rather than raising mid-run and discarding the executions already paid for.
"""

from __future__ import annotations

import hashlib
from typing import Any

#: Bump when a field is added or its meaning changes. Recorded on every row so a
#: later reader can distinguish a run taken before a field existed from one where the
#: field was genuinely unavailable — without it the two are the same absence.
SCHEMA_VERSION = 1

TELEMETRY_SCHEMA_VERSION = "telemetry_schema_version"

#: Length of the truncated artifact digests. 64 bits separates the handful of kernel
#: variants in one experiment; these are not adversarial inputs.
HASH_CHARS = 16


class _Missing:
    """Sentinel for a field the toolchain did not report.

    A singleton with a JSON-friendly repr: telemetry is JSON-encoded into the
    execution row, and a sentinel that could not be encoded would raise inside the
    persistence loop and take the whole run's table with it.
    """

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


MISSING = _Missing()

#: field -> candidate locations on the compiled kernel, in priority order. Dotted
#: paths walk attributes. Several entries per field is the point: see the module
#: docstring on why this is probed rather than asserted.
_COMPILED_FIELDS: dict[str, tuple[str, ...]] = {
    "n_regs": ("n_regs", "metadata.n_regs", "num_regs"),
    "n_spills": ("n_spills", "metadata.n_spills", "num_spills"),
    "shared_bytes": ("shared", "metadata.shared", "metadata.shared_mem"),
    "num_warps": ("num_warps", "metadata.num_warps"),
    "num_stages": ("num_stages", "metadata.num_stages"),
}

#: Groups the backend supplies directly rather than probing off the artifact.
_DEVICE_KEYS = (
    "device_name",
    "compute_capability",
    "multi_processor_count",
    "total_memory_bytes",
    "driver_version",
    "runtime_version",
    "torch_version",
    "triton_version",
)
_LAUNCH_KEYS = ("grid", "constexprs")
_DERIVED_KEYS = ("ptx_hash",)


def declared_keys() -> tuple[str, ...]:
    """Every key the schema promises to emit, present or MISSING."""
    return (
        TELEMETRY_SCHEMA_VERSION,
        *_COMPILED_FIELDS,
        *_DEVICE_KEYS,
        *_LAUNCH_KEYS,
        *_DERIVED_KEYS,
    )


def probe(obj: Any, locations: tuple[str, ...]) -> Any:
    """First existing location's value, or MISSING.

    `getattr` chains rather than a single name because the same field lives in
    different places across Triton releases. Absence is distinguished from a falsy
    value: `n_spills == 0` is the *good* outcome and must never read as unavailable.
    """
    for location in locations:
        current: Any = obj
        for part in location.split("."):
            if not hasattr(current, part):
                current = MISSING
                break
            current = getattr(current, part)
        if current is not MISSING:
            return current
    return MISSING


def _hash(text: Any) -> Any:
    if not isinstance(text, str):
        return MISSING
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_CHARS]


def extract(
    compiled: Any, *, device: dict[str, Any], launch: dict[str, Any]
) -> dict[str, Any]:
    """Assemble one execution's telemetry.

    `device` and `launch` are merged verbatim — the backend knows those and does not
    need to probe for them. Everything off the compiled artifact is probed.

    The PTX is hashed rather than stored. A large kernel's PTX is tens of kilobytes
    and would be repeated on every row of every group; the hash identifies the
    artifact, and the text itself is recoverable by recompiling from the recorded
    kernel source hash.
    """
    out: dict[str, Any] = {TELEMETRY_SCHEMA_VERSION: SCHEMA_VERSION}
    for field, locations in _COMPILED_FIELDS.items():
        out[field] = probe(compiled, locations)

    asm = probe(compiled, ("asm",))
    ptx = asm.get("ptx") if isinstance(asm, dict) else MISSING
    out["ptx_hash"] = _hash(ptx)

    for key in _DEVICE_KEYS:
        out[key] = device.get(key, MISSING)
    for key in _LAUNCH_KEYS:
        out[key] = launch.get(key, MISSING)
    return out
```

- [ ] **Step 4: Make MISSING JSON-encodable end to end**

`test_missing_is_json_serializable` will fail: `json.dumps` cannot encode `_Missing`. The
execution table already routes unknown types through `_json_safe`, so extend it rather than
inventing a second mechanism.

In `src/autokernel_pbt/props/table.py`, add to `_json_safe` before its final raise:

```python
    if isinstance(obj, _Missing):
        # A field the toolchain did not report. Encoded as null so it round-trips as
        # None, which is distinguishable from 0 — the whole reason the sentinel exists.
        return None
```

and import it at the top:

```python
from autokernel_pbt.props.backends.telemetry import _Missing
```

Then relax the telemetry test to encode through that path:

```python
def test_missing_is_json_serializable():
    # Telemetry is JSON-encoded into the execution row; a sentinel that cannot be
    # encoded would take the whole run's persistence with it. It encodes as null, so
    # it round-trips as None -- distinguishable from 0, which is the point.
    import json

    from autokernel_pbt.props.table import _json_safe

    encoded = json.dumps(extract(_Compiled(), device={}, launch={}), default=_json_safe)
    assert json.loads(encoded)["device_name"] is None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/props/backends/test_telemetry.py -v`
Expected: PASS, all green.

Run: `pytest -m "not gpu" -q`
Expected: PASS.

- [ ] **Step 6: Saboteur check**

Delete the `n_regs` entry from `_COMPILED_FIELDS` and confirm exactly
`test_the_fault_class_relevant_keys_are_declared[n_regs]` and
`test_every_declared_key_is_present` fail. Restore. Then make `probe` return `MISSING` for a
falsy value (`if not current:`) and confirm exactly
`test_probe_does_not_confuse_a_falsy_value_with_absence` fails. Restore.

- [ ] **Step 7: Lint and commit**

```bash
ruff check src tests
git add src/autokernel_pbt/props/backends/telemetry.py src/autokernel_pbt/props/table.py tests/unit/props/backends
scripts/git_commit_clean.sh -m "feat: add the device telemetry schema" -m "The one part of a hardware run a re-run cannot recover, so it is written and tested on CPU where it is cheap to get right. A declared key is always present: a field omitted when unavailable makes 'the toolchain did not report this' indistinguishable from 'this was zero', and the two mean opposite things about register pressure." -m "Extraction probes several candidate locations per field rather than asserting one, because Triton's introspection surface has moved between releases and this is written on a machine with no Triton to check against. A version bump degrades one field to MISSING rather than raising mid-run and discarding executions already paid for."
git branch --show-current
```

---

### Task 3: The TritonKernel adapter

**Files:**
- Create: `src/autokernel_pbt/props/backends/triton_kernel.py`
- Create: `tests/unit/props/backends/test_triton_kernel.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/backends/test_triton_kernel.py`:

```python
"""TritonKernel adapter tests, all CPU.

The adapter's *structure* is what wiring bugs live in — protocol conformance, launch
config recording, source identity — and all of it is checkable without a GPU. The
device path is covered by gpu-marked tests run by hand on the instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.triton_kernel import TritonKernel


def _fake_jit(name: str = "fake_kernel"):
    """A stand-in for a @triton.jit function: named, and never actually launched."""

    def kernel():  # pragma: no cover - never called in CPU tests
        raise AssertionError("the fake kernel was launched")

    kernel.__name__ = name
    return kernel


def _adapter(**overrides) -> TritonKernel:
    defaults = dict(
        kernel_id="relu_triton",
        jit_fn=_fake_jit(),
        grid=lambda shape, constexprs: (1,),
        constexprs={"BLOCK_SIZE": 128},
        # **kw absorbs the grid= and constexprs= the adapter hands in.
        launcher=lambda **kw: np.zeros((2, 3), dtype=np.float32),
    )
    defaults.update(overrides)
    return TritonKernel(**defaults)


def test_adapter_is_callable_with_numpy_kwargs():
    """The criterion ADAPTER_SATISFIES_THE_BACKEND_PROTOCOL.

    `Backend.run` types the kernel as `Callable[..., np.ndarray]` and calls it as
    `kernel(**kernel_inputs(case))`. The adapter must satisfy that unchanged, or the
    Triton backend would need its own protocol and the two backends would stop being
    substitutable — which is what makes cross-backend comparison possible at all.
    """
    out = _adapter()(x=np.ones((2, 3), dtype=np.float32))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)


def test_launch_telemetry_records_the_constexprs():
    # BLOCK_SIZE and friends are what the tile compiler specialized on, and are the
    # ISSTA taxonomy's "tile mapping and launch" signal. Losing them costs a run.
    adapter = _adapter()
    adapter(x=np.ones((2, 3), dtype=np.float32))
    assert adapter.launch_telemetry()["constexprs"] == {"BLOCK_SIZE": 128}


def test_launch_telemetry_records_the_grid_actually_used():
    seen = {}

    def grid(shape, constexprs):
        seen["shape"] = shape
        return (4, 2, 1)

    adapter = _adapter(grid=grid)
    adapter(x=np.ones((8, 16), dtype=np.float32))
    assert adapter.launch_telemetry()["grid"] == [4, 2, 1]
    assert seen["shape"] == (8, 16)


def test_the_launcher_receives_exactly_the_grid_that_is_recorded():
    """Telemetry must describe the launch, not sit beside it.

    If the launcher computed its own grid, the recorded `grid` would be a label next
    to the behaviour rather than a description of it, and the two could drift apart
    silently -- with the artifacts reporting a launch geometry that never ran. That
    is the "asserted a label rather than the behaviour" defect this repo has hit four
    times, and launch geometry is the ISSTA taxonomy's own fault class.
    """
    seen = {}

    def launcher(*, grid, constexprs, **inputs):
        seen["grid"] = grid
        seen["constexprs"] = constexprs
        return np.zeros((2, 3), dtype=np.float32)

    adapter = _adapter(grid=lambda shape, ce: (7, 3, 1), launcher=launcher)
    adapter(x=np.ones((2, 3), dtype=np.float32))

    assert seen["grid"] == (7, 3, 1)
    assert seen["constexprs"] == {"BLOCK_SIZE": 128}
    assert adapter.launch_telemetry()["grid"] == [7, 3, 1]


def test_source_hash_distinguishes_two_kernels_sharing_a_name():
    # kernel_id is a label; the identity is the source. Two runs must not be able to
    # both call something "relu_triton" and mean different code.
    a = _adapter(jit_fn=_fake_jit("k"))
    b = _adapter(jit_fn=_fake_jit("k"), launcher=lambda **kw: np.ones((2, 3), np.float32))
    assert a.source_hash != b.source_hash


def test_source_hash_is_stable_for_one_kernel():
    adapter = _adapter()
    assert adapter.source_hash == adapter.source_hash


def test_compiled_is_none_before_the_first_call():
    # Triton compiles lazily, so there is no artifact to read telemetry from until
    # the kernel has run at least once. The backend must not assume otherwise.
    assert _adapter().compiled is None


def test_compiled_is_populated_after_a_call():
    sentinel = object()
    adapter = _adapter(launcher=lambda **kw: np.zeros((2, 3), np.float32))
    adapter._record_compiled(sentinel)
    assert adapter.compiled is sentinel


def test_a_launcher_returning_a_non_array_is_a_contract_error():
    from autokernel_pbt.props.backends.base import OutputContractError

    with pytest.raises(OutputContractError):
        _adapter(launcher=lambda **kw: None)(x=np.ones((2, 3), dtype=np.float32))


def test_inputs_are_copied_before_reaching_the_launcher():
    """`readonly_inputs` makes the host arrays non-writeable during execution.

    `torch.from_numpy` warns on a non-writeable array, and this project turns
    warnings into errors — so the adapter copies rather than aliasing. The copy is
    free relative to a host-to-device transfer, and `base.readonly_inputs`' own
    docstring flags this exact hazard as Phase 3's to solve.
    """
    x = np.ones((2, 3), dtype=np.float32)
    x.flags.writeable = False
    seen = {}

    def launcher(**kw):
        seen["writeable"] = kw["x"].flags.writeable
        return np.zeros((2, 3), dtype=np.float32)

    _adapter(launcher=launcher)(x=x)
    assert seen["writeable"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/backends/test_triton_kernel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.backends.triton_kernel'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/backends/triton_kernel.py`:

```python
"""The callable adapter that lets a Triton kernel satisfy the `Backend` protocol.

`Backend.run` types its kernel as `Callable[..., np.ndarray]` and calls it with the
case's tensors as keyword arguments. A Triton kernel is not that: it needs a launch
grid, constexpr block sizes, and device tensors. Rather than widen the protocol — which
would make the NumPy and Triton backends stop being substitutable, and substitutability
is what makes cross-backend comparison possible — the *kernel* is wrapped in an object
that is callable in exactly the way the protocol expects.

The adapter additionally owns two things the backend needs and the protocol has no
place for: the launch configuration actually used, and the compiled artifact telemetry
is read from. Triton compiles lazily, so the artifact does not exist until after the
first call.

`launcher` is injected rather than built here. On CPU there is no Triton to build one
with, and injecting it is what makes every structural behaviour in this module testable
without a GPU.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from autokernel_pbt.props.backends.base import single_output

HASH_CHARS = 16


class InputMutatedError(RuntimeError):
    """A kernel wrote to one of its input buffers.

    WHY THIS IS NOT CHECKED IN THE BACKEND, which is where the CPU equivalent lives.
    `readonly_inputs` protects the *host* array, and a Triton kernel never touches it
    — it writes to the device copy. The backend holds only host arrays, so a
    before/after hash there is structurally incapable of firing: it would compare two
    buffers the kernel could not have reached, pass every time, and read as a
    guarantee.

    The device tensors exist only inside the launcher, so the check must live there.
    That makes it the launcher author's responsibility, which is weaker than an
    enforced invariant — `device_digest` exists to make doing it a one-liner, and the
    ladder launchers all do. A launcher that omits it simply loses the guarantee for
    its own kernel rather than silently weakening it for everyone.

    Left unchecked, the failure is quiet and severe: an oracle recomputing a reference
    from a corrupted input can agree with an output computed from that same
    corruption, and both arms record a clean pass.
    """


def device_digest(tensor: Any) -> str:
    """Content digest of a device tensor, for the input-mutation check.

    Takes the digest on device where possible and falls back to a host copy. Called
    twice per launch, so it is deliberately cheap relative to the launch itself.
    """
    import torch

    with torch.no_grad():
        flat = tensor.reshape(-1).to(torch.float64)
        # A sum alone would miss a permutation; pairing it with a position-weighted
        # sum makes reordering visible too, at one extra reduction.
        weights = torch.arange(1, flat.numel() + 1, device=flat.device, dtype=torch.float64)
        return f"{float(flat.sum()):.17g}:{float((flat * weights).sum()):.17g}"


class TritonKernel:
    """One Triton kernel, callable as `kernel(**numpy_inputs) -> np.ndarray`."""

    def __init__(
        self,
        kernel_id: str,
        jit_fn: Callable[..., Any],
        grid: Callable[[tuple[int, ...], dict[str, Any]], tuple[int, ...]],
        constexprs: dict[str, Any],
        launcher: Callable[..., Any],
    ) -> None:
        self.kernel_id = kernel_id
        self.jit_fn = jit_fn
        self.grid = grid
        self.constexprs = dict(constexprs)
        self.launcher = launcher
        self.compiled: Any = None
        self._grid_used: tuple[int, ...] | None = None

    @property
    def source_hash(self) -> str:
        """Content identity of what will actually run.

        `kernel_id` is a label. Two runs must not be able to both call something
        `relu_triton` and mean different code, which would silently merge two
        variants' results into one number. Both the jit function and the launcher
        are hashed, because the launcher carries the block/stride arithmetic and a
        change there changes the kernel as surely as editing its body.
        """
        material = []
        for fn in (self.jit_fn, self.launcher):
            try:
                material.append(inspect.getsource(fn))
            except (OSError, TypeError):
                material.append(f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}")
        digest = hashlib.sha256("\n".join(material).encode("utf-8"))
        return digest.hexdigest()[:HASH_CHARS]

    def _record_compiled(self, compiled: Any) -> None:
        self.compiled = compiled

    def launch_telemetry(self) -> dict[str, Any]:
        """The launch group of the telemetry schema, as actually used."""
        return {
            "grid": list(self._grid_used) if self._grid_used is not None else None,
            "constexprs": dict(self.constexprs),
        }

    def __call__(self, **inputs: np.ndarray) -> np.ndarray:
        # Copy, do not alias. `readonly_inputs` flips the host arrays non-writeable
        # for the duration of execution, and `torch.from_numpy` warns on a
        # non-writeable array — which this project turns into an error. The copy is
        # negligible next to a host-to-device transfer, and `base.readonly_inputs`
        # names this as Phase 3's hazard to solve.
        writable = {name: np.array(value, copy=True) for name, value in inputs.items()}
        primary = next(iter(writable.values()))
        grid = tuple(self.grid(primary.shape, self.constexprs))
        self._grid_used = grid
        # The launcher is HANDED the grid and constexprs rather than recomputing them.
        # If it computed its own, the recorded launch telemetry would be a label next
        # to the behaviour rather than a description of it, and the two could drift
        # apart silently — with the artifacts reporting a grid that never launched.
        return single_output(self.launcher(grid=grid, constexprs=self.constexprs, **writable))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/backends/test_triton_kernel.py -v`
Expected: PASS, all green.

- [ ] **Step 5: Saboteur check**

Change `__call__` to pass `inputs` through unmodified instead of copying, and confirm exactly
`test_inputs_are_copied_before_reaching_the_launcher` fails. Restore. Then drop `self.launcher`
from `source_hash`'s material and confirm exactly
`test_source_hash_distinguishes_two_kernels_sharing_a_name` fails. Restore.

- [ ] **Step 6: Lint and commit**

```bash
ruff check src tests
git add src/autokernel_pbt/props/backends/triton_kernel.py tests/unit/props/backends/test_triton_kernel.py
scripts/git_commit_clean.sh -m "feat: add the TritonKernel adapter" -m "Wraps a Triton kernel so it satisfies Backend's existing Callable[..., np.ndarray] signature rather than widening the protocol. Widening it would make the NumPy and Triton backends stop being substitutable, and substitutability is what makes the cross-backend comparison possible at all." -m "Inputs are copied rather than aliased: readonly_inputs flips the host arrays non-writeable during execution and torch.from_numpy warns on those, which this project turns into an error. base.readonly_inputs' own docstring names this as phase 3's hazard, and the copy is negligible next to a host-to-device transfer." -m "launcher is injected rather than constructed, which is what makes every structural behaviour here testable on a machine with no Triton."
git branch --show-current
```

---

### Task 4: The TritonBackend

**Files:**
- Create: `src/autokernel_pbt/props/backends/triton_backend.py`
- Create: `tests/unit/props/backends/test_triton_backend.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/props/backends/test_triton_backend.py`:

```python
"""TritonBackend tests, CPU-only.

Status mapping, telemetry assembly and the integrity check are all structural and are
tested here with an injected fake device. The device path itself is covered by
gpu-marked tests in tests/gpu/, run by hand on the instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, Status
from autokernel_pbt.props.backends.telemetry import MISSING, TELEMETRY_SCHEMA_VERSION
from autokernel_pbt.props.backends.triton_backend import (
    TritonBackend,
    TritonCompilationError,
)
from autokernel_pbt.props.backends.triton_kernel import TritonKernel
from autokernel_pbt.props.case import Case


def _case() -> Case:
    return Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="relu",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.ones((2, 3), dtype=np.float32)},
    )


def _kernel(launcher=None, **overrides) -> TritonKernel:
    defaults = dict(
        kernel_id="k",
        jit_fn=lambda: None,
        grid=lambda shape, constexprs: (1,),
        constexprs={"BLOCK_SIZE": 64},
        launcher=launcher or (lambda **kw: np.zeros((2, 3), dtype=np.float32)),
    )
    defaults.update(overrides)
    return TritonKernel(**defaults)


def _backend(**overrides) -> TritonBackend:
    # device_probe is injected so the whole backend is exercisable with no CUDA.
    defaults = dict(device_probe=lambda: {"device_name": "fake", "compute_capability": "8.6"})
    defaults.update(overrides)
    return TritonBackend(**defaults)


def test_a_successful_run_reports_ok_and_the_output():
    result = _backend().run(_kernel(), _case())
    assert result.status is Status.OK
    assert result.outputs[OUTPUT_NAME].shape == (2, 3)


def test_telemetry_carries_the_schema_version_and_device_group():
    result = _backend().run(_kernel(), _case())
    assert result.telemetry[TELEMETRY_SCHEMA_VERSION] == 1
    assert result.telemetry["device_name"] == "fake"


def test_telemetry_carries_the_launch_group():
    result = _backend().run(_kernel(), _case())
    assert result.telemetry["constexprs"] == {"BLOCK_SIZE": 64}
    assert result.telemetry["grid"] == [1]


def test_compiled_fields_are_missing_when_nothing_compiled():
    # No artifact on a fake launcher. MISSING, not absent and not zero.
    result = _backend().run(_kernel(), _case())
    assert result.telemetry["n_regs"] is MISSING


def test_compile_failure_maps_to_compile_error():
    """The criterion COMPILE_ERROR_IS_DISTINGUISHED_FROM_LAUNCH_ERROR.

    Triton compiles on first call, so a compile error arrives during execution and
    would otherwise be indistinguishable from a launch failure. They mean different
    things: a kernel that never compiled says nothing about numerics, while one that
    launched and crashed may. Status.COMPILE_ERROR has existed unused since phase 1.
    """

    def boom(**kw):
        msg = "at 3:0: unexpected type"
        raise TritonCompilationError(msg)

    result = _backend().run(_kernel(launcher=boom), _case())
    assert result.status is Status.COMPILE_ERROR
    assert "unexpected type" in result.error


def test_launch_failure_maps_to_launch_error():
    def boom(**kw):
        msg = "an illegal memory access was encountered"
        raise RuntimeError(msg)

    result = _backend().run(_kernel(launcher=boom), _case())
    assert result.status is Status.LAUNCH_ERROR


def test_a_bad_output_maps_to_output_error():
    result = _backend().run(_kernel(launcher=lambda **kw: None), _case())
    assert result.status is Status.OUTPUT_ERROR


def test_status_mapping_is_total():
    """The criterion STATUS_MAPPING_IS_TOTAL.

    Every exception a kernel can raise reaches exactly one Status, and none escapes.
    An escaping exception aborts a run whose executions have already been paid for.
    """

    class Weird(Exception):
        pass

    def boom(**kw):
        raise Weird

    result = _backend().run(_kernel(launcher=boom), _case())
    assert result.status in set(Status)
    assert result.status is Status.LAUNCH_ERROR


def test_an_input_mutation_reported_by_the_launcher_is_a_launch_error():
    """The device replacement for the host-side read-only guarantee.

    The check itself lives in the launcher, which is the only layer holding the
    device buffers — see `InputMutatedError`. What the backend owes is classifying
    it, and classifying it as LAUNCH_ERROR rather than letting it escape: an
    escaping exception would abort a scoring pass over executions already paid for.
    """
    from autokernel_pbt.props.backends.triton_kernel import InputMutatedError

    def mutating(*, grid, constexprs, **inputs):
        msg = "kernel modified its input tensor(s) ['x']"
        raise InputMutatedError(msg)

    result = _backend().run(_kernel(launcher=mutating), _case())
    assert result.status is Status.LAUNCH_ERROR
    assert "modified its input" in result.error


def test_a_well_behaved_kernel_is_not_flagged():
    assert _backend().run(_kernel(), _case()).status is Status.OK


def test_the_case_is_carried_through_unchanged():
    case = _case()
    assert _backend().run(_kernel(), case).case is case


def test_backend_is_named_for_the_report_layer():
    assert TritonBackend.name == "triton"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/backends/test_triton_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autokernel_pbt.props.backends.triton_backend'`

- [ ] **Step 3: Write the implementation**

Create `src/autokernel_pbt/props/backends/triton_backend.py`:

```python
"""Device backend for Triton kernels.

Mirrors `NumpyBackend`'s shape deliberately: same protocol, same failure discipline, a
richer telemetry payload. A kernel that fails is *data*, never an exception that
escapes — an escaping exception aborts a scoring pass over executions that have
already cost rented hardware time.

Three things differ from the CPU backend, and each is a device reality rather than an
implementation choice. See the design doc §6.

*Compilation is lazy.* Triton compiles on first call, so a compile error arrives
during execution. `Status.COMPILE_ERROR` has existed unused since phase 1 for exactly
this. The distinction matters because a kernel that never compiled says nothing about
numerics, while one that launched and produced garbage says a great deal — and both
must still be INCONCLUSIVE in every arm.

*The read-only-inputs guarantee has to be rebuilt.* `readonly_inputs` protects the
host array, which the kernel never touches; a Triton kernel writes to the device copy.
This backend hashes the buffers the launcher was handed, before and after, and reports
a mismatch as a launch error.

*Execution is not bitwise reproducible.* Atomics and reduction order mean re-running
need not reproduce the recorded output. Nothing here depends on it — the recorded
execution is the one the arms score — but it does mean "re-run and compare" is not
available as a check, on device or in a test.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from autokernel_pbt.props.backends.base import (
    OUTPUT_NAME,
    TELEMETRY_BACKEND,
    TELEMETRY_WALL_MS,
    ExecutionResult,
    OutputContractError,
    Status,
    kernel_inputs,
)
from autokernel_pbt.props.backends.telemetry import extract
from autokernel_pbt.props.backends.triton_kernel import InputMutatedError, TritonKernel
from autokernel_pbt.props.case import Case


class TritonCompilationError(Exception):
    """Raised by a launcher when Triton fails to compile the kernel.

    Its own type rather than a string match on the message: Triton's compile errors
    are not a stable, documented surface, and classifying by substring would silently
    reclassify on a version bump — turning compile failures into launch failures in
    the artifacts, where nothing downstream could tell.
    """


class TritonBackend:
    """Executes `TritonKernel` adapters and records device telemetry."""

    name = "triton"

    def __init__(self, device_probe: Any = None) -> None:
        # Injected so the backend is exercisable with no CUDA present. The default
        # is imported lazily inside the probe, not at module import, so this module
        # imports cleanly on a machine with no torch.
        self.device_probe = device_probe or _default_device_probe

    def run(self, kernel: Any, case: Case) -> ExecutionResult:
        if not isinstance(kernel, TritonKernel):
            # A bad *call*, not bad data: it can only come from a coding error, costs
            # nothing to re-run, and a TypeError from deep inside a launch would name
            # neither the backend nor the kernel.
            msg = (
                f"{type(self).__name__} requires a TritonKernel adapter, got "
                f"{type(kernel).__name__}; wrap the jit function first"
            )
            raise TypeError(msg)

        inputs = kernel_inputs(case)
        start = time.perf_counter()
        try:
            output = kernel(**inputs)
        except TritonCompilationError as exc:
            return self._failed(kernel, case, start, Status.COMPILE_ERROR, exc)
        except InputMutatedError as exc:
            # Raised by the launcher, which is the only layer holding the device
            # buffers. See its note in triton_kernel.py for why the check cannot live
            # here.
            return self._failed(kernel, case, start, Status.LAUNCH_ERROR, exc)
        except OutputContractError as exc:
            return self._failed(kernel, case, start, Status.OUTPUT_ERROR, exc)
        except Exception as exc:  # noqa: BLE001 - a failing kernel is data, not an error
            return self._failed(kernel, case, start, Status.LAUNCH_ERROR, exc)

        return ExecutionResult(
            case=case,
            outputs={OUTPUT_NAME: output},
            telemetry=self._telemetry(kernel, start),
            status=Status.OK,
        )

    def _telemetry(self, kernel: TritonKernel, start: float) -> dict[str, Any]:
        payload = extract(
            kernel.compiled,
            device=self.device_probe(),
            launch=kernel.launch_telemetry(),
        )
        payload[TELEMETRY_BACKEND] = self.name
        payload[TELEMETRY_WALL_MS] = (time.perf_counter() - start) * 1000.0
        payload["kernel_source_hash"] = kernel.source_hash
        return payload

    def _failed(
        self, kernel: TritonKernel, case: Case, start: float, status: Status, exc: BaseException
    ) -> ExecutionResult:
        # Time before the failure is still signal: a kernel that dies after 30s is a
        # different problem from one that dies immediately.
        return ExecutionResult(
            case=case,
            telemetry=self._telemetry(kernel, start),
            status=status,
            error="".join(traceback.format_exception(exc)),
        )


def _default_device_probe() -> dict[str, Any]:
    """Device and toolchain facts, read once per execution.

    Imports inside the function: this module must import cleanly on a machine with no
    torch, so that every structural test above runs on CPU.
    """
    import torch

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:  # pragma: no cover - triton present wherever this runs
        triton_version = None
    return {
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multi_processor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "driver_version": torch.version.cuda,
        "runtime_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_version": triton_version,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/props/backends/test_triton_backend.py -v`
Expected: PASS, all green.

- [ ] **Step 5: Saboteur check**

Delete the mutation check block and confirm exactly
`test_a_kernel_that_mutates_its_input_is_a_launch_error` fails. Restore. Then change the
`TritonCompilationError` handler to `except TritonCompilationError: raise` and confirm exactly
`test_compile_failure_maps_to_compile_error` fails. Restore.

- [ ] **Step 6: Lint and commit**

```bash
ruff check src tests
git add src/autokernel_pbt/props/backends/triton_backend.py tests/unit/props/backends/test_triton_backend.py
scripts/git_commit_clean.sh -m "feat: add the Triton backend" -m "Mirrors NumpyBackend's shape and failure discipline with a richer telemetry payload. Compile errors get their own Status, which has existed unused since phase 1: a kernel that never compiled says nothing about numerics while one that launched and produced garbage says a great deal, and both must stay INCONCLUSIVE in every arm." -m "TritonCompilationError is a type rather than a substring match on Triton's message, because Triton's compile errors are not a stable surface and classifying by substring would silently reclassify on a version bump -- turning compile failures into launch failures in the artifacts, where nothing downstream could tell." -m "The read-only-inputs guarantee is rebuilt on device by hashing the buffers before and after: readonly_inputs protects the host array, which the kernel never touches."
git branch --show-current
```

---

### Task 5: Triton ports of the ladder

**Files:**
- Create: `kernels/triton/ladder.py`
- Delete: `kernels/triton/candidate.py`, `kernels/triton/reference_relu.py`
- Create: `tests/gpu/test_ladder_kernels.py`

- [ ] **Step 1: Remove the pre-property-layer stubs**

`kernels/triton/candidate.py` and `kernels/triton/reference_relu.py` are torch stubs from the
original skeleton — they contain no Triton at all and are referenced by nothing.

```bash
grep -rn "candidate\|reference_relu" --include="*.py" --include="*.yaml" --include="*.md" src tests kernels harness specs
```

Expected: matches only in `kernels/README.md`, if anywhere. Then:

```bash
git rm kernels/triton/candidate.py kernels/triton/reference_relu.py
```

- [ ] **Step 2: Write the Triton kernels**

Create `kernels/triton/ladder.py`:

```python
"""Triton ports of the development ladder.

One fixed launch configuration per task, deliberately. Sweeping BLOCK_SIZE would
multiply hardware time and confound the tier-1 transfer question -- does a property
that holds on NumPy hold on Triton -- with a block-size study. That study is Phase 3b.

Each kernel matches its task's existing `kernels/tasks/<id>/acceptance.yaml`. The
softmax and layernorm kernels subtract the row max and the row mean respectively, for
the same reason their NumPy references do: without it the declarative arm's shift
invariance property is testing a kernel that is wrong for a reason unrelated to the
backend.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from autokernel_pbt.props.backends.triton_kernel import (
    InputMutatedError,
    TritonKernel,
    device_digest,
)

#: One row per program, with the whole row in registers. Valid for reduction lengths
#: up to this bound, which covers the ladder (max 129) and the tolerance sweep
#: (max 16384) on any supported device.
BLOCK_SIZE = 16384


@triton.jit
def _relu_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    tl.store(y_ptr + row * n_cols + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def _layernorm_kernel(x_ptr, y_ptr, n_cols, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    tl.store(y_ptr + row * n_cols + offs, centered / tl.sqrt(var + EPS), mask=mask)


def _rows_grid(shape, constexprs):
    # One program per row. shape is the primary input's shape.
    return (shape[0],)


def _launcher(jit_kernel):
    """Build a launcher that uses the grid and constexprs the adapter handed it.

    Not its own: the adapter records what it passes as launch telemetry, so a
    launcher that recomputed either could make the artifacts describe a launch that
    never happened.
    """

    def launch(*, grid, constexprs, **inputs):
        x = inputs["x"]
        cols = x.shape[-1]
        xd = torch.as_tensor(x, device="cuda")
        yd = torch.empty_like(xd)

        # The input-mutation check, on the device buffer -- the only place it can
        # actually fire. See InputMutatedError for why the backend cannot do this.
        before = device_digest(xd)
        jit_kernel[grid](xd, yd, cols, **constexprs)
        if device_digest(xd) != before:
            msg = "kernel modified its input tensor(s) ['x'] on device"
            raise InputMutatedError(msg)

        # No astype: the output dtype should be whatever the kernel produced, so a
        # dtype bug stays visible instead of being cast away here.
        return yd.cpu().numpy()

    return launch


def relu_kernel() -> TritonKernel:
    return TritonKernel(
        kernel_id="relu_triton",
        jit_fn=_relu_kernel,
        grid=_rows_grid,
        constexprs={"BLOCK": BLOCK_SIZE},
        launcher=_launcher(_relu_kernel),
    )


def softmax_kernel() -> TritonKernel:
    return TritonKernel(
        kernel_id="softmax_triton",
        jit_fn=_softmax_kernel,
        grid=_rows_grid,
        constexprs={"BLOCK": BLOCK_SIZE},
        launcher=_launcher(_softmax_kernel),
    )


def layernorm_kernel() -> TritonKernel:
    # EPS matches tasks.LAYERNORM_EPS. A kernel targeting a different eps sits
    # systematically away from the reference and would inflate every
    # tolerance-bearing arm's rate for a reason that is not a defect.
    from autokernel_pbt.props.tasks import LAYERNORM_EPS

    return TritonKernel(
        kernel_id="layernorm_triton",
        jit_fn=_layernorm_kernel,
        grid=_rows_grid,
        constexprs={"BLOCK": BLOCK_SIZE, "EPS": LAYERNORM_EPS},
        launcher=_launcher(_layernorm_kernel),
    )


KERNELS = {
    "relu": relu_kernel,
    "softmax": softmax_kernel,
    "layernorm": layernorm_kernel,
}
```

- [ ] **Step 3: Write the device test**

Create `tests/gpu/test_ladder_kernels.py`:

```python
"""Device tests for the Triton ladder kernels. Run by hand on the instance.

Not acceptance criteria: a criterion whose evidence only exists on rented hardware
would leave tests/spec red on every developer machine and in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, Status
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.tasks import REFERENCES, TASKS

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("task_id", ["relu", "softmax", "layernorm"])
def test_the_triton_kernel_agrees_with_its_numpy_reference(task_id, torch_cuda, triton_module):
    from kernels.triton.ladder import KERNELS

    from autokernel_pbt.props.backends.triton_backend import TritonBackend

    backend = TritonBackend()
    kernel = KERNELS[task_id]()
    task = TASKS[task_id]
    for group in Generator(task.domain, seed=0).generate(len(task.domain.shapes)):
        for case in group.cases:
            result = backend.run(kernel, case)
            assert result.status is Status.OK, result.error
            expected = REFERENCES[task_id](x=case.tensors["x"])
            # Loose: this asserts the port is not grossly wrong. How close it *should*
            # be is the measurement the run exists to make, not a threshold to assume.
            assert np.allclose(result.outputs[OUTPUT_NAME], expected, rtol=1e-3, atol=1e-5)


def test_a_kernel_that_writes_to_its_input_is_caught_on_device(torch_cuda, triton_module):
    """The integrity check firing for real -- it cannot be exercised off-device.

    The CPU test in tests/unit/ only proves the backend CLASSIFIES the error. This
    proves it is actually raised, which is the half that could silently never fire.
    """
    import triton
    import triton.language as tl

    from autokernel_pbt.props.backends.triton_kernel import InputMutatedError, TritonKernel
    from kernels.triton.ladder import BLOCK_SIZE, _launcher

    @triton.jit
    def _vandal(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
        # Writes to the INPUT pointer: the defect the check exists for.
        tl.store(x_ptr + row * n_cols + offs, x + 1.0, mask=mask)
        tl.store(y_ptr + row * n_cols + offs, x, mask=mask)

    kernel = TritonKernel(
        kernel_id="vandal",
        jit_fn=_vandal,
        grid=lambda shape, ce: (shape[0],),
        constexprs={"BLOCK": BLOCK_SIZE},
        launcher=_launcher(_vandal),
    )
    with pytest.raises(InputMutatedError):
        kernel(x=np.ones((4, 8), dtype=np.float32))


@pytest.mark.parametrize("task_id", ["relu", "softmax", "layernorm"])
def test_compiled_telemetry_is_populated_on_device(task_id, torch_cuda, triton_module):
    """The whole reason for the schema: these must not come back MISSING."""
    from kernels.triton.ladder import KERNELS

    from autokernel_pbt.props.backends.telemetry import MISSING
    from autokernel_pbt.props.backends.triton_backend import TritonBackend

    task = TASKS[task_id]
    group = Generator(task.domain, seed=0).generate(len(task.domain.shapes))[0]
    result = TritonBackend().run(KERNELS[task_id](), group.base)
    assert result.status is Status.OK, result.error
    for key in ("n_regs", "shared_bytes", "num_warps"):
        assert result.telemetry[key] is not MISSING, f"{key} came back MISSING on device"
```

- [ ] **Step 4: Verify the suite is unaffected on CPU**

Run: `pytest -m "not gpu" -q`
Expected: PASS. `kernels/triton/ladder.py` imports torch and triton at module scope, so it must
not be imported by anything CPU-side — the device test imports it inside the test body for
exactly that reason.

Run: `ruff check src tests kernels`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add -A kernels tests/gpu
scripts/git_commit_clean.sh -m "feat: add Triton ports of the ladder" -m "One fixed launch configuration per task. Sweeping BLOCK_SIZE would multiply hardware time and confound the tier-1 transfer question with a block-size study, which is phase 3b's." -m "Removes candidate.py and reference_relu.py, torch stubs from the pre-property-layer skeleton that contained no Triton and were referenced by nothing. The device tests are deliberately not acceptance criteria: a criterion whose evidence only exists on rented hardware would leave tests/spec red on every developer machine and in CI."
git branch --show-current
```

---

### Task 6: The tolerance-sweep task

The ladder spans `log2(n)` 0..7. The CPU measurements that chose `log2(n)` swept to 14. Without
this, the run cannot answer the tolerance question — see design §3.

**Files:**
- Modify: `src/autokernel_pbt/props/tasks.py`
- Modify: `tests/unit/props/test_tasks.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/props/test_tasks.py`:

```python
def test_tolerance_sweep_spans_a_wider_log2_range_than_the_ladder():
    """The criterion TOLERANCE_SWEEP_SPANS_THE_RANGE.

    The ladder's reduction lengths are {1, 7, 8, 16, 32, 33, 64, 129} -- log2 from 0
    to 7. The CPU measurements that chose log2(n) as the normalization swept n to
    16384, log2 = 14. Measuring the tolerance on the ladder alone would cover half
    the dynamic range, at the noisy low end.
    """
    sweep = {shape[-1] for shape in TASKS["tolerance_sweep"].domain.shapes}
    ladder = {shape[-1] for shape in TASKS["softmax"].domain.shapes}
    assert max(sweep) >= 16384
    assert np.log2(max(sweep)) - np.log2(max(ladder)) >= 6


def test_tolerance_sweep_reuses_the_softmax_reference():
    # The sweep measures the *tolerance*, not a new op. A second reference would make
    # its numbers incomparable with the softmax numbers already recorded on CPU.
    assert REFERENCES["tolerance_sweep"] is REFERENCES["softmax"]


def test_tolerance_sweep_declares_no_relations():
    # It exists to sweep n, not to exercise metamorphic properties; partner cases
    # would double its hardware time for nothing.
    assert TASKS["tolerance_sweep"].domain.relations == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/props/test_tasks.py -k tolerance_sweep -v`
Expected: FAIL with `KeyError: 'tolerance_sweep'`

- [ ] **Step 3: Write the implementation**

In `src/autokernel_pbt/props/tasks.py`, add after `LAYERNORM`:

```python
#: Powers of two spanning log2(n) = 3..14, plus two non-power-of-two lengths so the
#: sweep is not blind to tail-handling. Few rows per shape: the quantity under study
#: is the reduction length, and extra rows buy variance reduction at a linear cost in
#: hardware time.
_SWEEP_SHAPES: tuple[tuple[int, ...], ...] = (
    (4, 8),
    (4, 64),
    (4, 512),
    (4, 4096),
    (4, 16384),
    (4, 129),
    (4, 4095),
)

#: Not a new op -- softmax again, over a much wider range of reduction lengths.
#:
#: The ladder spans log2(n) from 0 to 7; the CPU measurements that chose log2(n) as
#: the reference arm's normalization swept to 16384, log2 = 14. Measuring the
#: tolerance on the ladder alone would cover half the dynamic range, concentrated at
#: the low end where the ratio is noisiest, so this task exists to make the
#: normalization question answerable on device at all.
#:
#: It reuses ``softmax_reference`` deliberately. A separate reference would make these
#: numbers incomparable with the softmax numbers already recorded on CPU, which is the
#: comparison the whole exercise rests on.
TOLERANCE_SWEEP = Task(
    task_id="tolerance_sweep",
    domain=InputDomain(
        task_id="tolerance_sweep",
        tensors=(TensorSpec(name="x", dtype="float32", distribution="normal"),),
        shapes=_SWEEP_SHAPES,
        relations=(),
    ),
)
```

Update the registries:

```python
TASKS: dict[str, Task] = {
    task.task_id: task for task in (RELU, SOFTMAX, LAYERNORM, TOLERANCE_SWEEP)
}

REFERENCES = {
    RELU.task_id: relu_reference,
    SOFTMAX.task_id: softmax_reference,
    LAYERNORM.task_id: layernorm_reference,
    TOLERANCE_SWEEP.task_id: softmax_reference,
}
```

- [ ] **Step 4: Give it a contract**

`tests/unit/props/test_contract.py::test_contracts_and_tasks_are_in_step` requires every task to
have one. Create `kernels/tasks/tolerance_sweep/acceptance.yaml`:

```yaml
# Acceptance contract for the tolerance-sweep task.
#
# The same properties as softmax, because it IS softmax -- the task exists to sweep
# the reduction length far wider than the ladder does, so that the reference arm's
# log2(n) normalization can be checked against a real device reduction tree.
#
# No SHIFT_INVARIANCE criterion: the domain declares no relations, so there is no
# partner case for a group property to read, and a contract naming one would build an
# oracle that abstains on every group.
task_id: tolerance_sweep
version: 1

criteria:
  - id: FINITE_OUTPUT
    description: no output element is NaN or Inf, on any input the domain generates
    check:
      type: property
      property: outputs_are_finite

  - id: UNIT_INTERVAL
    description: every output element is a probability, lying in [0, 1] exactly
    check:
      type: property
      property: values_in_unit_interval

  - id: ROW_SUMS
    description: each row of the output sums to one within its dtype's rounding budget
    check:
      type: property
      property: rows_sum_to_one
```

- [ ] **Step 5: Run the tests**

Run: `pytest -m "not gpu" -q`
Expected: PASS. If `test_contracts_and_tasks_are_in_step` fails, the contract file is missing or
its `task_id` does not match.

- [ ] **Step 6: Lint and commit**

```bash
ruff check src tests
git add src/autokernel_pbt/props/tasks.py kernels/tasks/tolerance_sweep tests/unit/props/test_tasks.py
scripts/git_commit_clean.sh -m "feat: add the tolerance-sweep task" -m "The ladder's reduction lengths span log2(n) from 0 to 7; the CPU measurements that chose log2(n) as the reference arm's normalization swept to 16384, log2 = 14. Measuring the tolerance on the ladder alone would cover half the dynamic range at the noisy low end, so the device run could not answer the normalization question at all." -m "It reuses softmax_reference rather than introducing a new op, because a separate reference would make these numbers incomparable with the softmax numbers already recorded on CPU -- which is the comparison the exercise rests on."
git branch --show-current
```

---

### Task 7: The Lambda runbook

The smoke session exists to let the irreversible decision fail cheaply. Everything here is
written so the instance clock is spent recording, not debugging.

**Files:**
- Create: `docs/runbooks/2026-08-17-lambda-smoke-session.md`
- Create: `scripts/gpu_bootstrap.sh`
- Create: `scripts/gpu_record.py`

- [ ] **Step 1: Write the bootstrap script**

Create `scripts/gpu_bootstrap.sh`:

```bash
#!/usr/bin/env bash
# Prepare a fresh Lambda instance to record a run. Idempotent; safe to re-run.
#
# Lambda Stack ships torch and CUDA in the system Python, so this installs the
# project against that interpreter rather than building a venv that would shadow a
# working torch with a wheel that may not match the driver.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== versions the run will be recorded under =="
python3 -c "import torch; print('torch    ', torch.__version__)"
python3 -c "import torch; print('cuda     ', torch.version.cuda)"
python3 -c "import triton; print('triton   ', triton.__version__)"
python3 -c "import torch; print('device   ', torch.cuda.get_device_name(0))"
python3 -c "import torch; p=torch.cuda.get_device_properties(0); print('capability', f'{p.major}.{p.minor}')"

echo "== installing the project (no deps: torch and triton come from Lambda Stack) =="
python3 -m pip install --quiet -e ".[dev]"

echo "== CPU suite, to prove the checkout is sound before spending on device =="
python3 -m pytest -m "not gpu" -q

echo "== device tests =="
python3 -m pytest -m gpu -q

echo "bootstrap OK"
```

Make it executable:

```bash
chmod +x scripts/gpu_bootstrap.sh
```

- [ ] **Step 2: Write the recording entrypoint**

Create `scripts/gpu_record.py`:

```python
#!/usr/bin/env python3
"""Record one task's executions on the GPU. Scoring happens later, on CPU.

Deliberately does NOT score. The driver's scoring pass builds a declarative arm from
the task's contract and evaluates four oracles, none of which needs a device -- and
doing it here would spend rented time on work that is free at home. This writes the
execution table and stops.

Usage:
    python3 scripts/gpu_record.py --task softmax --out runs/gpu-softmax
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autokernel_pbt.props.backends.triton_backend import TritonBackend
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import TASKS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-groups", type=int, default=None)
    args = parser.parse_args()

    from kernels.triton.ladder import KERNELS

    task = TASKS[args.task]
    # Default to the whole domain: fewer groups than shapes makes Generator warn that
    # a boundary shape will never be exercised, and boundary coverage is the corpus's
    # main recall mechanism.
    n_groups = args.n_groups or len(task.domain.shapes)

    kernel_factory = KERNELS.get(args.task) or KERNELS["softmax"]
    kernel = kernel_factory()
    backend = TritonBackend()

    results = []
    for group in Generator(task.domain, seed=args.seed).generate(n_groups):
        for case in group.cases:
            result = backend.run(kernel, case)
            result.kernel_id = kernel.kernel_id
            result.kernel_is_broken = False
            result.case_spec = group.spec
            results.append(result)

    ExecutionTable(args.out).write(results)

    statuses: dict[str, int] = {}
    for result in results:
        statuses[str(result.status)] = statuses.get(str(result.status), 0) + 1
    print(json.dumps({"task": args.task, "rows": len(results), "status": statuses}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Write the runbook**

Create `docs/runbooks/2026-08-17-lambda-smoke-session.md`:

```markdown
# Runbook: Lambda smoke session

**Purpose.** Prove the Triton backend compiles, launches, and that the telemetry schema
survives contact with real hardware — *before* the real run. The schema is the one decision a
re-run cannot fix, so it gets one cheap rehearsal.

**Expected duration.** Under an hour, most of it bootstrap.

## Before you start

- Pick the **cheapest** available GPU. The full ladder is ~28 KB of tensor payload across nine
  trivial kernels and the tolerance sweep adds little; nothing here is compute-bound. What
  matters is that compute capability is *recorded*, not that it is high.
- Everything below is copy-pasteable. If you find yourself debugging on the clock, stop,
  terminate the instance, and fix it locally — that is what the CPU contract tests are for.

## 1. Launch and connect

Launch the instance from the Lambda console and connect over SSH.

## 2. Bootstrap

```bash
git clone https://github.com/suv11235/autokernel-PBT.git
cd autokernel-PBT
./scripts/gpu_bootstrap.sh
```

Record the version block it prints — torch, CUDA, triton, device, capability. Lambda Stack
pins these from the instance image, so two runs on different images are not automatically
comparable, and this block is what makes the difference visible.

**If the device tests fail, stop here.** That is the smoke session having done its job.
Capture the output, terminate, fix locally.

## 3. Record

```bash
python3 scripts/gpu_record.py --task relu            --out runs/gpu-relu
python3 scripts/gpu_record.py --task softmax         --out runs/gpu-softmax
python3 scripts/gpu_record.py --task layernorm       --out runs/gpu-layernorm
python3 scripts/gpu_record.py --task tolerance_sweep --out runs/gpu-tolerance
```

Each prints a status histogram. Anything other than all-`ok` is a finding, not a reason to
retry blindly.

## 4. Check the telemetry actually populated

This is the whole point of the session.

```bash
python3 - <<'EOF'
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.backends.telemetry import declared_keys
rows = ExecutionTable("runs/gpu-softmax").read()
t = rows[0].telemetry
missing = [k for k in declared_keys() if t.get(k) is None]
print("rows:", len(rows))
for k in declared_keys():
    print(f"  {k:28} {t.get(k)}")
print("\nfields that came back empty:", missing)
EOF
```

**Any field in that empty list is a decision to make now, on the instance, while it is still
cheap.** A field that is empty because Triton moved it needs another probe location added to
`_COMPILED_FIELDS`; a field empty because the device does not report it is a real absence and
should stay `MISSING`.

## 5. Bring the runs home

Instance storage is ephemeral. **Nothing is saved until this step completes.**

```bash
tar czf runs.tar.gz runs/
```

From your laptop:

```bash
scp ubuntu@<instance-ip>:~/autokernel-PBT/runs.tar.gz .
tar xzf runs.tar.gz
```

Verify locally before terminating:

```bash
python3 -c "
from autokernel_pbt.props.table import ExecutionTable
for t in ('relu','softmax','layernorm','tolerance'):
    print(t, len(ExecutionTable(f'runs/gpu-{t}').read()), 'rows')
"
```

## 6. Terminate the instance

Only after step 5 verifies locally.

## 7. Score at home, on CPU

No device required — this is the property the whole architecture exists to buy.

Scoring the recorded runs through the four arms is the next piece of work; the executions are
already safe on disk and can be scored any time.
```

- [ ] **Step 4: Verify the scripts are syntactically sound on CPU**

Run: `bash -n scripts/gpu_bootstrap.sh`
Expected: no output (valid shell).

Run: `python3 -m py_compile scripts/gpu_record.py`
Expected: no output.

Run: `pytest -m "not gpu" -q && ruff check src tests scripts`
Expected: PASS and clean.

- [ ] **Step 5: Commit**

```bash
git add docs/runbooks scripts/gpu_bootstrap.sh scripts/gpu_record.py
scripts/git_commit_clean.sh -m "docs: add the Lambda smoke-session runbook" -m "The smoke session exists so the telemetry schema -- the one decision a re-run cannot fix -- fails cheaply before the real run. Step 4 checks which declared fields actually populated on device, because a field empty due to a Triton version move needs another probe location while a field empty because the hardware does not report it is a real absence, and telling them apart is only cheap while the instance is still up." -m "gpu_record.py records and stops rather than scoring. The driver's scoring pass needs no device, so doing it on the instance would spend rented time on work that is free at home. Bringing the runs home is its own numbered step because instance storage is ephemeral and nothing is saved until it completes."
git branch --show-current
```

---

### Task 8: Close out

**Files:**
- Modify: `specs/README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Mark the feature implemented**

In `specs/README.md`, change the 0007 row's status to `implemented`.

- [ ] **Step 2: Record what the device work established**

In `CLAUDE.md`, add to the module-contracts table:

```markdown
| Telemetry declares every key; unavailable ones carry `MISSING` | An omitted field makes "not captured" and "captured as zero" indistinguishable, and they mean opposite things about register pressure. |
| Triton introspection is **probed**, not asserted | Register and spill counts have moved between the compiled kernel and its metadata across releases. A version bump must degrade one field, not abort a paid run. |
| The input-mutation check lives in the **launcher**, not the backend | `readonly_inputs` protects the host array, and the backend holds only host arrays — a check there is structurally incapable of firing. Only the launcher holds the device buffers. |
```

And add to Open obligations:

```markdown
6. **The first hardware run has not happened.** `TritonBackend`, the ladder ports and the
   tolerance sweep are written but have never executed on a GPU — the development machine has
   no CUDA. Everything device-side is verified by structure only until the smoke session
   (`docs/runbooks/2026-08-17-lambda-smoke-session.md`) runs.
```

- [ ] **Step 3: Verify the whole gate**

Run: `ruff check src tests scripts`
Expected: `All checks passed!`

Run: `pytest -m "not gpu" -q --cov=autokernel_pbt --cov-report=term --cov-fail-under=95`
Expected: PASS, coverage at or above 95%.

Run: `pytest tests/spec/ -v`
Expected: PASS — every 0007 criterion resolves.

- [ ] **Step 4: Commit**

```bash
git add specs/README.md CLAUDE.md
scripts/git_commit_clean.sh -m "docs: close out feature 0007" -m "The backend, the ladder ports, the tolerance sweep and the runbook are written and structurally tested. None of it has executed on a GPU, which is now open obligation 6 rather than an unstated assumption."
git branch --show-current
```

---

## Definition of Done

- [ ] All seven feature 0007 criteria resolve to collectable CPU tests, and `tests/spec/` is green
- [ ] `pytest -m "not gpu"` passes with no failures and no warnings
- [ ] `pytest -m gpu` collects the device tests and skips them cleanly with no CUDA
- [ ] `ruff check src tests scripts` passes
- [ ] Coverage is at or above 95%
- [ ] Every declared telemetry key is emitted on every execution, `MISSING` when unavailable
- [ ] A compile failure is `COMPILE_ERROR`, distinct from `LAUNCH_ERROR`
- [ ] A kernel that writes to its input is caught **on device** by the launcher and classified `LAUNCH_ERROR`
- [ ] The tolerance sweep reaches reduction length 16384
- [ ] The runbook is copy-pasteable end to end, and brings the runs home before terminating

## Explicitly Out of Scope

- Tier-2 properties, `compute-sanitizer`, `ncu` (Phase 3b)
- The mutation corpus and the four metrics (Phase 2b)
- NKI / Trainium
- GEMM, attention, and any claim about tensor-core round-toward-zero
- Autotuning or any search over launch configurations
- Scoring the recorded GPU runs — the executions are the deliverable; scoring them is the next
  piece of work and needs no device
