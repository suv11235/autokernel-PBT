"""Kernel acceptance contracts, and the oracle built from them.

This is where the project's two halves meet. A kernel's ``acceptance.yaml`` names
the laws it must satisfy; ``oracle_from_contract`` turns that file into the
``DeclarativeOracle`` that scores it. Writing the spec *is* writing the oracle, and
the authoring cost of that file is the research metric the reference arm's
near-empty contract is compared against.

Because the file is the oracle, every way the file can be wrong is a way the
declarative arm can be silently weakened — a missing criterion, a misspelled
property name, a set that defers a finding to a property it does not contain. None
of those raise anything on their own; they produce an arm that quietly judges less
and reports a clean miss. So each is rejected at load or build time, and each
rejection is pinned here by a test that matches its own message.

That last part is the invariant this suite is built to satisfy, and it is stronger
than "every validation has a test": every validation is the *unique* catcher for at
least one test. Verified by deleting each in turn and re-running this module — each
of the thirteen (not-a-mapping, task_id, version, empty criteria, duplicate ids,
duplicate property, missing criterion key, unknown check type, blank id, empty
description, unknown property, deferral, one-contract-per-task) fails exactly its
own case or cases and nothing else.

Two of those guards are also pinned against a specific *ordering* regression rather
than only against deletion. ``id`` and ``description`` are type-checked before they
are coerced, because a bare ``id:`` or ``description:`` key parses to YAML ``None``
and ``str(None)`` is the perfectly non-empty ``"None"`` — reinstating the coerce-
first order leaves both guards in place and still walks past the blank-key tests
below. A guard can be defeated by moving a ``str()`` call, so the tests assert the
behaviour rather than the presence of the line.

The one deliberate exception to the unique-catcher rule is
``validate_property_set``'s empty-set guard, which this module cannot reach at all
because ``load_contract`` rejects an empty ``criteria`` list first; it is pinned in
``test_oracle.py``, where it is reachable.

``test_the_contract_is_what_catches_the_bug`` is the load-bearing one: it proves
the file is not decorative by deleting a criterion from a copy of it and watching
the detection disappear.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.contract import (
    CONTRACT_FILENAME,
    CONTRACT_VERSION,
    KERNEL_TASKS_DIR,
    PROPERTY_CHECK,
    Contract,
    Criterion,
    UnknownPropertyError,
    contract_paths,
    load_contract,
    oracle_from_contract,
)
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.oracle import summary
from autokernel_pbt.props.properties import (
    CASE_PROPERTY_REGISTRY,
    GROUP_PROPERTY_REGISTRY,
    OutputsAreFinite,
    RowsSumToOne,
    ShiftInvariance,
    ValuesInUnitInterval,
)
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import TASKS
from autokernel_pbt.props.verdict import Verdict

SEED = 20240815


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _criterion(name: str, *, id_: str | None = None, description: str = "a law") -> dict[str, Any]:
    """One criterion in the on-disk shape, keyed off the property's own name.

    The property names are taken from the classes rather than written as string
    literals: a literal here would keep passing after a rename, which is exactly
    the drift these tests exist to catch.
    """
    return {
        "id": id_ or name.upper(),
        "description": description,
        "check": {"type": PROPERTY_CHECK, "property": name},
    }


def _write(path: Path, document: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def _contract_file(tmp_path: Path, names: list[str], **overrides: Any) -> Path:
    document: dict[str, Any] = {
        "task_id": "softmax",
        "version": 1,
        "criteria": [_criterion(name) for name in names],
    }
    document.update(overrides)
    return _write(tmp_path / "acceptance.yaml", document)


def _document(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _softmax_contract(repo_root: Path) -> Contract:
    return load_contract(repo_root / "kernels" / "tasks" / "softmax" / "acceptance.yaml")


def unnormalized_softmax(x: np.ndarray) -> np.ndarray:
    """Broken: forgets the division by the row sum.

    The same kernel the record/replay suite uses, and for the same reason: it is
    deterministic and data-independent, so ``rows_sum_to_one`` — and only
    ``rows_sum_to_one`` — fails on every row of every non-degenerate shape. That
    makes "the contract caught it" an assertion about the contract rather than
    about a lucky draw.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(shifted).astype(x.dtype)


def _record(task_id: str, kernel: Callable[..., np.ndarray], run_dir: Path) -> ExecutionTable:
    """Generate, execute and persist one run, then hand back a fresh reader."""
    task = TASKS[task_id]
    groups = Generator(task.domain, SEED).generate(len(task.domain.shapes))
    backend = NumpyBackend()
    results = [backend.run(kernel, case) for group in groups for case in group.cases]
    ExecutionTable(run_dir).write(results)
    return ExecutionTable(run_dir)


def _verdicts(oracle: Any, table: ExecutionTable) -> dict[str, Verdict]:
    return {gid: summary(oracle.evaluate(rows)) for gid, rows in table.read_groups().items()}


# --------------------------------------------------------------------------- #
# The contracts on disk
# --------------------------------------------------------------------------- #


def test_softmax_contract_names_the_four_softmax_laws(repo_root: Path):
    contract = _softmax_contract(repo_root)
    assert contract.task_id == "softmax"
    assert contract.property_names == (
        OutputsAreFinite.name,
        ValuesInUnitInterval.name,
        RowsSumToOne.name,
        ShiftInvariance.name,
    )
    assert all(isinstance(c, Criterion) for c in contract.criteria)
    assert all(c.description.strip() for c in contract.criteria), (
        "the description is the spec half of the contract; an empty one is not a spec"
    )


def test_relu_contract_names_only_the_law_relu_obeys(repo_root: Path):
    """Relu's contract is deliberately one line long.

    Relu's outputs are unbounded above, its rows do not sum to one, and its domain
    declares no relations — so every other property would be either actively wrong
    or permanently INCONCLUSIVE. See
    ``test_record_replay.py::test_relu_uses_only_the_properties_relu_actually_obeys``
    for the measured cost of getting this wrong.
    """
    contract = load_contract(repo_root / "kernels" / "tasks" / "relu" / "acceptance.yaml")
    assert contract.task_id == "relu"
    assert contract.property_names == (OutputsAreFinite.name,)


def test_every_contract_on_disk_names_registered_properties_and_builds_an_oracle(repo_root: Path):
    """C: the round trip that stops a typo becoming a silently weaker oracle.

    A misspelled property name in a YAML file is not a syntax error anywhere. This
    walk is the only thing that turns it into a failure before a run is scored.
    """
    paths = contract_paths(repo_root)
    assert paths, "no kernel contracts found; the walk would certify nothing"
    for path in paths:
        contract = load_contract(path)
        for name in contract.property_names:
            assert name in CASE_PROPERTY_REGISTRY or name in GROUP_PROPERTY_REGISTRY, (
                f"{path} names unregistered property {name!r}"
            )
        oracle = oracle_from_contract(contract)
        assert oracle.case_properties or oracle.group_properties


def _assert_contracts_and_tasks_are_in_step(root: Path) -> None:
    """D, factored out so a duplicate-contract saboteur can be run against it.

    The one-contract-per-task assertion comes *first* and is stated over the path
    list rather than the dict. Keying by ``task_id`` is lossy: two contracts
    claiming one task collapse to a single entry, so ``contracts.keys() ==
    TASKS.keys()`` stays True and the surviving path — whichever sorts last — also
    satisfies the directory-name check. A second contract for softmax could then sit
    in the tree indefinitely, and which of the two a future driver scored would
    depend on how it happened to iterate.
    """
    paths = contract_paths(root)
    contracts = {load_contract(path).task_id: path for path in paths}
    assert len(contracts) == len(paths), (
        f"two contracts claim one task_id; {len(paths)} files collapsed to "
        f"{sorted(contracts)} ({[str(p) for p in paths]})"
    )
    assert contracts.keys() == TASKS.keys()
    for task_id, path in contracts.items():
        assert path.parent.name == task_id, (
            f"{path} declares task_id {task_id!r} but lives under {path.parent.name!r}"
        )


def test_contracts_and_tasks_are_in_step(repo_root: Path):
    """D: every task has a contract and every contract names a task.

    ``REFERENCES.keys() == TASKS.keys()`` is asserted for the reference arm one
    module over; this is the same obligation for the declarative arm. A task with
    no contract has no declarative arm at all, and a contract naming no task
    describes a corpus that is never generated.
    """
    _assert_contracts_and_tasks_are_in_step(repo_root)


def test_a_second_contract_claiming_one_task_is_caught(repo_root: Path, tmp_path: Path):
    """The duplicate must be caught wherever it sorts, not only after the real one.

    The copy is named so it sorts *before* ``softmax``, which is the case a
    dict-keyed check misses: the real directory would overwrite the impostor's entry
    and every assertion downstream would pass.
    """
    root = tmp_path / "root"
    for path in contract_paths(repo_root):
        _write(root / KERNEL_TASKS_DIR / path.parent.name / CONTRACT_FILENAME, _document(path))
    _assert_contracts_and_tasks_are_in_step(root)

    softmax = repo_root / KERNEL_TASKS_DIR / "softmax" / CONTRACT_FILENAME
    _write(root / KERNEL_TASKS_DIR / "aaa_copy" / CONTRACT_FILENAME, _document(softmax))
    with pytest.raises(AssertionError, match="two contracts claim one task_id"):
        _assert_contracts_and_tasks_are_in_step(root)


def test_contract_properties_split_across_the_two_scopes(repo_root: Path):
    """The loader must route each name to the scope its registry declares.

    A group property placed in ``case_properties`` would be called through
    ``check`` — which it does not have — and blow up mid-run rather than at build
    time.
    """
    oracle = oracle_from_contract(_softmax_contract(repo_root))
    assert [p.name for p in oracle.case_properties] == [
        OutputsAreFinite.name,
        ValuesInUnitInterval.name,
        RowsSumToOne.name,
    ]
    assert [p.name for p in oracle.group_properties] == [ShiftInvariance.name]


# --------------------------------------------------------------------------- #
# E: the contract is what catches the bug
# --------------------------------------------------------------------------- #


def test_the_contract_is_what_catches_the_bug(repo_root: Path, tmp_path: Path):
    """E: delete a criterion from a copy of the file, lose the detection.

    This is the whole point of the task stated as an experiment. The oracle is
    never assembled by hand here: it is built from a YAML file, scores a recorded
    broken-kernel table, and catches the bug. Then the same file minus one
    criterion is loaded, and the same table is scored again — and nothing is
    caught. The file is load-bearing, not documentation.
    """
    table = _record("softmax", unnormalized_softmax, tmp_path / "run")

    full = _softmax_contract(repo_root)
    caught = _verdicts(oracle_from_contract(full), table)
    assert Verdict.FAIL in caught.values(), caught

    document = yaml.safe_load(
        (repo_root / "kernels" / "tasks" / "softmax" / "acceptance.yaml").read_text()
    )
    document["criteria"] = [
        c for c in document["criteria"] if c["check"]["property"] != RowsSumToOne.name
    ]
    weakened = load_contract(_write(tmp_path / "weakened" / "acceptance.yaml", document))
    assert RowsSumToOne.name not in weakened.property_names

    missed = _verdicts(oracle_from_contract(weakened), table)
    assert missed.keys() == caught.keys()
    assert Verdict.FAIL not in missed.values(), (
        "the weakened contract still caught the bug; the removed criterion was not "
        "what was doing the catching, so this test proves nothing"
    )


# --------------------------------------------------------------------------- #
# A: the deferral validation fires through the contract
# --------------------------------------------------------------------------- #


def test_a_contract_that_defers_to_an_absent_property_is_rejected(tmp_path: Path):
    """A: the blind contract — the failure mode with no error anywhere.

    ``values_in_unit_interval`` and ``rows_sum_to_one`` both return INCONCLUSIVE on
    non-finite output, deferring the finding to ``outputs_are_finite``. A contract
    that selects them without it is structurally incapable of catching a
    NaN-producing kernel: every member defers, every member abstains, and the arm
    records a clean miss.
    """
    path = _contract_file(tmp_path, [ValuesInUnitInterval.name, RowsSumToOne.name])
    contract = load_contract(path)
    with pytest.raises(ValueError, match="defers non-finite output"):
        oracle_from_contract(contract)


def test_the_blind_contract_would_really_have_missed_the_nan(tmp_path: Path):
    """Why A is worth a validation: the miss, measured.

    Without the rejection above, the blind property set is not merely
    theoretically weaker — it scores a NaN-returning kernel as INCONCLUSIVE on
    every group while the contract's honest counterpart FAILs. Building the set
    directly here (bypassing the contract loader) is deliberate: it is the only way
    to observe what the validation is preventing.
    """
    table = _record("relu", lambda x: np.where(x > 0, x, np.nan).astype(x.dtype), tmp_path / "run")
    blind = [CASE_PROPERTY_REGISTRY[ValuesInUnitInterval.name]()]
    verdicts = {
        gid: summary([p.check(row) for row in rows for p in blind])
        for gid, rows in table.read_groups().items()
    }
    assert verdicts and Verdict.FAIL not in verdicts.values()

    honest = oracle_from_contract(
        load_contract(_contract_file(tmp_path, [OutputsAreFinite.name], task_id="relu"))
    )
    assert Verdict.FAIL in _verdicts(honest, table).values()


# --------------------------------------------------------------------------- #
# B and the rest of the load-time validations
# --------------------------------------------------------------------------- #


def test_an_empty_criteria_list_is_rejected(tmp_path: Path):
    """B: an oracle that judges nothing must not be constructible from a file.

    It would summarize to INCONCLUSIVE on every group, adding groups that
    established nothing to the denominator of the detection rate, with no error
    anywhere.
    """
    with pytest.raises(ValueError, match="declares no criteria"):
        load_contract(_contract_file(tmp_path, []))


def test_a_missing_criteria_key_is_rejected(tmp_path: Path):
    document = {"task_id": "softmax", "version": 1}
    with pytest.raises(ValueError, match="declares no criteria"):
        load_contract(_write(tmp_path / "acceptance.yaml", document))


def test_a_contract_with_no_task_id_is_rejected(tmp_path: Path):
    """A contract naming no task cannot be joined to the corpus it is meant to score."""
    with pytest.raises(ValueError, match="declares no task_id"):
        load_contract(_contract_file(tmp_path, [OutputsAreFinite.name], task_id=""))


def test_an_unregistered_property_name_names_both_the_property_and_the_task(tmp_path: Path):
    """A typo must fail loudly, and the message must say where to look."""
    path = _contract_file(tmp_path, [OutputsAreFinite.name, "rows_sum_to_1"])
    contract = load_contract(path)
    with pytest.raises(UnknownPropertyError, match="rows_sum_to_1"):
        oracle_from_contract(contract)
    with pytest.raises(UnknownPropertyError, match="softmax"):
        oracle_from_contract(contract)
    assert issubclass(UnknownPropertyError, KeyError)


def test_an_unknown_check_type_is_rejected(tmp_path: Path):
    """G: the shared ``check: {type: ...}`` shape, with this loader's vocabulary.

    A criterion carrying a feature-spec check type (``unit_test``, say) in a kernel
    contract is not a property selection and must not be silently skipped — a
    skipped criterion is a law nobody checks.
    """
    document = {
        "task_id": "softmax",
        "criteria": [
            _criterion(OutputsAreFinite.name),
            {
                "id": "SMOKE",
                "description": "a feature-spec check that wandered in",
                "check": {"type": "unit_test", "test": "tests/unit/props/test_contract.py"},
            },
        ],
    }
    with pytest.raises(ValueError, match="unknown check type"):
        load_contract(_write(tmp_path / "acceptance.yaml", document))


def test_a_criterion_missing_a_required_key_is_rejected(tmp_path: Path):
    document = {
        "task_id": "softmax",
        "criteria": [{"id": "FINITE", "description": "no key naming a property"}],
    }
    with pytest.raises(ValueError, match="missing required key"):
        load_contract(_write(tmp_path / "acceptance.yaml", document))


def test_a_criterion_with_no_description_is_rejected(tmp_path: Path):
    """The description is the spec half; without it the file is a property list."""
    document = {
        "task_id": "softmax",
        "criteria": [_criterion(OutputsAreFinite.name, description="   ")],
    }
    with pytest.raises(ValueError, match="empty description"):
        load_contract(_write(tmp_path / "acceptance.yaml", document))


def test_a_bare_description_key_is_rejected(tmp_path: Path):
    """The blank description an author actually writes: ``description:`` and nothing.

    YAML parses that as ``None``, and ``str(None)`` is the perfectly non-empty
    ``"None"`` — so a guard that coerced before stripping would accept it and record
    a criterion whose spec half reads "None". The guard exists precisely for the
    author who skipped the prose, so missing this case would leave it guarding only
    the deliberate whitespace nobody types.
    """
    document = {
        "task_id": "softmax",
        "criteria": [
            {
                "id": "FINITE_OUTPUT",
                "description": None,
                "check": {"type": PROPERTY_CHECK, "property": OutputsAreFinite.name},
            }
        ],
    }
    with pytest.raises(ValueError, match="empty description"):
        load_contract(_write(tmp_path / "acceptance.yaml", document))


def test_a_bare_id_key_is_rejected(tmp_path: Path):
    """The same trap on ``id``, where it additionally defeats the duplicate check.

    Two criteria written with a bare ``id:`` both become ``"None"`` under coercion —
    non-empty, and *equal*, so they collide under the duplicate-id check as though
    the author had deliberately reused a name. The reported error would then be
    about duplication rather than about the two missing ids, pointing the fix at the
    wrong place.
    """
    blank = {
        "id": None,
        "description": "a law with no traceable id",
        "check": {"type": PROPERTY_CHECK, "property": OutputsAreFinite.name},
    }
    document = {"task_id": "softmax", "criteria": [blank, {**blank, "check": dict(blank["check"])}]}
    with pytest.raises(ValueError, match="blank id"):
        load_contract(_write(tmp_path / "acceptance.yaml", document))


def test_an_unknown_contract_version_is_rejected(tmp_path: Path):
    """``version:`` is read, not merely written.

    An unvalidated version key reads as a compatibility guarantee the loader does
    not provide: a file written against a future format would be parsed under
    today's rules, silently losing whatever that format added.
    """
    path = _contract_file(tmp_path, [OutputsAreFinite.name], version=CONTRACT_VERSION + 98)
    with pytest.raises(ValueError, match="declares version"):
        load_contract(path)


def test_an_omitted_version_means_version_one(tmp_path: Path):
    document = {
        "task_id": "softmax",
        "criteria": [_criterion(OutputsAreFinite.name)],
    }
    assert load_contract(_write(tmp_path / "acceptance.yaml", document)).task_id == "softmax"


def test_the_contracts_on_disk_declare_the_version_the_loader_understands(repo_root: Path):
    for path in contract_paths(repo_root):
        assert _document(path)["version"] == CONTRACT_VERSION


def test_duplicate_criterion_ids_are_rejected(tmp_path: Path):
    """Two criteria under one id are not traceable back to a single obligation."""
    document = {
        "task_id": "softmax",
        "criteria": [
            _criterion(OutputsAreFinite.name, id_="SAME"),
            _criterion(ValuesInUnitInterval.name, id_="SAME"),
        ],
    }
    with pytest.raises(ValueError, match="duplicate criterion ids"):
        load_contract(_write(tmp_path / "acceptance.yaml", document))


def test_a_property_named_twice_is_rejected(tmp_path: Path):
    """A repeated property would be evaluated twice and counted twice.

    Every per-property tally the project reports is a count over
    ``PropertyResult``s, so a duplicated name inflates both the numerator and the
    denominator of a rate nobody would think to distrust.
    """
    document = {
        "task_id": "softmax",
        "criteria": [
            _criterion(OutputsAreFinite.name, id_="FIRST"),
            _criterion(OutputsAreFinite.name, id_="SECOND"),
        ],
    }
    with pytest.raises(ValueError, match="names property .* twice"):
        load_contract(_write(tmp_path / "acceptance.yaml", document))


def test_a_contract_file_that_is_not_a_mapping_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="is not a mapping"):
        load_contract(_write(tmp_path / "acceptance.yaml", [_criterion(OutputsAreFinite.name)]))
