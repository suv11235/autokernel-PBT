"""Kernel acceptance contracts: the file that *is* the declarative oracle.

This module is where the project's spec-driven and test-driven halves meet. A
kernel's ``kernels/tasks/<task>/acceptance.yaml`` names, one criterion at a time,
the laws that kernel must satisfy; ``oracle_from_contract`` looks each name up in
the property registries and returns the ``DeclarativeOracle`` that scores it.
Nothing assembles a property set by hand. Writing the spec is writing the oracle,
which is what makes "authoring cost" a measurable research quantity — it is the
cost of producing this file, and the reference arm's near-empty contract (one
trusted recomputation) is the comparison.

The format deliberately reuses the shape of ``specs/features/*/acceptance.yaml``:
``criteria[].id`` / ``.description`` / ``.check.type``, with ``type: property``
and a ``property:`` naming a registered law. A single loader can therefore read
both a feature spec and a kernel contract, and a criterion is a criterion whichever
file it lives in.

WHY THIS MODULE VALIDATES SO MUCH. Every way a contract can be wrong is a way the
declarative arm is silently weakened, never a way it errors. A misspelled property
name, an omitted criterion, a set that defers a finding to a property it does not
contain — each produces an arm that judges less and reports a clean, plausible
miss, in the very comparison the project exists to make. There is no runtime signal
for any of them, so they are rejected here, at load or build time, before any
hardware is touched. Each rejection names what it rejected and why, and
``tests/unit/props/test_contract.py`` pins each to its own message.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from autokernel_pbt.props.oracle import DeclarativeOracle
from autokernel_pbt.props.properties import (
    CASE_PROPERTY_REGISTRY,
    GROUP_PROPERTY_REGISTRY,
    CaseProperty,
    GroupProperty,
)

#: The one ``check.type`` a kernel contract understands. The feature specs use
#: ``unit_test``, ``json_schema`` and friends in the same slot; a criterion carrying
#: one of those here is not a property selection and is rejected rather than skipped,
#: because a skipped criterion is a law nobody checks.
PROPERTY_CHECK = "property"

#: Where contracts live, relative to the repo root, and what they are called. Kept
#: here rather than in the tests so the walk in ``test_contract.py`` and the loader
#: cannot disagree about which files are contracts — a test globbing a directory the
#: loader does not know about would certify an empty set.
KERNEL_TASKS_DIR = Path("kernels") / "tasks"
CONTRACT_FILENAME = "acceptance.yaml"

_REQUIRED_CRITERION_KEYS = ("id", "description", "check")


class UnknownPropertyError(KeyError):
    """A contract names a property no registry provides.

    A ``KeyError`` because that is what the lookup would otherwise have raised, and
    a distinct type because callers must be able to tell "this contract is wrong"
    from an incidental dict miss inside the oracle machinery.
    """


@dataclass(frozen=True)
class Criterion:
    """One obligation: a traceable id, a human-readable claim, and the law that checks it.

    ``description`` is not decoration. It is the half of the file a human writes and
    reviews — the spec — and the reason a contract can be read as a statement about
    the kernel rather than as a list of function names. An empty one is rejected at
    load time.
    """

    id: str
    description: str
    property_name: str


@dataclass(frozen=True)
class Contract:
    """A task's declarative obligations, in file order.

    Order is preserved rather than normalized to a set, because it is the order the
    author chose and the order results will be reported in; a reader comparing the
    file against a result table should not have to reconcile two orderings.
    """

    task_id: str
    criteria: tuple[Criterion, ...]

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(criterion.property_name for criterion in self.criteria)


def contract_paths(root: Path) -> tuple[Path, ...]:
    """Every kernel contract under ``root``, sorted.

    Takes the root explicitly rather than deriving it from ``__file__``: the package
    is installed from ``src/`` and an installed copy has no repo above it, so a
    derived root would be right under test and wrong in a wheel — silently returning
    an empty tuple, which is exactly the "certifies nothing" failure the round-trip
    test is built to prevent.
    """
    return tuple(sorted((root / KERNEL_TASKS_DIR).glob(f"*/{CONTRACT_FILENAME}")))


def _criterion_from(path: Path, task_id: str, raw: Any) -> Criterion:
    if not isinstance(raw, dict):
        msg = f"contract {path} for task {task_id!r} has a criterion that is not a mapping: {raw!r}"
        raise ValueError(msg)
    missing = [key for key in _REQUIRED_CRITERION_KEYS if key not in raw]
    if missing:
        msg = (
            f"contract {path} for task {task_id!r} has a criterion missing required "
            f"key(s) {missing}: {raw!r}"
        )
        raise ValueError(msg)

    criterion_id = str(raw["id"])
    check = raw["check"]
    if not isinstance(check, dict):
        msg = f"contract {path} criterion {criterion_id!r} has a check that is not a mapping"
        raise ValueError(msg)
    check_type = check.get("type")
    if check_type != PROPERTY_CHECK:
        msg = (
            f"contract {path} criterion {criterion_id!r} uses unknown check type "
            f"{check_type!r}; a kernel contract understands only {PROPERTY_CHECK!r}, and a "
            f"criterion this loader skipped would be a law nobody checks"
        )
        raise ValueError(msg)
    if PROPERTY_CHECK not in check:
        msg = (
            f"contract {path} criterion {criterion_id!r} is missing required key "
            f"['{PROPERTY_CHECK}'] under check"
        )
        raise ValueError(msg)

    description = str(raw["description"])
    if not description.strip():
        msg = (
            f"contract {path} criterion {criterion_id!r} has an empty description; the "
            f"description is the spec half of the contract, and without it the file is a "
            f"property list rather than a statement about the kernel"
        )
        raise ValueError(msg)

    return Criterion(
        id=criterion_id, description=description, property_name=str(check[PROPERTY_CHECK])
    )


def load_contract(path: Path | str) -> Contract:
    """Parse and validate one ``acceptance.yaml``.

    Validation here is structural only — it does not consult the registries, which
    is ``oracle_from_contract``'s job — but it is strict about the two things that
    would otherwise degrade an arm in silence.

    An absent or empty ``criteria`` list is rejected (rather than producing an
    oracle that judges nothing) for the same reason ``validate_property_set``
    rejects an empty property set: it summarizes to INCONCLUSIVE on every group and
    adds groups that established nothing to the denominator of the detection rate.
    ``DeclarativeOracle`` would in fact catch this one downstream, but only for a
    caller who got as far as building an oracle; rejecting it at load keeps the
    diagnostic pointing at the file, which is where the fix is.

    A repeated property is rejected too. It is not harmless duplication: every
    per-property tally the project reports counts ``PropertyResult`` rows, so a name
    listed twice inflates both halves of a rate nobody would think to distrust.
    """
    path = Path(path)
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        msg = f"contract {path} is not a mapping; got {type(document).__name__}"
        raise ValueError(msg)

    task_id = str(document.get("task_id") or "").strip()
    if not task_id:
        msg = (
            f"contract {path} declares no task_id; a contract naming no task cannot be "
            f"joined to the corpus it is meant to score"
        )
        raise ValueError(msg)

    raw_criteria = document.get("criteria") or []
    if not isinstance(raw_criteria, list) or not raw_criteria:
        msg = (
            f"contract {path} for task {task_id!r} declares no criteria; an oracle with "
            f"no properties judges nothing, summarizes to INCONCLUSIVE, and silently adds "
            f"groups that established nothing to the denominator of the detection rate"
        )
        raise ValueError(msg)

    criteria = tuple(_criterion_from(path, task_id, raw) for raw in raw_criteria)

    ids = [criterion.id for criterion in criteria]
    duplicate_ids = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicate_ids:
        msg = (
            f"contract {path} for task {task_id!r} has duplicate criterion ids "
            f"{duplicate_ids}; two obligations under one id are not separately traceable"
        )
        raise ValueError(msg)

    seen: set[str] = set()
    for criterion in criteria:
        if criterion.property_name in seen:
            msg = (
                f"contract {path} names property {criterion.property_name!r} twice; it "
                f"would be evaluated twice and counted twice in every per-property tally"
            )
            raise ValueError(msg)
        seen.add(criterion.property_name)

    return Contract(task_id=task_id, criteria=criteria)


def oracle_from_contract(contract: Contract) -> DeclarativeOracle:
    """Build the declarative arm a contract describes.

    Each name is looked up in ``CASE_PROPERTY_REGISTRY`` first and
    ``GROUP_PROPERTY_REGISTRY`` second — the two share one name pool, and a property
    is registered in exactly one of them — so the contract never has to state a
    property's scope. Stating it would be a second source of truth for something the
    property class already knows, and a contract that got it wrong would place a
    group property where ``check`` is called on it, failing mid-run.

    The returned oracle's constructor runs ``validate_property_set``, so a contract
    whose set defers a non-finite finding to a property it does not select is
    rejected here, before any evaluation. That check cannot be dropped in favour of
    "the author will notice": the failure it prevents produces no error at all, only
    an arm that abstains on every NaN and records a clean miss.
    """
    case_properties: list[CaseProperty] = []
    group_properties: list[GroupProperty] = []
    for name in contract.property_names:
        if name in CASE_PROPERTY_REGISTRY:
            case_properties.append(CASE_PROPERTY_REGISTRY[name]())
        elif name in GROUP_PROPERTY_REGISTRY:
            group_properties.append(GROUP_PROPERTY_REGISTRY[name]())
        else:
            known = sorted({*CASE_PROPERTY_REGISTRY, *GROUP_PROPERTY_REGISTRY})
            msg = (
                f"contract for task {contract.task_id!r} names property {name!r}, which is "
                f"not registered; known properties are {known}"
            )
            raise UnknownPropertyError(msg)
    return DeclarativeOracle(case_properties, group_properties)
