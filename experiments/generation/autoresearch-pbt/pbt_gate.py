"""The property gate: this repo's four oracle arms, run over a candidate's kernels.

Why this exists, in one line: in kernel optimization the reward and the truth point
in *opposite* directions -- the fastest softmax you can write is one that skips the
max subtraction, and the fastest layernorm is one that skips centering. A loop
gated only on ``val_bpb`` will happily accept both, because a subtly wrong kernel
still trains to a plausible loss. So the objective is scored only *after* a gate
that knows nothing about speed.

Deliberately NOT ``driver.run_task``: that persists a parquet corpus per call, which
is right for a recorded experiment and wrong for a gate that runs on every candidate.
The generation and execution path is identical -- same ``Generator``, same
``NumpyBackend``, same case groups, same contracts -- only the persistence is
dropped.

Reports every arm separately rather than an aggregate, because *which* arm fires is
the measurement this experiment exists to produce.

The declarative arm is built from one of two *bundles*: ``current``, the contract every
recorded number was measured under and the default, and ``completed``, which adds softmax's
log-ratio law (feature 0014, E2). The loop gates on the default.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from autokernel_pbt.props.backends.base import Status
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.contract import (
    COMPLETED_CONTRACT_FILENAME,
    CONTRACT_FILENAME,
    KERNEL_TASKS_DIR,
    Contract,
    load_contract,
    oracle_from_contract,
)
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.oracle import (
    AllcloseOracle,
    HybridOracle,
    ReferenceOracle,
    summary,
)
from autokernel_pbt.props.tasks import LAYERNORM, SOFTMAX, layernorm_reference, softmax_reference
from autokernel_pbt.props.verdict import Verdict

#: The gate's own seed and width. Fixed: a gate whose case set moves between
#: candidates is not a gate, it is a lottery.
GATE_SEED = 42
GATE_GROUPS = 9

TASKS = {
    "softmax": (SOFTMAX, softmax_reference, "softmax"),
    "layernorm": (LAYERNORM, layernorm_reference, "layernorm"),
}

#: The contract file each bundle's declarative arm is built from. Layernorm has no completed
#: bundle yet -- its affine-tie law is not built -- so asking for one is refused, never answered
#: with the current bundle under the completed bundle's name.
BUNDLES = {"current": CONTRACT_FILENAME, "completed": COMPLETED_CONTRACT_FILENAME}
DEFAULT_BUNDLE = "current"


@dataclass
class ArmVerdict:
    arm: str
    verdict: str
    groups_failed: int
    groups_total: int
    first_failure: str


@dataclass
class TaskVerdict:
    task: str
    passed: bool
    arms: list
    #: Why the gate refused the candidate whatever its arms say; empty if it did not.
    refused: str = ""


def _rows_by_group(kernel, task):
    backend = NumpyBackend()
    groups = Generator(task.domain, GATE_SEED).generate(GATE_GROUPS)
    out = {}
    for group in groups:
        rows = []
        for case in group.cases:
            r = backend.run(kernel, case)
            r.kernel_id = "candidate"
            r.case_spec = group.spec
            rows.append(r)
        out[group.group_id] = rows
    return out


def bundle_contract(task_key: str, bundle: str) -> Contract:
    """The contract ``bundle`` builds ``task_key``'s declarative arm from."""
    path = REPO / KERNEL_TASKS_DIR / TASKS[task_key][2] / BUNDLES.get(bundle, "")
    if bundle not in BUNDLES or not path.is_file():
        msg = f"there is no {bundle!r} bundle for task {task_key!r}; bundles: {sorted(BUNDLES)}"
        raise ValueError(msg)
    return load_contract(path)


def check_task(kernel, task_key: str, bundle: str = DEFAULT_BUNDLE) -> TaskVerdict:
    task, reference_fn, _ = TASKS[task_key]
    declarative = oracle_from_contract(bundle_contract(task_key, bundle))
    reference = ReferenceOracle(reference_fn)
    arms = {
        "allclose": AllcloseOracle(reference_fn),
        "reference": reference,
        "declarative": declarative,
        "hybrid": HybridOracle(declarative, reference),
    }
    by_group = _rows_by_group(kernel, task)

    verdicts = []
    judged = {}
    for name, arm in arms.items():
        failed, first = 0, ""
        judged[name] = 0
        for gid, rows in by_group.items():
            results = arm.evaluate(rows)
            if summary(results) is not Verdict.INCONCLUSIVE:
                judged[name] += 1
            if summary(results) is Verdict.FAIL:
                failed += 1
                if not first:
                    bad = next(r for r in results if r.verdict is Verdict.FAIL)
                    first = f"{gid} shape={rows[0].case.shape} {bad.property_name}: {bad.detail}"
        verdicts.append(
            ArmVerdict(
                arm=name,
                verdict="FAIL" if failed else "PASS",
                groups_failed=failed,
                groups_total=len(by_group),
                first_failure=first,
            )
        )
    # The gate's decision uses the DECLARATIVE arm, not allclose. That choice is the
    # experiment: allclose false-positives 5/9 on a correct layernorm
    # (docs/measurements/2026-08-16-allclose-layernorm-false-positives.md), so a loop
    # gated on it cannot reach a correct kernel at all. Every arm is still reported.
    decisive = next(v for v in verdicts if v.arm == "declarative")
    # Every arm counts only FAIL groups, so a candidate that never produced an output reads as
    # 0/9 failed everywhere -- a pass. A kernel that raises on the gate's own shapes while
    # running at the training shapes would game the loop that way. driver.run_task refuses an
    # arm that judged nothing; the gate refuses that, and any case the kernel failed to run.
    unrun = [(gid, r) for gid, rows in by_group.items() for r in rows if r.status != Status.OK]
    refused = ""
    if unrun:
        gid, r = unrun[0]
        total = sum(len(rows) for rows in by_group.values())
        refused = (
            f"the kernel produced no output on {len(unrun)} of {total} cases, first {gid} "
            f"shape={r.case.shape}: {r.status}"
        )
    elif judged["declarative"] == 0:
        refused = "the declarative arm abstained on every group, so nothing was judged"
    return TaskVerdict(
        task=task_key,
        passed=decisive.groups_failed == 0 and not refused,
        arms=verdicts,
        refused=refused,
    )


def gate(module_name: str = "train") -> dict:
    """Import (or reimport) the candidate and check both of its kernels."""
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    results = []
    for key, attr in (("softmax", "softmax"), ("layernorm", "layernorm")):
        fn = getattr(mod, attr, None)
        if fn is None:
            results.append(
                # Filed under the decisive arm: the runner reads first_failure from
                # "declarative" only, so any other name drops the reason on the floor.
                TaskVerdict(task=key, passed=False, arms=[
                    ArmVerdict("declarative", "FAIL", 1, 1, f"train.py exports no {attr}()")
                ])
            )
            continue
        results.append(check_task(fn, key))
    return {
        "passed": all(r.passed for r in results),
        "tasks": [asdict(r) for r in results],
    }


if __name__ == "__main__":
    print(json.dumps(gate(), indent=2))
