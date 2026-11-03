"""E2 for layernorm: the affine-tie law, run over every kernel set the pre-registration names.

Feature 0015 added ``rows_have_affine_tie`` and a completed layernorm bundle (the current one plus
that law). This script measures what E2's success criterion asks of it: the families the current
gate admits are rejected by the completed one, and no correct kernel acquires a false positive.
Kernel sets A-D, inputs and predictions were registered first:
``docs/measurements/2026-11-02-e2-affine-tie-law-prereg.md``.

    cd experiments/generation/e2-affine-tie-law && ../../../.venv/bin/python e2_layernorm.py
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOOP = HERE.parent / "autoresearch-pbt"
REPO = HERE.parents[2]
sys.path.insert(0, str(LOOP))
sys.path.insert(0, str(HERE.parent / "affine-tie-headroom"))

import pbt_gate  # puts src/ on the path
import tie_headroom
from saboteurs import LAYERNORM_COMPLETED, LAYERNORM_SABOTEURS, first_failing_property

from autokernel_pbt.props.backends.base import OUTPUT_NAME, PRIMARY_INPUT, ExecutionResult
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.case import BASE_RELATION, Case
from autokernel_pbt.props.contract import load_contract, oracle_from_contract
from autokernel_pbt.props.driver import contract_path
from autokernel_pbt.props.oracle import DeclarativeOracle, ReferenceOracle, summary
from autokernel_pbt.props.properties import RowsHaveAffineTie
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import LAYERNORM, TASKS, layernorm_reference
from autokernel_pbt.props.verdict import PropertyResult, Verdict

Kernel = Callable[[np.ndarray], np.ndarray]
ARMS = ("allclose", "reference", "declarative", "hybrid")
BUNDLES = ("current", "completed")
LAW = RowsHaveAffineTie()
REFERENCE_ARM = ReferenceOracle(layernorm_reference)

#: Registered: set D's draws come from default_rng([SEED_D, k]), outside the corpus generator.
SEED_D = 20261102
SHAPES_D = ((64, 129), (16, 4096))
SIGMAS = (1e-3, 1.0, 1e3, 1e6)
OFFSETS = (10.0, 100.0, 1e3, 1e4, 1e6)

# --------------------------------------------------------------------------- #
# Reading the law's verdicts
# --------------------------------------------------------------------------- #

_RATIO = re.compile(r"^(?:ratio=|affine-tie test ratio )(?P<ratio>[-+0-9.e]+|inf|nan)")


@dataclass(frozen=True)
class Judged:
    """The law on one case: its verdict, the ratio it reports (None on a slope FAIL or an
    abstention), the detail, and whether the output differs from the reference's."""

    case_id: str
    verdict: Verdict
    ratio: float | None
    detail: str
    differs: bool = True


def judge(row: ExecutionResult) -> Judged:
    result: PropertyResult = LAW.check(row)
    match = _RATIO.match(result.detail)
    x = row.case.tensors[PRIMARY_INPUT]
    with np.errstate(all="ignore"):
        differs = not np.array_equal(row.outputs[OUTPUT_NAME], layernorm_reference(x))
    return Judged(
        row.case.case_id,
        result.verdict,
        float(match["ratio"]) if match else None,
        result.detail,
        differs,
    )


def gate_rows(kernel: Kernel) -> list[ExecutionResult]:
    return [r for rows in pbt_gate._rows_by_group(kernel, LAYERNORM).values() for r in rows]


def worst_ratio(judged: list[Judged]) -> float:
    return max(j.ratio for j in judged if j.ratio is not None)


def check_of(j: Judged) -> str:
    """Which of the law's two checks a FAIL came from."""
    return "tie" if j.detail.startswith("affine-tie") else "slope"


# --------------------------------------------------------------------------- #
# Sets A and B: through the gate, under both bundles
# --------------------------------------------------------------------------- #


def gate_cells(kernel: Kernel, bundle: str) -> tuple[dict[str, tuple[int, str]], bool]:
    verdict = pbt_gate.check_task(kernel, "layernorm", bundle=bundle)
    cells = {a.arm: (a.groups_failed, first_failing_property(a)) for a in verdict.arms}
    return cells, verdict.passed


def _cell(cell: tuple[int, str]) -> str:
    failed, prop = cell
    return f"{failed}/9" + (f" ({prop})" if failed else "")


def print_gate_table(kernels: dict[str, Kernel]) -> dict[str, dict[str, tuple]]:
    print("| row | bundle | " + " | ".join(ARMS) + " | gate | law: worst ratio | law FAILs |")
    print("|---" * (len(ARMS) + 5) + "|")
    observed: dict[str, dict[str, tuple]] = {}
    for name, kernel in kernels.items():
        judged = [judge(r) for r in gate_rows(kernel)]
        worst = max((j.ratio for j in judged if j.ratio is not None), default=float("nan"))
        fails = [j for j in judged if j.verdict is Verdict.FAIL]
        checks = ", ".join(sorted({check_of(j) for j in fails}))
        observed[name] = {}
        for bundle in BUNDLES:
            cells, admitted = gate_cells(kernel, bundle)
            observed[name][bundle] = (cells, admitted)
            print(
                f"| {name} | {bundle} | "
                + " | ".join(_cell(cells[a]) for a in ARMS)
                + f" | {'admits' if admitted else 'rejects'} | {worst:.3g} | "
                f"{len(fails)}/{len(judged)}{f' ({checks})' if checks else ''} |"
            )
    return observed


def contradictions(observed: dict[str, dict[str, tuple]]) -> list[str]:
    """Where the completed gate contradicts the CURRENT record (P3)."""
    out = []
    for name, saboteur in LAYERNORM_SABOTEURS.items():
        cells, admitted = observed[name]["completed"]
        for arm in ARMS:
            if cells[arm] != saboteur.expected[arm]:
                out.append(f"{name}/{arm}: {saboteur.expected[arm]} -> {cells[arm]}")
        if admitted is not saboteur.admitted:
            out.append(f"{name}/admitted: {saboteur.admitted} -> {admitted}")
    return out


def judgement_on_the_reference() -> list[tuple[str, str]]:
    """P4: every declarative result that is not a PASS, on the reference, under the completed
    bundle. Built here from the same contract the gate's arm is built from."""
    arm = oracle_from_contract(pbt_gate.bundle_contract("layernorm", "completed"))
    out = []
    for rows in pbt_gate._rows_by_group(layernorm_reference, LAYERNORM).values():
        out += [
            (r.property_name, r.case_id)
            for r in arm.evaluate(rows)
            if r.verdict is not Verdict.PASS
        ]
    return sorted(out)


def loop_candidates() -> dict[str, Kernel]:
    """The 26 iterations of results.tsv: the 25 candidate files, and the kept winner."""
    kernels = {
        f"candidates/{p.stem}": tie_headroom._module_layernorm(p)
        for p in sorted((LOOP / "candidates").glob("*.py"))
    }
    kernels["best/train.py"] = tie_headroom._module_layernorm(LOOP / "best" / "train.py")
    return kernels


# --------------------------------------------------------------------------- #
# Set C: the recorded device runs, offline
# --------------------------------------------------------------------------- #


def device_runs() -> None:
    print(
        "| run | groups | executions | current FAIL / INC | completed FAIL / INC | law INC | "
        "law worst ratio |"
    )
    print("|---" * 7 + "|")
    worst_all = 0.0
    for name in tie_headroom.DEVICE_RUNS:
        groups = ExecutionTable(REPO / "runs" / name).read_groups()
        rows = [r for g in groups.values() for r in g]
        task = TASKS[rows[0].case.task_id]
        current = oracle_from_contract(load_contract(contract_path(task, REPO)))
        completed = DeclarativeOracle([*current.case_properties, LAW], current.group_properties)
        cur = [summary(current.evaluate(g)) for g in groups.values()]
        com = [summary(completed.evaluate(g)) for g in groups.values()]
        judged = [judge(r) for r in rows]
        worst = worst_ratio(judged)
        worst_all = max(worst_all, worst)
        print(
            f"| {name} | {len(groups)} | {len(rows)} | "
            f"{cur.count(Verdict.FAIL)} / {cur.count(Verdict.INCONCLUSIVE)} | "
            f"{com.count(Verdict.FAIL)} / {com.count(Verdict.INCONCLUSIVE)} | "
            f"{sum(j.verdict is Verdict.INCONCLUSIVE for j in judged)} | {worst:.3g} |"
        )
    print(f"\n- devices, the law's worst ratio over all executions: {worst_all:.3g}")


# --------------------------------------------------------------------------- #
# Set D: scaled, offset and constant rows, offline
# --------------------------------------------------------------------------- #


def set_d_inputs() -> Iterator[tuple[str, np.ndarray]]:
    """Registered: one generator per draw, index k, in this order."""
    k = 0
    for shape in SHAPES_D:
        for sigma in SIGMAS:
            rng = np.random.default_rng([SEED_D, k])
            k += 1
            yield f"scaled s={sigma:g} {shape}", rng.normal(0.0, sigma, shape).astype(np.float32)
        for mu in OFFSETS:
            rng = np.random.default_rng([SEED_D, k])
            k += 1
            yield f"offset m={mu:g} {shape}", (mu + rng.normal(0.0, 1.0, shape)).astype(np.float32)
        rng = np.random.default_rng([SEED_D, k])
        k += 1
        x = rng.normal(0.0, 1.0, shape).astype(np.float32)
        x[::2] = rng.normal(0.0, 1.0, (x[::2].shape[0], 1)).astype(np.float32)
        yield f"constant rows {shape}", x


def run_on(kernel: Kernel, x: np.ndarray) -> ExecutionResult:
    case = Case(
        case_id="d",
        group_id="d",
        relation=BASE_RELATION,
        task_id="layernorm",
        dtype=str(x.dtype),
        shape=x.shape,
        tensors={PRIMARY_INPUT: x},
    )
    with np.errstate(all="ignore"):
        return NumpyBackend().run(kernel, case)


def _cell_d(j: Judged) -> str:
    if j.verdict is Verdict.FAIL:
        return f"FAIL ({check_of(j)})" + ("" if j.ratio is None else f" {j.ratio:.3g}")
    if j.verdict is Verdict.PASS:
        return f"PASS {j.ratio:.3g}" + ("" if j.differs else " (= ref)")
    return f"{j.verdict} ({j.detail})"


def set_d() -> None:
    correct = tie_headroom.correct_cpu_kernels()
    wrong_kernels = tie_headroom.WRONG
    print(
        "| input | correct pass | correct worst ratio (kernel) | distributed | reference arm on "
        f"distributed | {' | '.join(wrong_kernels)} |"
    )
    print("|---" * (5 + len(wrong_kernels)) + "|")
    worst_overall = (0.0, "")
    not_passed, misses, wrong_checks, not_flat = [], [], [], []
    total = 0
    for label, x in set_d_inputs():
        rows = {name: run_on(kernel, x) for name, kernel in correct.items()}
        judged = {name: judge(row) for name, row in rows.items()}
        total += len(judged)
        passed = sum(j.verdict is Verdict.PASS for j in judged.values())
        not_passed += [
            f"{n} on {label}: {j.detail}"
            for n, j in judged.items()
            if j.verdict is not Verdict.PASS
        ]
        rated = {n: j.ratio for n, j in judged.items() if j.ratio is not None}
        top = max(rated, key=rated.get)
        if rated[top] > worst_overall[0]:
            worst_overall = (rated[top], f"{top}, {label}")
        if label.startswith("constant"):
            not_flat += [
                f"{n} on {label}"
                for n, row in rows.items()
                if np.any(np.ptp(row.outputs[OUTPUT_NAME][::2], axis=-1) != 0)
            ]
        distributed_row = run_on(tie_headroom.layernorm_distributed, x)
        reference_arm = REFERENCE_ARM.evaluate([distributed_row])[0]
        wrong = {name: judge(run_on(kernel, x)) for name, kernel in wrong_kernels.items()}
        misses += [
            f"{n} on {label}: {j.detail}"
            for n, j in wrong.items()
            if j.differs and j.verdict is not Verdict.FAIL
        ]
        wrong_checks += [(n, check_of(j)) for n, j in wrong.items() if j.verdict is Verdict.FAIL]
        print(
            f"| {label} | {passed}/{len(correct)} | {rated[top]:.3g} ({top}) | "
            f"{_cell_d(judge(distributed_row))} | {reference_arm.verdict} {reference_arm.detail} | "
            + " | ".join(_cell_d(j) for j in wrong.values())
            + " |"
        )
    print(f"\n- correct kernels, worst ratio on set D: {worst_overall[0]:.3g} ({worst_overall[1]})")
    print(f"- correct kernel cases: {total}; not passed: {len(not_passed)}")
    for line in not_passed:
        print(f"  - {line}")
    print(f"- correct constant rows not exactly flat: {len(not_flat)}")
    for line in not_flat:
        print(f"  - {line}")
    print(f"- wrong kernels not rejected on a case where they differ: {len(misses)}")
    for line in misses:
        print(f"  - {line}")
    for name in wrong_kernels:
        checks = sorted({c for n, c in wrong_checks if n == name})
        count = sum(n == name for n, _ in wrong_checks)
        print(f"- {name}: {count} FAILs, by {', '.join(checks)}")


# --------------------------------------------------------------------------- #


def p0() -> None:
    """The law's ratio against the headroom record's statistic (P0)."""
    cpu = [judge(r) for k in tie_headroom.correct_cpu_kernels().values() for r in gate_rows(k)]
    reference = [judge(r) for r in gate_rows(layernorm_reference)]
    distributed = [judge(r) for r in gate_rows(tie_headroom.layernorm_distributed)]
    wrong = [
        judge(r) for name, k in tie_headroom.WRONG.items() for r in gate_rows(k) if name != "-LN(x)"
    ]
    differing = [j.ratio for j in wrong if j.differs and j.ratio is not None]
    print(
        f"- correct CPU kernels, {len(cpu)} gate cases: worst ratio {worst_ratio(cpu):.3g}, "
        f"verdicts {sorted({str(j.verdict) for j in cpu})}"
    )
    print(f"- the reference, whose negation -LN is: worst ratio {worst_ratio(reference):.3g}")
    print(f"- distributed division: worst ratio {worst_ratio(distributed):.3g}")
    print(f"- reversed, sorted, rolled, differing cases: weakest ratio {min(differing):.3g}")


def main() -> None:
    print(f"numpy {np.__version__}, GATE_SEED={pbt_gate.GATE_SEED}, threshold 30")

    print("\n## P0: the law against the headroom record (gate cases)\n")
    p0()

    print("\n## Set A: the saboteur matrix, both bundles\n")
    observed = print_gate_table({n: s.kernel for n, s in LAYERNORM_SABOTEURS.items()})
    mismatched = [
        n for n in LAYERNORM_SABOTEURS if observed[n]["completed"] != LAYERNORM_COMPLETED[n]
    ]
    print(
        f"\n- completed cells differing from saboteurs.LAYERNORM_COMPLETED: {mismatched or 'none'}"
    )
    drift = [
        n
        for n, s in LAYERNORM_SABOTEURS.items()
        if observed[n]["current"] != (dict(s.expected), s.admitted)
    ]
    print(f"- current cells differing from the current record: {drift or 'none'}")
    found = contradictions(observed)
    print(f"- the current record, read against the completed gate: {len(found)} contradictions")
    for line in found:
        print(f"  - {line}")
    print("- P4, declarative non-PASS results on the reference, completed bundle:")
    for name, case_id in judgement_on_the_reference():
        print(f"  - {name}: {case_id}")

    print("\n## Reported: distributed division through the gate (not registered)\n")
    print_gate_table({"distributed division": tie_headroom.layernorm_distributed})

    print("\n## Set B: the 26 loop candidates, both bundles\n")
    candidates = loop_candidates()
    observed_b = print_gate_table(candidates)
    rejected = [n for n, o in observed_b.items() if not o["completed"][1]]
    moved = [
        n
        for n, o in observed_b.items()
        if {a: o["completed"][0][a] for a in ARMS if a != "declarative"}
        != {a: o["current"][0][a] for a in ARMS if a != "declarative"}
    ]
    print(
        f"\n- {len(candidates)} candidates; rejected under the completed bundle: "
        f"{rejected or 'none'}; non-declarative arms that moved: {moved or 'none'}"
    )

    print("\n## Set C: the recorded device runs, offline\n")
    device_runs()

    print("\n## Set D: scaled, offset and constant rows, offline\n")
    set_d()


if __name__ == "__main__":
    main()
