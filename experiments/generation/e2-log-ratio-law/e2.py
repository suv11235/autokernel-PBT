"""E2 for softmax: the log-ratio law, run over every kernel set the pre-registration names.

Feature 0014 added ``rows_have_constant_log_ratio`` and a completed softmax bundle (the current
one plus that law). This script measures what E2's success criterion asks of it: the families
the current gate admits are rejected by the completed one, and no correct kernel acquires a
false positive. Kernel sets A-E, inputs and predictions were registered first:
``docs/measurements/2026-10-14-e2-log-ratio-law-prereg.md``.

    cd experiments/generation/e2-log-ratio-law && ../../../.venv/bin/python e2.py
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
sys.path.insert(0, str(HERE.parent / "log-ratio-headroom"))

import headroom  # puts src/ on the path, through pbt_gate
import pbt_gate
from saboteurs import COMPLETED, SABOTEURS, first_failing_property

from autokernel_pbt.props.backends.base import (
    OUTPUT_NAME,
    PRIMARY_INPUT,
    ExecutionResult,
)
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.case import BASE_RELATION, Case
from autokernel_pbt.props.contract import load_contract, oracle_from_contract
from autokernel_pbt.props.driver import contract_path
from autokernel_pbt.props.oracle import DeclarativeOracle, summary
from autokernel_pbt.props.properties import RowsHaveConstantLogRatio
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import SOFTMAX, TASKS, softmax_reference
from autokernel_pbt.props.verdict import PropertyResult, Verdict

Kernel = Callable[[np.ndarray], np.ndarray]
ARMS = ("allclose", "reference", "declarative", "hybrid")
BUNDLES = ("current", "completed")
LAW = RowsHaveConstantLogRatio()
TINY32 = float(np.finfo(np.float32).tiny)

#: Registered: set E's draws come from default_rng([SEED_E, k]), outside the corpus generator.
SEED_E = 20261014
SHAPES_E = ((64, 129), (16, 4096))
SIGMAS_WIDE = (1, 4, 16, 64)
SIGMAS_MASKED = (1, 16)
MASK_RATE = 0.25


def after(symmetry: Callable[[np.ndarray], np.ndarray]) -> Kernel:
    """softmax_reference with one output symmetry applied, as the symmetry probe builds it.
    Defined here because ``record-symmetry-probe`` has not merged."""

    def kernel(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(symmetry(softmax_reference(x)))

    return kernel


SYMMETRY: dict[str, Kernel] = {
    "softmax_rows_rolled": after(lambda y: np.roll(y, 1, axis=0)),
    "softmax_columns_reversed": after(lambda y: y[:, ::-1]),
}


# --------------------------------------------------------------------------- #
# Reading the law's verdicts
# --------------------------------------------------------------------------- #

_RATIO = re.compile(r"^(?:ratio=|log-ratio test ratio )(?P<ratio>[-+0-9.e]+|inf|nan)")


@dataclass(frozen=True)
class Judged:
    """The law on one case: its verdict, the ratio it reports (None if the detail has none,
    i.e. a FAIL by positivity, masking or underflow, or an abstention), and the detail."""

    case_id: str
    verdict: Verdict
    ratio: float | None
    detail: str
    differs: bool = True  # the output is not byte-equal to the reference's


def judge(row: ExecutionResult) -> Judged:
    result: PropertyResult = LAW.check(row)
    match = _RATIO.match(result.detail)
    ratio = float(match["ratio"]) if match else None
    differs = True
    if row.outputs.get(OUTPUT_NAME) is not None:
        x = row.case.tensors[PRIMARY_INPUT]
        with np.errstate(all="ignore"):
            differs = not np.array_equal(row.outputs[OUTPUT_NAME], softmax_reference(x))
    return Judged(row.case.case_id, result.verdict, ratio, result.detail, differs)


def gate_rows(kernel: Kernel) -> list[ExecutionResult]:
    return [r for rows in pbt_gate._rows_by_group(kernel, SOFTMAX).values() for r in rows]


def worst_ratio(judged: list[Judged]) -> float:
    return max(j.ratio for j in judged if j.ratio is not None)


def checks_failed(judged: list[Judged]) -> str:
    """Which of the law's checks failed, by the first words of each FAIL detail."""
    kinds = {
        "log-ratio": "spread",
        "masked": "masked",
        "negative": "positivity",
        "output": "underflow",
    }
    seen = sorted({kinds[j.detail.split()[0]] for j in judged if j.verdict is Verdict.FAIL})
    return ", ".join(seen) or "-"


# --------------------------------------------------------------------------- #
# Sets A, B, C: through the gate, under both bundles
# --------------------------------------------------------------------------- #


def gate_cells(kernel: Kernel, bundle: str) -> tuple[dict[str, tuple[int, str]], bool]:
    verdict = pbt_gate.check_task(kernel, "softmax", bundle=bundle)
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
        observed[name] = {}
        for bundle in BUNDLES:
            cells, admitted = gate_cells(kernel, bundle)
            observed[name][bundle] = (cells, admitted)
            fails = sum(j.verdict is Verdict.FAIL for j in judged)
            print(
                f"| {name} | {bundle} | "
                + " | ".join(_cell(cells[a]) for a in ARMS)
                + f" | {'admits' if admitted else 'rejects'} | {worst_ratio(judged):.3g} | "
                f"{fails}/{len(judged)} |"
            )
    return observed


def contradictions(observed: dict[str, dict[str, tuple]]) -> list[str]:
    """Where the completed gate contradicts the CURRENT record (P3)."""
    out = []
    for name, saboteur in SABOTEURS.items():
        cells, admitted = observed[name]["completed"]
        for arm in ARMS:
            if cells[arm] != saboteur.expected[arm]:
                out.append(f"{name}/{arm}: {saboteur.expected[arm]} -> {cells[arm]}")
        if admitted is not saboteur.admitted:
            out.append(f"{name}/admitted: {saboteur.admitted} -> {admitted}")
    return out


def loop_candidates() -> dict[str, Kernel]:
    """The 26 iterations of results.tsv: the 25 candidate files, and the kept winner."""
    kernels = {
        f"candidates/{p.stem}": headroom._module_softmax(p)
        for p in sorted((LOOP / "candidates").glob("*.py"))
    }
    kernels["best/train.py"] = headroom._module_softmax(LOOP / "best" / "train.py")
    return kernels


# --------------------------------------------------------------------------- #
# Set D: the recorded A10 runs, offline
# --------------------------------------------------------------------------- #


def device_runs() -> None:
    print(
        "| run | task | groups | executions | current FAIL / INC | completed FAIL / INC |"
        " law INC | law worst ratio |"
    )
    print("|---" * 8 + "|")
    worst_all = 0.0
    for name in headroom.DEVICE_RUNS:
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
            f"| {name} | {task.task_id} | {len(groups)} | {len(rows)} | "
            f"{cur.count(Verdict.FAIL)} / {cur.count(Verdict.INCONCLUSIVE)} | "
            f"{com.count(Verdict.FAIL)} / {com.count(Verdict.INCONCLUSIVE)} | "
            f"{sum(j.verdict is Verdict.INCONCLUSIVE for j in judged)} | {worst:.3g} |"
        )
    print(f"\n- A10, the law's worst ratio over all executions: {worst_all:.3g}")


# --------------------------------------------------------------------------- #
# Set E: wide and masked rows, offline
# --------------------------------------------------------------------------- #


def set_e_inputs() -> Iterator[tuple[str, np.ndarray]]:
    """Registered: wide N(0, sigma^2) rows, then masked rows at a quarter of the entries with one
    entry per row kept, as -inf and as an additive -1e9. One generator per draw, index k."""
    k = 0
    for shape in SHAPES_E:
        for sigma in SIGMAS_WIDE:
            rng = np.random.default_rng([SEED_E, k])
            k += 1
            yield f"wide s={sigma} {shape}", rng.normal(0.0, sigma, shape).astype(np.float32)
    for shape in SHAPES_E:
        for sigma in SIGMAS_MASKED:
            rng = np.random.default_rng([SEED_E, k])
            k += 1
            base = rng.normal(0.0, sigma, shape).astype(np.float32)
            mask = rng.random(shape) < MASK_RATE
            mask[np.arange(shape[0]), rng.integers(0, shape[1], shape[0])] = False
            for label, value in (("-inf", -np.inf), ("-1e9", -1e9)):
                x = base.copy()
                x[mask] = np.float32(value)
                yield f"masked {label} s={sigma} {shape}", x


def run_on(kernel: Kernel, x: np.ndarray) -> ExecutionResult:
    case = Case(
        case_id="e",
        group_id="e",
        relation=BASE_RELATION,
        task_id="softmax",
        dtype=str(x.dtype),
        shape=x.shape,
        tensors={PRIMARY_INPUT: x},
    )
    with np.errstate(all="ignore"):
        return NumpyBackend().run(kernel, case)


WRONG_E: dict[str, Kernel] = {
    "softmax(2x)": headroom.WRONG["softmax(2x)"],
    "softmax(-x)": headroom.WRONG["softmax(-x)"],
    "constant 1/n fill": headroom.WRONG["constant 1/n fill"],
    "rows rolled by one": SYMMETRY["softmax_rows_rolled"],
    "columns reversed": SYMMETRY["softmax_columns_reversed"],
}


def _cell_e(j: Judged) -> str:
    if j.verdict is Verdict.FAIL:
        return f"FAIL ({checks_failed([j])})" + ("" if j.ratio is None else f" {j.ratio:.3g}")
    if j.verdict is Verdict.PASS:
        return f"PASS {j.ratio:.3g}" + ("" if j.differs else " (= ref)")
    return f"{j.verdict} ({j.detail})"


def wide_and_masked() -> None:
    correct = headroom.correct_cpu_kernels()
    fastexp = headroom.reported_kernels()["fast-exp (accuracy budget)"]
    print(
        "| input | correct pass | correct worst ratio (kernel) | zeros | subnormals | "
        f"fast-exp | {' | '.join(WRONG_E)} |"
    )
    print("|---" * (6 + len(WRONG_E)) + "|")
    worst_overall = (0.0, "")
    not_passed, misses = [], []
    for label, x in set_e_inputs():
        judged = {name: judge(run_on(kernel, x)) for name, kernel in correct.items()}
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
        # Underflow counts on the stock float32 kernel, at finite inputs only.
        stock = run_on(headroom.softmax_beta(1.0), x).outputs[OUTPUT_NAME]
        finite = np.isfinite(x)
        zeros = int(np.sum((stock == 0) & finite))
        subnormals = int(np.sum((stock > 0) & (stock < TINY32)))
        wrong = {name: judge(run_on(kernel, x)) for name, kernel in WRONG_E.items()}
        misses += [
            f"{n} on {label}: {j.detail}"
            for n, j in wrong.items()
            if j.differs and j.verdict is Verdict.PASS
        ]
        print(
            f"| {label} | {passed}/{len(correct)} | {rated[top]:.3g} ({top}) | {zeros} | "
            f"{subnormals} | {_cell_e(judge(run_on(fastexp, x)))} | "
            + " | ".join(_cell_e(j) for j in wrong.values())
            + " |"
        )
    print(f"\n- correct kernels, worst ratio on set E: {worst_overall[0]:.3g} ({worst_overall[1]})")
    print(f"- correct kernel cases not passed: {len(not_passed)}")
    for line in not_passed:
        print(f"  - {line}")
    print(f"- wrong kernels passed on a case where they differ: {len(misses)}")
    for line in misses:
        print(f"  - {line}")


# --------------------------------------------------------------------------- #


def p0() -> None:
    """The law's ratio against the headroom record's exploratory statistic (P0)."""
    cpu = [judge(r) for k in headroom.correct_cpu_kernels().values() for r in gate_rows(k)]
    fe = [judge(r) for r in gate_rows(headroom.reported_kernels()["fast-exp (accuracy budget)"])]
    wrong = [judge(r) for k in headroom.WRONG.values() for r in gate_rows(k)]
    differing = [j.ratio for j in wrong if j.differs and j.ratio is not None]
    unrated = [j for j in wrong if j.differs and j.ratio is None]
    print(
        f"- correct CPU kernels, {len(cpu)} gate cases: worst ratio {worst_ratio(cpu):.3g}, "
        f"verdicts {sorted({str(j.verdict) for j in cpu})}"
    )
    print(f"- fast-exp, 18 gate cases: worst ratio {worst_ratio(fe):.3g}")
    print(
        f"- wrong families, differing cases: weakest ratio {min(differing):.3g} over "
        f"{len(differing)} cases; FAILs without a ratio: {len(unrated)}"
    )


def main() -> None:
    print(f"numpy {np.__version__}, GATE_SEED={pbt_gate.GATE_SEED}, threshold 30")

    print("\n## P0: the law against the headroom record (gate cases)\n")
    p0()

    print("\n## Set A: the saboteur matrix, both bundles\n")
    observed = print_gate_table({n: s.kernel for n, s in SABOTEURS.items()})
    recorded = {n: COMPLETED[n] for n in SABOTEURS}
    mismatched = [n for n in SABOTEURS if observed[n]["completed"] != recorded[n]]
    print(f"\n- completed cells differing from saboteurs.COMPLETED: {mismatched or 'none'}")
    current_drift = [
        n for n, s in SABOTEURS.items() if observed[n]["current"] != (dict(s.expected), s.admitted)
    ]
    print(f"- current cells differing from the current record: {current_drift or 'none'}")
    found = contradictions(observed)
    print(f"- the current record, read against the completed gate: {len(found)} contradictions")
    for line in found:
        print(f"  - {line}")

    print("\n## Set B: the two softmax symmetry kernels, both bundles\n")
    print_gate_table(SYMMETRY)

    print("\n## Set C: the 26 loop candidates, both bundles\n")
    candidates = loop_candidates()
    observed_c = print_gate_table(candidates)
    rejected = [n for n, o in observed_c.items() if not o["completed"][1]]
    moved = [
        n
        for n, o in observed_c.items()
        if {a: o["completed"][0][a] for a in ARMS if a != "declarative"}
        != {a: o["current"][0][a] for a in ARMS if a != "declarative"}
    ]
    print(
        f"\n- {len(candidates)} candidates; rejected under the completed bundle: "
        f"{rejected or 'none'}; non-declarative arms that moved: {moved or 'none'}"
    )

    print("\n## Set D: the recorded A10 runs, offline\n")
    device_runs()

    print("\n## Set E: wide and masked rows, offline\n")
    wide_and_masked()


if __name__ == "__main__":
    main()
