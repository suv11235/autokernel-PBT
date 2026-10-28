"""Log-ratio headroom: the completing law's statistic on every softmax we hold.

Proposition 1(b): the softmax bundle plus the log-ratio law (log y_ij - x_ij constant along
each row) admits softmax and nothing else, over the reals. This script measures, per case,
how far a kernel's output is from satisfying that law in float32 -- the worst row spread of
log y - x, in float32 eps -- on correct kernels (NumPy on the gate's cases, and the recorded
A10 Triton runs) and on the wrong families the gate admits today. It changes no bundle, arm
or gate. The question, the threshold (30) and the predictions were registered first:
``docs/measurements/2026-10-14-log-ratio-headroom-prereg.md``.

    cd experiments/generation/log-ratio-headroom && ../../../.venv/bin/python headroom.py
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOOP = HERE.parent / "autoresearch-pbt"
REPO = HERE.parents[2]
sys.path.insert(0, str(LOOP))

import pbt_gate  # puts src/ on the path
from saboteurs import softmax_beta, uniform_fill, unnormalized

from autokernel_pbt.props.backends.base import OUTPUT_NAME, PRIMARY_INPUT
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference

EPS32 = float(np.finfo(np.float32).eps)
TINY32 = float(np.finfo(np.float32).tiny)
#: Registered before the first run: the gate's existing ratio threshold.
THRESHOLD = 30.0
LOG2E = np.float32(1.4426950408889634)
DEVICE_RUNS = ("gpu-softmax", "gpu-tolerance_sweep", "gpu-scale")

Kernel = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Spread:
    """One case's statistic, and where it came from."""

    case_id: str
    shape: tuple[int, ...]
    eps: float  # worst row spread of log y - x, in float32 eps
    row_range: float  # max x - min x on that worst row
    zeros: int  # outputs exactly 0, excluded from the spread
    subnormals: int  # outputs below float32's smallest normal
    #: EXPLORATORY, not registered: the worst row's spread / (1 + its row range), in eps.
    per_range: float = 0.0
    #: False when the kernel's output equals the reference's exactly, so it is not wrong here.
    differs: bool = True


def spread(case_id: str, x: np.ndarray, y: np.ndarray) -> Spread:
    """The statistic the pre-registration defines, on one case."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x2 = x.reshape(-1, x.shape[-1])
    y2 = y.reshape(-1, y.shape[-1])
    positive = y2 > 0
    zeros = int(np.sum(y2 == 0))
    subnormals = int(np.sum(positive & (y2 < TINY32)))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(positive, np.log(np.where(positive, y2, 1.0)) - x2, np.nan)
    hi = np.max(np.where(positive, d, -np.inf), axis=-1)
    lo = np.min(np.where(positive, d, np.inf), axis=-1)
    rows = np.where(positive.any(axis=-1), hi - lo, 0.0)
    worst = int(np.argmax(rows))
    per_range = rows / (1.0 + np.ptp(x2, axis=-1))
    return Spread(
        case_id=case_id,
        shape=tuple(x.shape),
        eps=float(rows[worst]) / EPS32,
        row_range=float(np.ptp(x2[worst])),
        zeros=zeros,
        subnormals=subnormals,
        per_range=float(np.max(per_range)) / EPS32,
    )


def on_gate_cases(kernel: Kernel) -> list[Spread]:
    """The gate's own 18 cases (seed 42, 9 groups, base + shift_rows), as ``pbt_gate`` runs them."""
    out = []
    for rows in pbt_gate._rows_by_group(kernel, SOFTMAX).values():
        for row in rows:
            x, y = row.case.tensors[PRIMARY_INPUT], row.outputs[OUTPUT_NAME]
            s = spread(row.case.case_id, x, y)
            out.append(replace(s, differs=not np.array_equal(y, softmax_reference(x))))
    return out


def on_device_run(name: str) -> list[Spread]:
    """A recorded A10 run, read offline; its inputs and outputs are the recorded bytes."""
    return [
        spread(r.case.case_id, r.case.tensors[PRIMARY_INPUT], r.outputs[OUTPUT_NAME])
        for r in ExecutionTable(REPO / "runs" / name).read()
    ]


def softmax_exp2(x: np.ndarray) -> np.ndarray:
    """NumPy float32 analogue of the correct Triton rewrite ``_softmax_exp2``."""
    e = np.exp2((x - np.max(x, axis=-1, keepdims=True)) * LOG2E)
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_reciprocal(x: np.ndarray) -> np.ndarray:
    """NumPy float32 analogue of the correct Triton rewrite ``_softmax_reciprocal``."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e * (np.float32(1.0) / np.sum(e, axis=-1, keepdims=True))


def _module_softmax(path: Path) -> Kernel:
    """A loop candidate's own ``softmax``, loaded from its file rather than copied."""
    spec = importlib.util.spec_from_file_location(f"_headroom_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.softmax


FASTEXP = "r3_softmax_fastexp"


def correct_cpu_kernels() -> dict[str, Kernel]:
    kernels: dict[str, Kernel] = {
        "reference (float64, cast)": softmax_reference,
        "stock float32": softmax_beta(1.0),
        "exp2 analogue": softmax_exp2,
        "reciprocal-multiply analogue": softmax_reciprocal,
        "best/train.py": _module_softmax(LOOP / "best" / "train.py"),
    }
    for path in sorted((LOOP / "candidates").glob("*.py")):
        if path.stem != FASTEXP:
            kernels[f"candidates/{path.stem}"] = _module_softmax(path)
    return kernels


WRONG: dict[str, Kernel] = {
    "softmax(2x)": softmax_beta(2.0),
    "softmax(0x)": softmax_beta(0.0),
    "softmax(-x)": softmax_beta(-1.0),
    "constant 1/n fill": uniform_fill,
    # The symmetry probe's two softmax rows (docs/measurements/2026-10-14-symmetry-probe.md),
    # built the same way: the reference's output with one symmetry applied.
    "rows rolled by one": lambda x: np.roll(softmax_reference(x), 1, axis=0),
    "columns reversed": lambda x: np.ascontiguousarray(softmax_reference(x)[:, ::-1]),
}


def reported_kernels() -> dict[str, Kernel]:
    return {
        "fast-exp (accuracy budget)": _module_softmax(LOOP / "candidates" / f"{FASTEXP}.py"),
        "exp(x - max), undivided (row-sum law's)": unnormalized,
    }


def worst(spreads: Iterable[Spread]) -> Spread:
    return max(spreads, key=lambda s: s.eps)


def least_multicolumn(spreads: Iterable[Spread]) -> Spread:
    """A wrong family's weakest case among those it can be caught on: one-column rows are
    softmax's identity, and a rolled one-row case is the kernel's identity too."""
    return min((s for s in spreads if s.eps > 0), key=lambda s: s.eps)


def _row(label: str, cls: str, s: Spread, spreads: list[Spread]) -> str:
    zeros = sum(t.zeros for t in spreads)
    subs = sum(t.subnormals for t in spreads)
    verdict = "accept" if s.eps < THRESHOLD else "reject"
    return (
        f"| {label} | {cls} | {s.eps:.3g} | {s.shape} | {s.row_range:.3g} | {zeros} | {subs} | "
        f"{verdict} |"
    )


def main() -> None:
    print(f"numpy {np.__version__}, GATE_SEED={pbt_gate.GATE_SEED}, threshold {THRESHOLD:g} eps")
    header = "| kernel | class | spread (eps) | at shape | row range | zeros | subnormals | at 30 |"
    print("\n### CPU, the gate's 18 cases per kernel (worst case shown; wrong rows: weakest)\n")
    print(header)
    print("|---" * 8 + "|")
    cpu_correct = []
    for label, kernel in correct_cpu_kernels().items():
        spreads = on_gate_cases(kernel)
        cpu_correct += spreads
        print(_row(label, "correct", worst(spreads), spreads))
    wrong_least, wrong_all = [], []
    for label, kernel in WRONG.items():
        spreads = on_gate_cases(kernel)
        wrong_all.append(spreads)
        least = least_multicolumn(spreads)
        wrong_least.append(least)
        print(_row(label, "wrong", least, spreads))
    reported = {}
    for label, kernel in reported_kernels().items():
        spreads = on_gate_cases(kernel)
        reported[label] = worst(spreads)
        print(_row(label, "reported", worst(spreads), spreads))

    print("\n### A10, recorded Triton softmax, every execution\n")
    print("| run | case | shape | spread (eps) | row range | zeros | subnormals | at 30 |")
    print("|---" * 8 + "|")
    device = []
    for name in DEVICE_RUNS:
        for s in on_device_run(name):
            device.append(s)
            verdict = "accept" if s.eps < THRESHOLD else "reject"
            print(
                f"| {name} | {s.case_id} | {s.shape} | {s.eps:.3g} | {s.row_range:.3g} | "
                f"{s.zeros} | {s.subnormals} | {verdict} |"
            )

    max_cpu, max_dev = worst(cpu_correct), worst(device)
    max_correct = max(max_cpu.eps, max_dev.eps)
    min_wrong = min(s.eps for s in wrong_least)
    print("\n### Summary\n")
    print(
        f"- correct, CPU: {len(cpu_correct)} cases, max {max_cpu.eps:.3g} eps ({max_cpu.case_id})"
    )
    print(
        f"- correct, A10: {len(device)} executions, max {max_dev.eps:.3g} eps ({max_dev.case_id})"
    )
    print(
        f"- correct zeros / subnormals: "
        f"{sum(s.zeros for s in cpu_correct + device)} / "
        f"{sum(s.subnormals for s in cpu_correct + device)}"
    )
    print(f"- wrong, weakest multi-column case: {min_wrong:.3g} eps")
    for label, s in reported.items():
        print(f"- {label}: {s.eps:.3g} eps")
    print(f"- margin, weakest wrong / largest correct: {min_wrong / max_correct:.3g}")

    # POST HOC, added after the registered numbers were seen; see the record's deviations.
    differing = [s for spreads in wrong_all for s in spreads if s.differs]
    weakest = min(differing, key=lambda s: s.eps)
    print("\n### Post hoc (not registered)\n")
    print(
        f"- wrong, weakest case whose output differs from the reference: {weakest.eps:.3g} eps "
        f"({weakest.case_id}, {weakest.shape}); margin over the largest correct: "
        f"{weakest.eps / max_correct:.3g}"
    )
    print(
        f"- exploratory spread / (1 + row range), in eps: correct CPU max "
        f"{max(s.per_range for s in cpu_correct):.3g}, correct A10 max "
        f"{max(s.per_range for s in device):.3g}, "
        + ", ".join(f"{label} {s.per_range:.3g}" for label, s in reported.items())
        + f", wrong differing min {min(s.per_range for s in differing):.3g}"
    )


if __name__ == "__main__":
    main()
