"""Affine-tie headroom: the completing law's statistic on every layernorm we hold.

Proposition 1(c): the layernorm bundle plus the affine-tie law (in each row, y is an increasing
affine function of x) admits LN and nothing else on every non-constant row, over the reals. This
script measures, per case, how far a kernel's output is from satisfying that law in float32 -- the
worst row's least-squares residual, against eps * max |y| -- on correct kernels (NumPy on the
gate's cases, and the recorded device runs) and on the wrong families the gate admits today. It
changes no bundle, arm or gate. The question, the threshold (30) and the predictions were
registered first: ``docs/measurements/2026-11-02-affine-tie-headroom-prereg.md``.

    cd experiments/generation/affine-tie-headroom && ../../../.venv/bin/python tie_headroom.py
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOOP = HERE.parent / "autoresearch-pbt"
REPO = HERE.parents[2]
sys.path.insert(0, str(LOOP))

import pbt_gate  # puts src/ on the path

from autokernel_pbt.props.backends.base import OUTPUT_NAME, PRIMARY_INPUT
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import LAYERNORM, LAYERNORM_EPS, layernorm_reference

#: Registered before the first run: the gate's existing ratio threshold.
THRESHOLD = 30.0
DEVICE_RUNS = ("gpu-layernorm", "te1-triton-layernorm", "te1-cuda-layernorm")

Kernel = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Tie:
    """One case's statistic, and where it came from."""

    case_id: str
    shape: tuple[int, ...]
    ratio: float  # the worst row's max |r| / (eps * max |y| + tiny)
    slope: float | None  # the smallest slope over rows with a non-constant input
    top_slope: float | None  # the largest such slope, which P6 reads
    max_y: float  # max |y| on the worst row
    offset: float  # max |x| / max |x - mean x| on the worst row
    #: False when the kernel's output equals the reference's exactly, so it is not wrong here.
    differs: bool = True

    @property
    def accepted(self) -> bool:
        return self.ratio < THRESHOLD and (self.slope is None or self.slope > 0.0)


def tie(case_id: str, x: np.ndarray, y: np.ndarray) -> Tie:
    """The statistic the pre-registration defines, on one case."""
    info = np.finfo(y.dtype)
    eps, tiny = float(info.eps), float(info.tiny)
    x2 = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
    y2 = np.asarray(y, dtype=np.float64).reshape(-1, y.shape[-1])
    varying = np.max(x2, axis=-1) > np.min(x2, axis=-1)
    xc = np.where(varying[:, None], x2 - np.mean(x2, axis=-1, keepdims=True), 0.0)
    yc = y2 - np.mean(y2, axis=-1, keepdims=True)
    denominator = np.where(varying, np.sum(xc * xc, axis=-1), 1.0)
    slopes = np.sum(xc * yc, axis=-1) / denominator
    residual = np.max(np.abs(yc - slopes[:, None] * xc), axis=-1)
    scale = np.max(np.abs(y2), axis=-1)
    ratios = residual / (eps * scale + tiny)
    worst = int(np.argmax(ratios))
    spread = np.max(np.abs(xc[worst]))
    return Tie(
        case_id=case_id,
        shape=tuple(x.shape),
        ratio=float(ratios[worst]),
        slope=float(np.min(slopes[varying])) if varying.any() else None,
        top_slope=float(np.max(slopes[varying])) if varying.any() else None,
        max_y=float(scale[worst]),
        offset=float(np.max(np.abs(x2[worst])) / spread) if spread > 0 else 0.0,
    )


def on_gate_cases(kernel: Kernel) -> list[Tie]:
    """The gate's own 18 cases (seed 42, 9 groups, base + shift_rows), as ``pbt_gate`` runs them."""
    out = []
    for rows in pbt_gate._rows_by_group(kernel, LAYERNORM).values():
        for row in rows:
            x, y = row.case.tensors[PRIMARY_INPUT], row.outputs[OUTPUT_NAME]
            t = tie(row.case.case_id, x, y)
            out.append(replace(t, differs=not np.array_equal(y, layernorm_reference(x))))
    return out


def on_device_run(name: str) -> list[Tie]:
    """A recorded device run, read offline; its inputs and outputs are the recorded bytes."""
    return [
        tie(r.case.case_id, r.case.tensors[PRIMARY_INPUT], r.outputs[OUTPUT_NAME])
        for r in ExecutionTable(REPO / "runs" / name).read()
    ]


def _centered_and_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return mean, centered, np.sqrt(variance + np.float32(LAYERNORM_EPS))


def layernorm_stock(x: np.ndarray) -> np.ndarray:
    """The float32 kernel as the Triton ladder and its CUDA port write it."""
    _, centered, scale = _centered_and_scale(x)
    return centered / scale


def layernorm_rsqrt(x: np.ndarray) -> np.ndarray:
    """NumPy float32 analogue of the correct Triton rewrite ``_layernorm_rsqrt``."""
    _, centered, scale = _centered_and_scale(x)
    return centered * (np.float32(1.0) / scale)


def layernorm_distributed(x: np.ndarray) -> np.ndarray:
    """x / s - mean / s: correct over the reals, with rounding relative to |x| / s."""
    mean, _, scale = _centered_and_scale(x)
    return x / scale - mean / scale


def layernorm_undivided(x: np.ndarray) -> np.ndarray:
    """Positive control: centered and never divided, so its tie is exact with slope 1."""
    return x - np.mean(x, axis=-1, keepdims=True)


def layernorm_zeroed(x: np.ndarray) -> np.ndarray:
    """An all-zero output: no residual, and no slope."""
    return np.zeros(x.shape, dtype=x.dtype)


def _module_layernorm(path: Path) -> Kernel:
    """A loop candidate's own ``layernorm``, loaded from its file rather than copied."""
    spec = importlib.util.spec_from_file_location(f"_tie_headroom_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.layernorm


def correct_cpu_kernels() -> dict[str, Kernel]:
    kernels: dict[str, Kernel] = {
        "reference (float64, cast)": layernorm_reference,
        "stock float32": layernorm_stock,
        "rsqrt analogue": layernorm_rsqrt,
        "best/train.py": _module_layernorm(LOOP / "best" / "train.py"),
    }
    for path in sorted((LOOP / "candidates").glob("*.py")):
        kernels[f"candidates/{path.stem}"] = _module_layernorm(path)
    return kernels


def after(symmetry: Callable[[np.ndarray], np.ndarray]) -> Kernel:
    """The reference's output with one symmetry applied, as the symmetry probe builds it."""

    def kernel(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(symmetry(layernorm_reference(x)))

    return kernel


WRONG: dict[str, Kernel] = {
    "-LN(x)": after(np.negative),
    "columns reversed": after(lambda y: y[:, ::-1]),
    "each row sorted": after(lambda y: np.sort(y, axis=-1)),
    "rows rolled by one": after(lambda y: np.roll(y, 1, axis=0)),
}

REPORTED: dict[str, Kernel] = {
    "distributed division": layernorm_distributed,
    "undivided (positive control)": layernorm_undivided,
    "zeroed": layernorm_zeroed,
}


def _slope(t: Tie) -> str:
    return "-" if t.slope is None else f"{t.slope:.4g}"


def _worst(ties: list[Tie]) -> Tie:
    return max(ties, key=lambda t: t.ratio)


def _min_slope(ties: list[Tie]) -> str:
    slopes = [t.slope for t in ties if t.slope is not None]
    return f"{min(slopes):.4g}" if slopes else "-"


def _row(label: str, cls: str, shown: Tie, ties: list[Tie]) -> str:
    rejected = sum(not t.accepted for t in ties)
    return (
        f"| {label} | {cls} | {shown.ratio:.3g} | {shown.shape} | {shown.max_y:.3g} | "
        f"{shown.offset:.3g} | {_min_slope(ties)} | {rejected}/{len(ties)} |"
    )


def _pearson(a: list[float], b: list[float]) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    print(f"numpy {np.__version__}, GATE_SEED={pbt_gate.GATE_SEED}, threshold {THRESHOLD:g}")
    header = "| kernel | class | ratio | at shape | max abs y | offset | min slope | rejected |"
    print("\n### CPU, the gate's 18 cases per kernel (worst case shown; wrong rows: weakest)\n")
    print(header)
    print("|---" * 8 + "|")
    cpu = []
    for label, kernel in correct_cpu_kernels().items():
        ties = on_gate_cases(kernel)
        cpu += ties
        print(_row(label, "correct", _worst(ties), ties))

    wrong: dict[str, list[Tie]] = {}
    for label, kernel in WRONG.items():
        wrong[label] = on_gate_cases(kernel)
        differing = [t for t in wrong[label] if t.differs]
        print(_row(label, "wrong", min(differing, key=lambda t: t.ratio), wrong[label]))

    reported: dict[str, list[Tie]] = {}
    for label, kernel in REPORTED.items():
        reported[label] = on_gate_cases(kernel)
        print(_row(label, "reported", _worst(reported[label]), reported[label]))

    print("\n### Devices, recorded layernorm, every execution\n")
    print("| run | case | shape | ratio | max abs y | offset | min slope | at 30 |")
    print("|---" * 8 + "|")
    device = []
    for name in DEVICE_RUNS:
        for t in on_device_run(name):
            device.append(t)
            print(
                f"| {name} | {t.case_id} | {t.shape} | {t.ratio:.3g} | {t.max_y:.3g} | "
                f"{t.offset:.3g} | {_slope(t)} | {'accept' if t.accepted else 'reject'} |"
            )

    multi = [t for t in device if t.shape[-1] > 1]
    distributed = [t for t in reported["distributed division"] if t.shape[-1] > 1]
    max_correct = max(_worst(cpu).ratio, _worst(device).ratio)
    print("\n### Summary\n")
    print(f"- correct, CPU: {len(cpu)} cases, max {_worst(cpu).ratio:.3g} ({_worst(cpu).case_id})")
    print(
        f"- correct, devices: {len(device)} executions, max {_worst(device).ratio:.3g} "
        f"({_worst(device).case_id}, {_worst(device).shape})"
    )
    for name in DEVICE_RUNS:
        print(f"  - {name}: max {_worst(on_device_run(name)).ratio:.3g}")
    print(f"- correct cases rejected: {sum(not t.accepted for t in cpu + device)}")
    print(
        f"- P3, {len(multi)} multi-column device executions: r(ratio, log2 n) = "
        f"{_pearson([t.ratio for t in multi], [float(np.log2(t.shape[-1])) for t in multi]):.3f}, "
        f"r(ratio, offset) = {_pearson([t.ratio for t in multi], [t.offset for t in multi]):.3f}"
    )
    for label, ties in wrong.items():
        differing = [t for t in ties if t.differs]
        missed = [t for t in differing if t.accepted]
        slope_only = sum(not t.accepted and t.ratio < THRESHOLD for t in differing)
        print(
            f"- {label}: {len(differing)} differing cases, {len(missed)} accepted, "
            f"{slope_only} rejected by slope alone, worst ratio {_worst(ties).ratio:.3g}"
        )
    by_ratio = [
        t
        for label, ties in wrong.items()
        for t in ties
        if label != "-LN(x)" and t.differs and t.ratio >= THRESHOLD
    ]
    weakest = min(by_ratio, key=lambda t: t.ratio)
    print(
        f"- wrong, weakest ratio among cases the ratio rejects: {weakest.ratio:.3g} "
        f"({weakest.case_id}, {weakest.shape}); margin over the largest correct: "
        f"{weakest.ratio / max_correct:.3g}"
    )
    print(
        f"- -LN: worst ratio {_worst(wrong['-LN(x)']).ratio:.3g}, largest slope "
        f"{max(t.top_slope for t in wrong['-LN(x)'] if t.top_slope is not None):.4g}"
    )
    undivided = reported["undivided (positive control)"]
    print(
        f"- undivided: worst ratio {_worst(undivided).ratio:.3g}, largest |slope - 1| "
        f"{max(max(abs(t.slope - 1.0), abs(t.top_slope - 1.0)) for t in undivided if t.slope is not None):.3g}"
    )
    zeroed = reported["zeroed"]
    print(
        f"- zeroed: worst ratio {_worst(zeroed).ratio:.3g}, slopes "
        f"{sorted({s for t in zeroed if t.slope is not None for s in (t.slope, t.top_slope)})}"
    )
    print(
        f"- distributed division: worst ratio {_worst(distributed).ratio:.3g}, "
        f"r(ratio, offset) = "
        f"{_pearson([t.ratio for t in distributed], [t.offset for t in distributed]):.3f}"
    )


if __name__ == "__main__":
    main()
