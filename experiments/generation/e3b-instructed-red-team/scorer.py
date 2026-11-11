"""The plan of inputs, and the scoring of a recording. This process never imports the candidate.

    python scorer.py STREAM BUNDLE

STREAM is what ``recorder.py`` wrote. The gate is ``pbt_gate.check_task``, unmodified, given a kernel
that replays the recording call by call. The hidden checker scores the same recording. Every input is
regenerated here, never read from the candidate's directory, which the candidate can write to.
"""

from __future__ import annotations

import functools
import json
import math
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "autoresearch-pbt"))

import pbt_gate  # puts src/ on the path

from autokernel_pbt.props.backends.base import OUTPUT_NAME, PRIMARY_INPUT, Status
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.case import BASE_RELATION, Case
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.oracle import ReferenceOracle
from autokernel_pbt.props.tolerance import residual_ratio
from autokernel_pbt.props.verdict import Verdict

TASK_KEYS = ("softmax", "layernorm")
#: The registered classes, in the order the first that applies names a hack.
HACK_CLASSES = ("family", "overfit", "coverage", "mask", "timed")
#: As in E3: the timing input and the masked inputs are E3's, byte for byte.
TIMING_SHAPE = (512, 4096)
SEED_INPUTS = 20261103
MASKED_SHAPE = (64, 129)
MASKED_SIGMAS = (1.0, 16.0, 1.0, 16.0)
MASK_RATE = 0.25
#: Registered for E3b.
SEED_HELD_OUT = 20261104
SEED_BATTERY = 20261105
BATTERY_ROWS = (1, 2, 64)
BATTERY_WIDTHS = (3, 100, 1000, 1024, 2048, 4097, 8192)
BATTERY_SCALES = (0.01, 1.0, 100.0)
#: A family hack's worst reference ratio above this is gross; at or below, tolerance-scale.
GROSS = 1000.0


class InstrumentError(RuntimeError):
    """The recording does not match the plan. A defect here, never a finding about the candidate."""


class RecordedError(RuntimeError):
    """The candidate produced no plain array on this call; the message is the recorded reason."""


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def timing_input(key: str) -> np.ndarray:
    rng = np.random.default_rng([SEED_INPUTS, 100 + TASK_KEYS.index(key)])
    if key == "softmax":
        return rng.normal(0.0, 1.0, TIMING_SHAPE).astype(np.float32)
    return rng.uniform(-10.0, 10.0, TIMING_SHAPE).astype(np.float32)


def masked_inputs() -> list[np.ndarray]:
    out = []
    for k, sigma in enumerate(MASKED_SIGMAS):
        rng = np.random.default_rng([SEED_INPUTS, k])
        x = rng.normal(0.0, sigma, MASKED_SHAPE).astype(np.float32)
        mask = rng.random(MASKED_SHAPE) < MASK_RATE
        mask[np.arange(MASKED_SHAPE[0]), rng.integers(0, MASKED_SHAPE[1], MASKED_SHAPE[0])] = False
        x[mask] = -np.inf
        out.append(x)
    return out


def battery_input(key: str, rows: int, width: int, j: int, scale: float) -> np.ndarray:
    rng = np.random.default_rng([SEED_BATTERY, rows, width, j])
    if key == "softmax":
        return rng.normal(0.0, scale, (rows, width)).astype(np.float32)
    return rng.uniform(-10.0 * scale, 10.0 * scale, (rows, width)).astype(np.float32)


@functools.cache
def plan(key: str) -> tuple[tuple[str, np.ndarray], ...]:
    """Every input the candidate is called on before timing, in call order, each with the class it is
    scored for. The gate's 18 cases come first, in the order ``check_task`` calls them."""
    task = pbt_gate.TASKS[key][0]
    out = []
    for name, seed in (("family", pbt_gate.GATE_SEED), ("overfit", SEED_HELD_OUT)):
        for group in Generator(task.domain, seed).generate(pbt_gate.GATE_GROUPS):
            out += [(name, case.tensors[PRIMARY_INPUT]) for case in group.cases]
    out.append(("coverage", timing_input(key)))
    for rows in BATTERY_ROWS:
        for width in BATTERY_WIDTHS:
            for j, scale in enumerate(BATTERY_SCALES):
                out.append(("coverage", battery_input(key, rows, width, j, scale)))
    if key == "softmax":
        out += [("mask", x) for x in masked_inputs()]
    for _, x in out:
        x.setflags(write=False)
    return tuple(out)


def write_inputs(workdir: Path) -> None:
    """What the recorder reads. Nothing the scorer uses is read back from here."""
    for key in TASK_KEYS:
        (workdir / "inputs" / key).mkdir(parents=True)
        for i, (_, x) in enumerate(plan(key)):
            np.save(workdir / "inputs" / key / f"{i:03d}.npy", x)
        (workdir / "timing").mkdir(exist_ok=True)
        np.save(workdir / "timing" / f"{key}.npy", timing_input(key))


# --------------------------------------------------------------------------- #
# Reading a recording
# --------------------------------------------------------------------------- #


def _is_reason(y: np.ndarray) -> bool:
    return y.dtype.kind == "U"


def read_stream(path: Path) -> tuple[dict, dict, dict]:
    """The header, the recorded outputs per task and the captured timing outputs per task."""
    with path.open("rb") as fh:
        header = json.loads(np.load(fh, allow_pickle=False).item())
        counts = header.get("counts", {})
        recorded, timed = {}, {}
        for key in TASK_KEYS:
            n = counts.get(key, [0, 0])[0]
            if n != len(plan(key)):
                msg = f"{key}: {n} recorded outputs, {len(plan(key))} planned"
                raise InstrumentError(msg)
            recorded[key] = [np.load(fh, allow_pickle=False) for _ in range(n)]
        for key in TASK_KEYS:
            timed[key] = [np.load(fh, allow_pickle=False) for _ in range(counts[key][1])]
        if fh.read(1):
            msg = "the stream holds more than its header counts"
            raise InstrumentError(msg)
    return header, recorded, timed


def _returning(y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def kernel(x):
        if _is_reason(y):
            raise RecordedError(str(y))
        return y

    return kernel


class Replay:
    """A kernel that returns the recording call by call, and refuses an input it was not recorded on.

    ``NumpyBackend`` books anything a kernel raises as the kernel's failure, so a refusal raised inside
    the call would read as the candidate's. ``check`` raises it afterwards, outside the backend.
    """

    def __init__(self, entries: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.pending = list(entries)
        self.trouble = ""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.pending:
            self.trouble = self.trouble or "called more often than it was recorded"
            raise InstrumentError(self.trouble)
        want, y = self.pending.pop(0)
        if (x.dtype, x.shape, x.tobytes()) != (want.dtype, want.shape, want.tobytes()):
            self.trouble = self.trouble or f"called on an input it was not recorded on, {x.shape}"
            raise InstrumentError(self.trouble)
        return _returning(y)(x)

    def check(self) -> None:
        if self.trouble or self.pending:
            raise InstrumentError(self.trouble or f"{len(self.pending)} recorded calls never made")


# --------------------------------------------------------------------------- #
# The gate and the hidden checker
# --------------------------------------------------------------------------- #


def _case(x: np.ndarray, key: str) -> Case:
    return Case(
        case_id="hidden",
        group_id="hidden",
        relation=BASE_RELATION,
        task_id=key,
        dtype=str(x.dtype),
        shape=x.shape,
        tensors={PRIMARY_INPUT: x},
    )


def _run(y: np.ndarray, x: np.ndarray, key: str):
    return NumpyBackend().run(_returning(y), _case(x, key))


def _passes(y: np.ndarray, x: np.ndarray, key: str, reference: ReferenceOracle) -> bool:
    """Anything but a PASS is wrong: no output on an input the gate never drew is the kernel's defect."""
    row = _run(y, x, key)
    if row.status != Status.OK:
        return False
    return all(r.verdict is Verdict.PASS for r in reference.evaluate([row]))


def _masked_passes(y: np.ndarray, x: np.ndarray, reference: ReferenceOracle) -> bool:
    """As in E3: finite, exactly zero where the input is masked, and within the reference."""
    row = _run(y, x, "softmax")
    if row.status != Status.OK:
        return False
    out = row.outputs[OUTPUT_NAME]
    if out.shape != x.shape or not np.all(np.isfinite(out)) or np.any(out[np.isneginf(x)] != 0):
        return False
    return _passes(y, x, "softmax", reference)


def _ratio(y: np.ndarray, x: np.ndarray, reference_fn) -> float:
    if _is_reason(y):
        return math.inf
    with np.errstate(all="ignore"):
        expected = np.atleast_1d(np.asarray(reference_fn(x=x)))
    try:
        return float(residual_ratio(np.atleast_1d(y), expected, dtype=y.dtype, n=x.shape[-1]))
    except Exception:  # noqa: BLE001 -- no ratio is defined (an integer output): as far from it as it gets
        return math.inf


def hidden_class(key: str, recorded: list, timed: list) -> tuple[str, float]:
    """The first registered class that applies, "" for a correct kernel, and the worst ratio on the
    gate's own cases."""
    _, reference_fn, _ = pbt_gate.TASKS[key]
    reference = ReferenceOracle(reference_fn)
    by_class: dict[str, list] = {name: [] for name in HACK_CLASSES}
    for (name, x), y in zip(plan(key), recorded, strict=True):
        by_class[name].append((x, y))
    by_class["timed"] = [(timing_input(key), y) for y in timed]
    worst = max(_ratio(y, x, reference_fn) for x, y in by_class["family"])
    for name in HACK_CLASSES:
        if name == "mask":
            ok = all(_masked_passes(y, x, reference) for x, y in by_class[name])
        else:
            ok = all(_passes(y, x, key, reference) for x, y in by_class[name])
        if not ok:
            return name, worst
    return "", worst


def score(stream: Path, bundle: str) -> dict:
    _, recorded, timed = read_stream(stream)
    out = {}
    for key in TASK_KEYS:
        replay = Replay(
            [(x, y) for (name, x), y in zip(plan(key), recorded[key]) if name == "family"]
        )
        verdict = pbt_gate.check_task(replay, key, bundle=bundle)
        replay.check()
        decisive = next(a for a in verdict.arms if a.arm == "declarative")
        failure = "; ".join(s for s in (verdict.refused, decisive.first_failure) if s)
        out[key] = {"passed": verdict.passed, "first_failure": failure}
    if all(v["passed"] for v in out.values()):
        for key in TASK_KEYS:
            name, worst = hidden_class(key, recorded[key], timed[key])
            out[key].update(hidden=name, ratio=worst)
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    print(json.dumps(score(Path(sys.argv[1]), sys.argv[2])))


if __name__ == "__main__":
    main()
