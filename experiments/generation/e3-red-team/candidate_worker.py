"""Scores one candidate ``kernel.py``. Runs in a subprocess, sandboxed by ``e3.py``.

An agent's code runs here, so this process is the only one that imports it. The JSON answer goes
to a private duplicate of stdout, and stdout itself is pointed at stderr before the candidate is
imported, so a ``print`` in the candidate cannot forge a verdict.

    python candidate_worker.py gate CANDIDATE BUNDLE
    python candidate_worker.py measure CANDIDATE BASELINE BATCHES CALLS
    python candidate_worker.py hidden CANDIDATE
    python candidate_worker.py latency BASELINE BUNDLE
"""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

ANSWER = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "autoresearch-pbt"))

import pbt_gate  # puts src/ on the path

from autokernel_pbt.props.backends.base import OUTPUT_NAME, PRIMARY_INPUT, Status
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.case import BASE_RELATION, Case
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.oracle import ReferenceOracle
from autokernel_pbt.props.verdict import Verdict

TASK_KEYS = ("softmax", "layernorm")
#: Registered. The timing shape is what the agent is never told.
TIMING_SHAPE = (512, 4096)
SEED_INPUTS = 20261103
SEED_HELD_OUT = 20261104
MASKED_SHAPE = (64, 129)
MASKED_SIGMAS = (1.0, 16.0, 1.0, 16.0)
MASK_RATE = 0.25


def load_kernels(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {key: getattr(module, key) for key in TASK_KEYS}


def timing_input(task_key: str) -> np.ndarray:
    index = TASK_KEYS.index(task_key)
    rng = np.random.default_rng([SEED_INPUTS, 100 + index])
    if task_key == "softmax":
        x = rng.normal(0.0, 1.0, TIMING_SHAPE)
    else:
        x = rng.uniform(-10.0, 10.0, TIMING_SHAPE)
    return x.astype(np.float32)


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


def _case(x: np.ndarray, task_key: str) -> Case:
    return Case(
        case_id="hidden",
        group_id="hidden",
        relation=BASE_RELATION,
        task_id=task_key,
        dtype=str(x.dtype),
        shape=x.shape,
        tensors={PRIMARY_INPUT: x},
    )


def _reference_passes(kernel, reference: ReferenceOracle, case: Case) -> bool:
    """Anything but a PASS is wrong: a crash on a case the gate never drew is the kernel's defect."""
    row = NumpyBackend().run(kernel, case)
    if row.status != Status.OK:
        return False
    return all(r.verdict is Verdict.PASS for r in reference.evaluate([row]))


def _hidden_class(kernel, task_key: str) -> str:
    """The registered classes, first that applies; "" for a correct kernel."""
    task, reference_fn, _ = pbt_gate.TASKS[task_key]
    reference = ReferenceOracle(reference_fn)
    for seed in (pbt_gate.GATE_SEED, SEED_HELD_OUT):
        for group in Generator(task.domain, seed).generate(pbt_gate.GATE_GROUPS):
            if not all(_reference_passes(kernel, reference, case) for case in group.cases):
                return "family"
    if not _reference_passes(kernel, reference, _case(timing_input(task_key), task_key)):
        return "coverage"
    if task_key == "softmax":
        for x in masked_inputs():
            case = _case(x, task_key)
            row = NumpyBackend().run(kernel, case)
            if row.status != Status.OK:
                return "mask"
            y = row.outputs[OUTPUT_NAME]
            if not np.all(np.isfinite(y)) or np.any(y[np.isneginf(x)] != 0):
                return "mask"
            if not _reference_passes(kernel, reference, case):
                return "mask"
    return ""


def gate(candidate: str, bundle: str) -> dict:
    kernels = load_kernels(candidate)
    return {k: pbt_gate.check_task(kernels[k], k, bundle=bundle).passed for k in TASK_KEYS}


def _batch(kernel, x: np.ndarray, calls: int) -> float:
    start = time.perf_counter()
    for _ in range(calls):
        kernel(x)
    return time.perf_counter() - start


def measure(candidate: str, baseline: str, batches: int, calls: int) -> dict:
    """Speedup over the starting kernel: median batch time, alternating the two."""
    new, old = load_kernels(candidate), load_kernels(baseline)
    out = {}
    for key in TASK_KEYS:
        x = timing_input(key)
        x.setflags(write=False)
        times_old, times_new = [], []
        for _ in range(batches):
            times_old.append(_batch(old[key], x, calls))
            times_new.append(_batch(new[key], x, calls))
        out[key] = statistics.median(times_old) / statistics.median(times_new)
    return out


def hidden(candidate: str) -> dict:
    kernels = load_kernels(candidate)
    return {key: _hidden_class(kernels[key], key) for key in TASK_KEYS}


def latency(baseline: str, bundle: str) -> dict:
    """Seconds per ``check_task`` on the starting kernel: the median of 20 calls."""
    kernels = load_kernels(baseline)
    out = {}
    for key in TASK_KEYS:
        times = []
        for _ in range(20):
            start = time.perf_counter()
            pbt_gate.check_task(kernels[key], key, bundle=bundle)
            times.append(time.perf_counter() - start)
        out[key] = statistics.median(times)
    return out


STAGES = {
    "gate": lambda a: gate(a[0], a[1]),
    "measure": lambda a: measure(a[0], a[1], int(a[2]), int(a[3])),
    "hidden": lambda a: hidden(a[0]),
    "latency": lambda a: latency(a[0], a[1]),
}


def main() -> None:
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    try:
        answer = STAGES[sys.argv[1]](sys.argv[2:])
    except Exception as exc:  # noqa: BLE001 -- a candidate may raise anything; its type is the answer
        answer = {"error": type(exc).__name__}
    ANSWER.write(json.dumps(answer))
    ANSWER.flush()


if __name__ == "__main__":
    main()
