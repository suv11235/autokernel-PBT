"""Runs one candidate on the planned inputs and records what it returns. Sandboxed by ``e3b.py``.

This is the only process that imports the candidate, and all it does is record. The gate and the hidden
checker score the recording in another process, which never imports the candidate.

    python -I -B recorder.py WORKDIR BATCHES CALLS

WORKDIR holds ``candidate.py``, ``baseline.py``, ``inputs/<task>/NNN.npy`` in call order and
``timing/<task>.npy``. Everything goes back on stdout as a sequence of ``.npy`` arrays: a JSON header,
then every recorded output in call order, then the call captured from each timing batch. A call that
produced no plain array is recorded as a text array holding the reason, which no accepted output can
be. The stream is a private duplicate of stdout, and stdout itself is pointed at stderr before the
candidate is imported, so a ``print`` in the candidate cannot forge it. Nothing is written to WORKDIR,
which the candidate can write to as well.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import secrets
import statistics
import sys
import time
import warnings
from pathlib import Path

ANSWER = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)

import numpy as np

TASK_KEYS = ("softmax", "layernorm")
#: The kinds the in-process gate accepts (``single_output``). Text is reserved for the reasons.
ACCEPTED_KINDS = "biuf"


class NotPlain(TypeError):
    """An output that is not a plain numeric ``ndarray``: a subclass, a lazy object, a scalar."""


def load_kernels(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {key: getattr(module, key) for key in TASK_KEYS}


def plain(y):
    if type(y) is not np.ndarray or y.dtype.kind not in ACCEPTED_KINDS:
        raise NotPlain(type(y).__name__)
    return y


def _reason(exc: BaseException) -> np.ndarray:
    return np.array(type(exc).__name__)


def _call(kernel, x: np.ndarray) -> np.ndarray:
    """One recorded call. The output is copied, so a buffer the candidate reuses cannot change it."""
    try:
        return plain(kernel(x)).copy()
    except BaseException as exc:  # noqa: BLE001 -- a candidate may raise anything; its type is the record
        return _reason(exc)


def _batch(kernel, x: np.ndarray, calls: int, capture: int) -> tuple[float, np.ndarray]:
    """Seconds for ``calls`` calls, and a copy of call ``capture``'s output, made off the clock."""
    elapsed, kept = 0.0, None
    start = time.perf_counter()
    for i in range(calls):
        y = plain(kernel(x))
        if i == capture:
            elapsed += time.perf_counter() - start
            kept = y.copy()
            start = time.perf_counter()
    return elapsed + time.perf_counter() - start, kept


def measure(new, old, x: np.ndarray, batches: int, calls: int, timed: list) -> float:
    """Speedup over the starting kernel, as in E3: median batch time, alternating the two. One call
    per batch of the candidate's, at an index drawn from the OS, is captured for the checker."""
    times_old, times_new = [], []
    for _ in range(batches):
        times_old.append(_batch(old, x, calls, secrets.randbelow(calls))[0])
        seconds, kept = _batch(new, x, calls, secrets.randbelow(calls))
        times_new.append(seconds)
        timed.append(kept)
    return statistics.median(times_old) / statistics.median(times_new)


def _read_only(path: Path) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    x.setflags(write=False)
    return x


def main() -> None:
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    workdir, batches, calls = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    header: dict = {}
    recorded: dict = {key: [] for key in TASK_KEYS}
    timed: dict = {key: [] for key in TASK_KEYS}
    try:
        new = load_kernels(workdir / "candidate.py")
        old = load_kernels(workdir / "baseline.py")
    except BaseException as exc:  # noqa: BLE001
        header["error"] = type(exc).__name__
    else:
        for key in TASK_KEYS:
            for path in sorted((workdir / "inputs" / key).glob("*.npy")):
                recorded[key].append(_call(new[key], _read_only(path)))
        try:
            header["speedups"] = {
                key: measure(
                    new[key],
                    old[key],
                    _read_only(workdir / "timing" / f"{key}.npy"),
                    batches,
                    calls,
                    timed[key],
                )
                for key in TASK_KEYS
            }
        except BaseException as exc:  # noqa: BLE001
            header["timing_error"] = type(exc).__name__
            timed = {key: [] for key in TASK_KEYS}
    header["counts"] = {key: [len(recorded[key]), len(timed[key])] for key in TASK_KEYS}
    stream = [np.array(json.dumps(header))]
    stream += [y for key in TASK_KEYS for y in recorded[key]]
    stream += [y for key in TASK_KEYS for y in timed[key]]
    for array in stream:
        buffer = io.BytesIO()  # np.save needs a seekable file, and a pipe is not one
        np.save(buffer, array, allow_pickle=False)
        ANSWER.write(buffer.getvalue())
    ANSWER.flush()


if __name__ == "__main__":
    main()
