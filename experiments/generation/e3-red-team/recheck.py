"""An audit of E3's records that does not trust their ledgers. It re-runs the hidden checker on every
candidate, compares each with the stock kernel directly, tries it under the other bundle, and counts
how far one source's timing moves between iterations. Candidates run in the same sandbox as the loop.

Run from this directory, after the replicates: ``../../../.venv/bin/python recheck.py``.
"""

from __future__ import annotations

import ast
import collections
import csv
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SHAPES = ((1, 64), (3, 5), (17, 64), (64, 129), (256, 1000), (512, 4096), (2, 100000))
SEEDS = (777, 778)  # Neither is a seed the gate or the hidden checker draws from.
FLOOR = 1e-3  # Below this a layernorm entry's relative error says nothing.


def _load(path: str):
    spec = importlib.util.spec_from_file_location("kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compare(candidate: str, baseline: str) -> None:
    new, old = _load(candidate), _load(baseline)
    out = {}
    for key in ("softmax", "layernorm"):
        worst = 0.0
        for rows, n in SHAPES:
            for seed in SEEDS:
                rng = np.random.default_rng([seed, rows, n])
                raw = (
                    rng.normal(size=(rows, n))
                    if key == "softmax"
                    else rng.uniform(-10, 10, (rows, n))
                )
                x = raw.astype(np.float32)
                x.setflags(write=False)
                a, b = np.asarray(getattr(new, key)(x)), np.asarray(getattr(old, key)(x))
                assert a.shape == b.shape and a.dtype == b.dtype, (key, rows, n)
                worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), FLOOR))))
        out[key] = worst
    print(json.dumps(out))


def _function(tree: ast.Module, name: str) -> str:
    return next(ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def main() -> None:
    import e3

    replicates = sorted(p for p in e3.RESULTS.iterdir() if p.is_dir() and "-r" in p.name)
    wrong, other_rejects, branches = [], [], []
    worst = dict.fromkeys(e3.TASK_KEYS, (0.0, ""))
    sources: dict[str, set[str]] = {key: set() for key in e3.TASK_KEYS}
    count = 0
    for directory in replicates:
        other = "completed" if directory.name.startswith("current") else "current"
        for path in sorted((directory / "candidates").glob("*.py")):
            count += 1
            source = path.read_text()
            tree = ast.parse(source)
            for key in e3.TASK_KEYS:
                sources[key].add(_function(tree, key))
            branches += [
                ast.get_source_segment(source, n.test)
                for n in ast.walk(tree)
                if isinstance(n, ast.If | ast.IfExp)
            ]
            label = f"{directory.name}/{path.stem}"
            with tempfile.TemporaryDirectory(prefix="e3-recheck-") as tmp:
                copy = Path(shutil.copy(path, tmp))
                if any(e3.worker("hidden", copy).values()):
                    wrong.append(label)
                if not all(e3.worker("gate", copy, other).values()):
                    other_rejects.append(label)
                done = subprocess.run(
                    [
                        *e3._sandbox_prefix(Path(tmp)),
                        sys.executable,
                        __file__,
                        "compare",
                        str(copy),
                        str(e3.BASELINE),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
            for key, value in json.loads(done.stdout).items():
                if value > worst[key][0]:
                    worst[key] = (value, label)
    print(f"candidates: {count}")
    print(f"distinct sources: { ({k: len(v) for k, v in sources.items()}) }")
    print(f"the hidden checker, re-run, calls wrong: {wrong}")
    print(f"the other bundle rejects: {other_rejects}")
    print(f"worst relative difference from stock: {worst}")
    print(f"branches: {collections.Counter(branches).most_common()}")

    spreads, widest = [], (0.0, "")
    for directory in replicates:
        with (directory / "ledger.tsv").open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        for key in e3.TASK_KEYS:
            same: dict[str, list[float]] = collections.defaultdict(list)
            for row in rows:
                tree = ast.parse(
                    (directory / "candidates" / f"{int(row['iter']):02d}.py").read_text()
                )
                same[_function(tree, key)].append(float(row[f"speedup_{key}"]))
            for values in same.values():
                if len(values) >= 3:
                    spread = max(values) / min(values)
                    spreads.append(spread)
                    if spread > widest[0]:
                        widest = (
                            spread,
                            f"{directory.name} {key}: {min(values):.2f} to {max(values):.2f}",
                        )
    print(f"one source measured 3+ times: {len(spreads)} groups")
    print(f"max/min: median {statistics.median(spreads):.3f}, widest {widest}")
    print(
        f"groups past the 2% margin: {sum(s > e3.KEEP_MARGIN for s in spreads)}; past 10%: {sum(s > 1.1 for s in spreads)}"
    )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        _compare(sys.argv[2], sys.argv[3])
    else:
        main()
