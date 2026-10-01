"""Multi-seed re-measurement. The single-run ledger cannot separate the top
candidates -- they sit inside the run-to-run spread -- so the winner is decided
here or not at all.

Borrowed directly from the autoresearch-mlx fork, which hit the same wall: a
single val_bpb at a wall-clock budget carries noise from the step count, and a
keep/revert loop run on one seed will happily chase it.

Usage:
    rigor_sweep.py CAND[,CAND...] SEED[,SEED...] [OUT]

``OUT`` defaults to ``rigor.json`` and is never overwritten: the recorded sweep is
what the measurement doc quotes, so a re-run must name a new file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: The loop's own venv (see requirements.txt), falling back to the project's. The
#: fallback is loud: under the project's numpy<2 pin the same code takes ~6x fewer
#: steps in the budget, so its val_bpb is not comparable with the recorded ledger.
_LOCAL_PY = HERE / ".venv" / "bin" / "python"
PY = _LOCAL_PY if _LOCAL_PY.exists() else HERE.parents[1] / ".venv" / "bin" / "python"
if PY != _LOCAL_PY:
    print(f"warning: {_LOCAL_PY} not found; using the project venv, whose numpy<2 "
          f"pin makes timings incomparable with results.tsv", file=sys.stderr)
BEST = HERE / "best" / "train.py"
WORK = HERE / "train.py"


def run(cand: str, seed: int) -> dict:
    shutil.copy(HERE / cand, WORK)
    env = {**os.environ, "AUTORESEARCH_SEED": str(seed)}
    out = subprocess.run([str(PY), "train.py"], cwd=HERE, capture_output=True,
                         text=True, timeout=900, env=env, check=False)
    if out.returncode != 0:
        return {"error": (out.stderr or out.stdout)[-400:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    cands = sys.argv[1].split(",")
    seeds = [int(s) for s in sys.argv[2].split(",")]
    dest = HERE / (sys.argv[3] if len(sys.argv) > 3 else "rigor.json")
    # Checked before the first run, not after: a sweep is minutes of compute per seed.
    if dest.exists():
        sys.exit(f"{dest.name} exists; pass a new output name as the third argument")
    table = {}
    try:
        for c in cands:
            vals, rows = [], []
            for s in seeds:
                m = run(c, s)
                rows.append(m)
                if "val_bpb" in m:
                    vals.append(m["val_bpb"])
                print(f"{Path(c).stem:22s} seed={s} {json.dumps(m)}", flush=True)
            if vals:
                mean = sum(vals) / len(vals)
                sd = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
                table[Path(c).stem] = {
                    "mean_val_bpb": round(mean, 5), "sd": round(sd, 5),
                    "min": round(min(vals), 5), "max": round(max(vals), 5),
                    "n": len(vals),
                    "mean_tok_per_s": round(
                        sum(r["tokens_per_sec"] for r in rows if "tokens_per_sec" in r)
                        / len(vals), 1),
                    "mean_steps": round(
                        sum(r["steps"] for r in rows if "steps" in r) / len(vals), 1),
                    "values": vals,
                }
    finally:
        # Each run overwrites the working train.py; leave it as the runner expects,
        # or the next ``runner.py evaluate`` gates the last candidate swept.
        shutil.copy(BEST, WORK)
    print("\n" + json.dumps(table, indent=2))
    dest.write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
