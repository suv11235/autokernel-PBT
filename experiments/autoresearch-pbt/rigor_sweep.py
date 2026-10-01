"""Multi-seed re-measurement. The single-run ledger cannot separate the top
candidates -- they sit inside the run-to-run spread -- so the winner is decided
here or not at all.

Borrowed directly from the autoresearch-mlx fork, which hit the same wall: a
single val_bpb at a wall-clock budget carries noise from the step count, and a
keep/revert loop run on one seed will happily chase it.
"""
from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parents[1] / ".venv" / "bin" / "python"

def run(cand: str, seed: int) -> dict:
    shutil.copy(HERE / cand, HERE / "train.py")
    import os
    env = {**os.environ, "AUTORESEARCH_SEED": str(seed)}
    out = subprocess.run([str(PY), "train.py"], cwd=HERE, capture_output=True,
                         text=True, timeout=900, env=env)
    if out.returncode != 0:
        return {"error": (out.stderr or out.stdout)[-400:]}
    return json.loads(out.stdout.strip().splitlines()[-1])

def main() -> None:
    cands = sys.argv[1].split(",")
    seeds = [int(s) for s in sys.argv[2].split(",")]
    table = {}
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
                "mean_tok_per_s": round(sum(r["tokens_per_sec"] for r in rows if "tokens_per_sec" in r) / len(vals), 1),
                "mean_steps": round(sum(r["steps"] for r in rows if "steps" in r) / len(vals), 1),
                "values": vals,
            }
    print("\n" + json.dumps(table, indent=2))
    (HERE / "rigor.json").write_text(json.dumps(table, indent=2))

main()
