#!/usr/bin/env python3
"""The outer loop. Not modified by agents.

One iteration = gate, then measure, then keep-or-revert. The order is the design:
the property gate runs FIRST and knows nothing about speed, so a candidate that is
faster because it is wrong is rejected before its ``val_bpb`` is ever computed --
and is recorded as rejected, which is the datum this experiment exists to collect.

Usage:
    runner.py baseline          establish the ledger's first row from best/train.py
    runner.py evaluate "note"   gate + measure the working train.py, keep or revert
    runner.py rigor N           re-measure best/train.py under N seeds
    runner.py status            print the ledger
"""

from __future__ import annotations

import json
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
LEDGER = HERE / "results.tsv"
BEST = HERE / "best" / "train.py"
WORK = HERE / "train.py"
COLUMNS = (
    "iter", "decision", "gate", "val_bpb", "tok_per_s", "steps", "params",
    "gate_detail", "note",
)


def rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    lines = LEDGER.read_text().strip().splitlines()
    if not lines:
        return []
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def _cell(value: object) -> str:
    """One TSV cell. A tab or newline inside a free-text field (an agent's note, a
    gate's failure detail) would otherwise split the row and misalign every later
    column when ``rows()`` reads it back."""
    return " ".join(str(value).split())


def append(row: dict) -> None:
    new = not LEDGER.exists()
    with LEDGER.open("a") as fh:
        if new:
            fh.write("\t".join(COLUMNS) + "\n")
        fh.write("\t".join(_cell(row.get(c, "")) for c in COLUMNS) + "\n")


def best_bpb() -> float:
    kept = [r for r in rows() if r["decision"] in ("baseline", "keep")]
    return float(kept[-1]["val_bpb"]) if kept else float("inf")


def run_gate() -> dict:
    out = subprocess.run(
        [str(PY), "pbt_gate.py"], cwd=HERE, capture_output=True, text=True, timeout=600,
        check=False,
    )
    if out.returncode != 0:
        return {"passed": False, "error": (out.stderr or out.stdout)[-1500:]}
    return json.loads(out.stdout)


def run_train(env_extra: dict | None = None) -> dict:
    import os

    env = {**os.environ, **(env_extra or {})}
    out = subprocess.run(
        [str(PY), "train.py"], cwd=HERE, capture_output=True, text=True,
        timeout=900, env=env, check=False,
    )
    if out.returncode != 0:
        return {"error": (out.stderr or out.stdout)[-1500:]}
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": f"unparseable stdout: {out.stdout[-600:]}"}


def gate_detail(g: dict) -> str:
    if "error" in g:
        return f"gate crashed: {g['error'][:200].replace(chr(10), ' ')}"
    bits = []
    for t in g["tasks"]:
        for a in t["arms"]:
            bits.append(f"{t['task']}/{a['arm']}={a['groups_failed']}/{a['groups_total']}")
    return " ".join(bits)


def evaluate(note: str) -> int:
    n = len(rows())
    g = run_gate()
    detail = gate_detail(g)
    if not g.get("passed"):
        first = ""
        for t in g.get("tasks", []):
            for a in t["arms"]:
                if a["arm"] == "declarative" and a["first_failure"]:
                    first = a["first_failure"]
        append({"iter": n, "decision": "reject-gate", "gate": "FAIL", "val_bpb": "",
                "tok_per_s": "", "steps": "", "params": "",
                "gate_detail": detail, "note": f"{note} || {first}"[:400]})
        shutil.copy(BEST, WORK)
        print(json.dumps({"decision": "reject-gate", "gate": detail, "first_failure": first,
                          "reverted": True}, indent=2))
        return 1

    m = run_train()
    if "error" in m:
        append({"iter": n, "decision": "reject-crash", "gate": "PASS", "val_bpb": "",
                "tok_per_s": "", "steps": "", "params": "", "gate_detail": detail,
                "note": f"{note} || {m['error'][:200]}"})
        shutil.copy(BEST, WORK)
        print(json.dumps({"decision": "reject-crash", "error": m["error"][-800:],
                          "reverted": True}, indent=2))
        return 1

    prev = best_bpb()
    keep = m["val_bpb"] < prev
    append({"iter": n, "decision": "keep" if keep else "revert", "gate": "PASS",
            "val_bpb": m["val_bpb"], "tok_per_s": m["tokens_per_sec"], "steps": m["steps"],
            "params": m["params"], "gate_detail": detail,
            "note": note[:300]})
    if keep:
        shutil.copy(WORK, BEST)
    else:
        shutil.copy(BEST, WORK)
    print(json.dumps({"decision": "keep" if keep else "revert", "val_bpb": m["val_bpb"],
                      "previous_best": prev, "tok_per_s": m["tokens_per_sec"],
                      "steps": m["steps"], "gate": detail}, indent=2))
    return 0


def measure(path: str, note: str) -> int:
    """Gate + measure a candidate WITHOUT keep/revert.

    Round-level attribution needs every candidate scored against the same starting
    point. Greedy keep/revert chains them instead, which is right for the loop and
    wrong for "what was this one change worth".
    """
    shutil.copy(HERE / path, WORK)
    n = len(rows())
    g = run_gate()
    detail = gate_detail(g)
    if not g.get("passed"):
        first = ""
        for t in g.get("tasks", []):
            for a in t["arms"]:
                if a["arm"] == "declarative" and a["first_failure"]:
                    first = a["first_failure"]
        append({"iter": n, "decision": "measure-reject-gate", "gate": "FAIL",
                "gate_detail": detail, "note": f"{path} || {note} || {first}"[:400]})
        print(json.dumps({"candidate": path, "decision": "reject-gate",
                          "first_failure": first}, indent=2))
        shutil.copy(BEST, WORK)
        return 1
    m = run_train()
    if "error" in m:
        append({"iter": n, "decision": "measure-crash", "gate": "PASS",
                "gate_detail": detail,
                "note": f"{path} || {m['error'][:200]}"})
        print(json.dumps({"candidate": path, "decision": "crash",
                          "error": m["error"][-800:]}, indent=2))
        shutil.copy(BEST, WORK)
        return 1
    append({"iter": n, "decision": "measure", "gate": "PASS", "val_bpb": m["val_bpb"],
            "tok_per_s": m["tokens_per_sec"], "steps": m["steps"], "params": m["params"],
            "gate_detail": detail, "note": f"{path} || {note}"[:300]})
    print(json.dumps({"candidate": path, **m, "gate": detail}, indent=2))
    shutil.copy(BEST, WORK)
    return 0


def baseline() -> int:
    shutil.copy(BEST, WORK)
    g = run_gate()
    m = run_train()
    if "error" in m:
        # No row: a baseline without a val_bpb would make best_bpb() raise on the
        # next evaluate, and there is nothing to compare against anyway.
        print(json.dumps({"decision": "baseline-crash", "error": m["error"][-800:]}, indent=2))
        return 1
    append({"iter": 0, "decision": "baseline", "gate": "PASS" if g.get("passed") else "FAIL",
            "val_bpb": m["val_bpb"], "tok_per_s": m["tokens_per_sec"], "steps": m["steps"],
            "params": m["params"], "gate_detail": gate_detail(g), "note": "plain NumPy GPT baseline"})
    print(json.dumps(m, indent=2))
    return 0


def rigor(n: int) -> int:
    shutil.copy(BEST, WORK)
    out = []
    for s in range(n):
        m = run_train({"AUTORESEARCH_SEED": str(s)})
        out.append(m)
        print(f"seed {s}: {json.dumps(m)}", flush=True)
    vals = [m["val_bpb"] for m in out if "val_bpb" in m]
    if vals:
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        print(json.dumps({"seeds": len(vals), "mean_val_bpb": round(mean, 6),
                          "sd": round(sd, 6), "values": vals}, indent=2))
    return 0


def status() -> int:
    for r in rows():
        print("\t".join(f"{r.get(c, '')}" for c in
                        ("iter", "decision", "val_bpb", "tok_per_s", "steps", "note")))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "baseline":
        sys.exit(baseline())
    if cmd == "measure":
        sys.exit(measure(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))
    if cmd == "evaluate":
        sys.exit(evaluate(sys.argv[2] if len(sys.argv) > 2 else ""))
    if cmd == "rigor":
        sys.exit(rigor(int(sys.argv[2]) if len(sys.argv) > 2 else 3))
    sys.exit(status())
