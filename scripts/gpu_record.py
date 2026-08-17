#!/usr/bin/env python3
"""Record one task's executions on the GPU. Scoring happens later, on CPU.

Deliberately does NOT score. The driver's scoring pass builds a declarative arm from
the task's contract and evaluates four oracles, none of which needs a device -- doing
it here would spend rented time on work that is free at home.

Usage:
    python3 scripts/gpu_record.py --task softmax --out runs/gpu-softmax
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# `kernels/` is a namespace package at the repo root, not under `src/`. pytest is
# configured with `pythonpath = ["src", "."]`, but that is a PYTEST setting -- it does
# nothing for a standalone script, which is how this runs on the instance. Without
# this the device tests pass and the recording script dies with ModuleNotFoundError,
# which is exactly what the first hardware run hit.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from autokernel_pbt.props.backends.triton_backend import TritonBackend
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import TASKS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-groups", type=int, default=None)
    args = parser.parse_args()

    from kernels.triton.ladder import KERNELS

    task = TASKS[args.task]
    # Default to the whole domain: fewer groups than shapes makes Generator warn that
    # a boundary shape will never be exercised, and boundary coverage is the corpus's
    # main recall mechanism.
    n_groups = args.n_groups or len(task.domain.shapes)

    backend = TritonBackend()

    results = []
    for group in Generator(task.domain, seed=args.seed).generate(n_groups):
        for case in group.cases:
            # One kernel per shape: BLOCK is derived from the row width, so a new
            # shape means a new compiled artifact -- which is precisely why the
            # compiled telemetry now varies instead of being constant.
            kernel = KERNELS[args.task](case.shape[-1])
            result = backend.run(kernel, case)
            result.kernel_id = kernel.kernel_id
            # Stated correct, not "not stated": these are the reference ports, and
            # collapsing the two would enlarge the correct-kernel denominator of the
            # false-positive rate.
            result.kernel_is_broken = False
            result.case_spec = group.spec
            results.append(result)

    ExecutionTable(args.out).write(results)

    statuses: dict[str, int] = {}
    for result in results:
        statuses[str(result.status)] = statuses.get(str(result.status), 0) + 1
    print(json.dumps({"task": args.task, "rows": len(results), "status": statuses}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
