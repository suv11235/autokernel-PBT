#!/usr/bin/env python3
"""Record one task's executions on the GPU. Scoring happens later, on CPU.

Deliberately does NOT score. The driver's scoring pass builds a declarative arm from
the task's contract and evaluates four oracles, none of which needs a device -- doing
it here would spend rented time on work that is free at home.

Usage:
    python3 scripts/gpu_record.py --task softmax --out runs/gpu-softmax

If anything raises mid-run, every case group already recorded in full is written to
`<out>.partial` before the error propagates, and the exit status is non-zero. `--out`
itself is never written by a failed run, so an earlier complete table there survives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# `kernels/` lives at the repo root and the package at `src/`. pytest is configured with
# `pythonpath = ["src", "."]`, but that is a PYTEST setting -- it does nothing for a
# standalone script, which is how this runs on the instance. Without this the device
# tests pass and the recording script dies with ModuleNotFoundError, which is exactly
# what the first hardware run hit. `src` is added too, so the script does not depend on
# `pip install -e` having run first.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from autokernel_pbt.props.backends.base import ExecutionResult
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

    results: list[ExecutionResult] = []
    try:
        for group in Generator(task.domain, seed=args.seed).generate(n_groups):
            rows = []
            for case in group.cases:
                # One kernel per shape: BLOCK is derived from the row width, so a new
                # shape means a new compiled artifact -- which is precisely why the
                # compiled telemetry now varies instead of being constant.
                kernel = KERNELS[args.task](case.shape[-1])
                result = backend.run(kernel, case)
                result.kernel_id = kernel.kernel_id
                # Stated correct, not "not stated": these are the reference ports, and
                # collapsing the two would enlarge the correct-kernel denominator of
                # the false-positive rate.
                result.kernel_is_broken = False
                result.case_spec = group.spec
                rows.append(result)
            # Whole groups only: the case group is the unit the arms are compared on,
            # and a group missing some of its relations is not that unit.
            results.extend(rows)
    except BaseException:
        # Including KeyboardInterrupt: an interrupted run has still been paid for.
        _write_partial(args.out, results)
        raise

    ExecutionTable(args.out).write(results)

    statuses: dict[str, int] = {}
    for result in results:
        statuses[str(result.status)] = statuses.get(str(result.status), 0) + 1
    print(json.dumps({"task": args.task, "rows": len(results), "status": statuses}, indent=2))
    print(f"wrote {args.out}")
    return 0


def _write_partial(out: Path, results: list[ExecutionResult]) -> None:
    """Persist the completed groups of a failed run beside `out`, never at it.

    Never at `out`: a re-run that dies must not replace an earlier complete table with
    an incomplete one, and nothing inside a table records that it is incomplete -- so
    the path has to say it.
    """
    if not results:
        print("recording FAILED before any case group completed; nothing written", file=sys.stderr)
        return
    partial = out.with_name(f"{out.name}.partial")
    ExecutionTable(partial).write(results)
    groups = len({result.case.group_id for result in results})
    print(
        f"recording FAILED; wrote the {groups} complete group(s), {len(results)} rows, "
        f"to {partial}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
