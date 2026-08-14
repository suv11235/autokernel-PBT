#!/usr/bin/env python3
"""Fixed harness entrypoint for agent/PBT loops (spec 0001).

Usage:
  python harness/bench.py --kernel kernels/triton/candidate.py \\
      --reference kernels/triton/reference_relu.py --dry-run --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autokernel_pbt.harness.runner import load_config, run_harness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="autokernel-PBT evaluation harness")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "harness" / "configs" / "default.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit HarnessResult JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_harness(
        args.kernel,
        args.reference,
        cfg,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
