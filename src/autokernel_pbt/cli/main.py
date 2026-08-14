"""CLI entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autokernel_pbt.harness.runner import load_config, run_harness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="akpbt", description="autokernel-PBT utilities")
    sub = parser.add_subparsers(dest="command")

    bench = sub.add_parser("bench", help="Run harness evaluation")
    bench.add_argument("--kernel", required=True, help="Path to candidate kernel module")
    bench.add_argument("--reference", required=True, help="Path to reference module")
    bench.add_argument("--config", type=Path, default=None, help="YAML config path")
    bench.add_argument("--dry-run", action="store_true", help="Skip GPU benchmark")
    bench.add_argument("--json", action="store_true", help="Print HarnessResult JSON")

    args = parser.parse_args(argv)
    if args.command == "bench":
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

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
