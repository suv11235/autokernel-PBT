"""Spec-derived acceptance tests (features 0001–0003)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autokernel_pbt.harness.runner import run_harness
from autokernel_pbt.schema import validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.spec
def test_0001_harness_dry_run_schema(repo_root: Path):
    result = run_harness(
        str(repo_root / "kernels/triton/candidate.py"),
        str(repo_root / "kernels/triton/reference_relu.py"),
        dry_run=True,
    )
    validate(result, "harness_result.schema.json")
    assert result["benchmark"]["ran"] is True
    assert "speedup_vs_eager" in result["benchmark"]


@pytest.mark.spec
def test_0001_bench_cli_help(repo_root: Path):
    proc = subprocess.run(
        [sys.executable, str(repo_root / "harness/bench.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert proc.returncode == 0
    assert "--kernel" in proc.stdout


@pytest.mark.spec
def test_0001_bench_cli_json(repo_root: Path):
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "harness/bench.py"),
            "--kernel",
            "kernels/triton/candidate.py",
            "--reference",
            "kernels/triton/reference_relu.py",
            "--dry-run",
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    validate(data, "harness_result.schema.json")


@pytest.mark.spec
def test_0002_default_config_stage_order(repo_root: Path):
    import yaml

    cfg = yaml.safe_load((repo_root / "harness/configs/default.yaml").read_text())
    assert cfg["correctness"]["stages"] == [
        "smoke",
        "shape_sweep",
        "numerical_stress",
        "determinism",
        "edge_cases",
    ]
