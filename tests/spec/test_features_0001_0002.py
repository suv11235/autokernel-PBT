"""Spec-derived acceptance tests (features 0001–0002)."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from autokernel_pbt.cli.main import main
from autokernel_pbt.harness.runner import load_config, run_harness
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
        # The exit code is the assertion below, not a precondition for it.
        check=False,
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
        # The exit code is the assertion below, not a precondition for it.
        check=False,
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


def test_load_config_without_a_path_means_the_defaults():
    assert load_config(None) == {}


def test_load_config_reads_an_explicit_path(tmp_path: Path):
    path = tmp_path / "one_stage.yaml"
    path.write_text("correctness:\n  stages: [smoke]\n")
    assert load_config(path) == {"correctness": {"stages": ["smoke"]}}


def test_load_config_refuses_a_path_that_does_not_exist(tmp_path: Path):
    """A mistyped ``--config`` once fell back to the defaults without a word.

    ``None`` means "use the defaults"; a path is a request for THAT file. Treating a
    missing file as the defaults ran all five stages for someone who asked for one,
    and the result validated cleanly, so nothing downstream could tell.
    """
    missing = tmp_path / "configs" / "defualt.yaml"
    with pytest.raises(
        FileNotFoundError, match=rf"^config file {re.escape(str(missing))} does not exist; "
    ):
        load_config(missing)


def test_bench_cli_reports_a_missing_config_as_a_usage_error(tmp_path: Path, capsys):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SystemExit) as excinfo:
        main(["bench", "--kernel", "k", "--reference", "r", "--config", str(missing), "--dry-run"])
    assert excinfo.value.code == 2
    assert f"error: config file {missing} does not exist; " in capsys.readouterr().err
