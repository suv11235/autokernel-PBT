"""`scripts/gpu_record.py`, exercised on CPU with a fake ladder.

The script is the one piece of code whose failure costs rented time directly: it runs
on the instance, once, after everything else has passed. These tests cover the two
ways it has lost or nearly lost a run -- dying on import, and discarding completed
executions when something late in the loop raised.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import Status
from autokernel_pbt.props.backends.triton_kernel import TritonKernel
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.table import ExecutionTable
from autokernel_pbt.props.tasks import TASKS

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "gpu_record.py"
SEED = 42
#: Softmax, because its case groups have more than one case (base plus a relation), so
#: a failure can land mid-group. Relu's groups are single-case.
TASK = "softmax"


def _fake_kernel(n_cols: int) -> TritonKernel:
    def launcher(*, grid, constexprs, record_compiled, **inputs):
        return np.zeros_like(inputs["x"])

    return TritonKernel("fake_triton", lambda: None, lambda shape, ce: (shape[0],), {}, launcher)


def _fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.__version__ = "2.7.0"
    torch.version = types.SimpleNamespace(cuda="12.8")
    properties = types.SimpleNamespace(
        name="FAKE", major=8, minor=6, multi_processor_count=72, total_memory=1 << 30
    )
    torch.cuda = types.SimpleNamespace(
        current_device=lambda: 0, get_device_properties=lambda index: properties
    )
    torch._C = types.SimpleNamespace(_cuda_getDriverVersion=lambda: 12090)
    return torch


def _run_script(monkeypatch, out: Path, factory) -> int:
    """Run `main()` in-process against a fake `kernels.triton.ladder`."""
    ladder = types.ModuleType("kernels.triton.ladder")
    ladder.KERNELS = {TASK: factory}
    monkeypatch.setitem(sys.modules, "kernels.triton.ladder", ladder)
    # The script builds its backend with the default device probe; give it a torch
    # that answers, so these tests do not lean on the probe's failure guard.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["gpu_record.py", "--task", TASK, "--out", str(out)])
    spec = importlib.util.spec_from_file_location("gpu_record_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


def _groups():
    task = TASKS[TASK]
    return Generator(task.domain, seed=SEED).generate(len(task.domain.shapes))


def test_a_complete_run_writes_the_table_and_exits_zero(monkeypatch, tmp_path, capsys):
    out = tmp_path / "gpu-run"
    assert _run_script(monkeypatch, out, _fake_kernel) == 0
    rows = ExecutionTable(out).read()
    assert len(rows) == sum(len(group.cases) for group in _groups())
    assert {row.status for row in rows} == {Status.OK}
    assert f"wrote {out}" in capsys.readouterr().out


def test_a_failure_mid_run_persists_the_completed_groups_and_reraises(monkeypatch, tmp_path):
    """Executions already paid for are written before the failure propagates.

    Written to a `.partial` sibling, not to `--out`: a re-run that dies must not
    replace an earlier complete table with an incomplete one, and nothing inside a
    table says it is incomplete -- the path has to. Only WHOLE groups are kept: the
    case group is the unit the arms are compared on.
    """
    groups = _groups()
    complete = groups[:2]
    # Fail on the second case of the third group: mid-group, after real work.
    assert len(groups[2].cases) >= 2
    fail_at = sum(len(group.cases) for group in complete) + 2
    calls = []

    def factory(n_cols: int) -> TritonKernel:
        calls.append(n_cols)
        if len(calls) == fail_at:
            msg = "BLOCK exceeds MAX_BLOCK (stand-in for any late bad call)"
            raise ValueError(msg)
        return _fake_kernel(n_cols)

    out = tmp_path / "gpu-run"
    with pytest.raises(ValueError, match="stand-in for any late bad call"):
        _run_script(monkeypatch, out, factory)

    assert not out.exists()
    partial = ExecutionTable(tmp_path / "gpu-run.partial").read()
    assert [row.case.case_id for row in partial] == [
        case.case_id for group in complete for case in group.cases
    ]


def _without(path: Path) -> str:
    """Python code that drops `path` from sys.path, undoing an editable install."""
    return (
        "import os, sys\n"
        f"_drop = os.path.realpath({str(path)!r})\n"
        "sys.path[:] = [p for p in sys.path if os.path.realpath(p or '.') != _drop]\n"
    )


def _python(code: str, cwd: Path) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


def test_the_script_runs_without_the_package_installed(tmp_path):
    # pytest's `pythonpath` covers `src/` for tests only; on the instance the script
    # is run directly, possibly before (or without) `pip install -e`.
    code = _without(REPO_ROOT / "src") + (
        "import runpy\n"
        "sys.argv = ['gpu_record.py', '--help']\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
    )
    proc = _python(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "--task" in proc.stdout


def test_the_repo_s_kernels_wins_over_an_installed_kernels_distribution(tmp_path):
    """A third-party regular package named `kernels` must not shadow this repo's.

    While `kernels/` was a namespace package it lost to ANY regular `kernels` package
    anywhere on sys.path -- the import system prefers a regular package to a namespace
    portion regardless of order -- so the script's own sys.path insert did not help.
    """
    site = tmp_path / "site"
    (site / "kernels").mkdir(parents=True)
    (site / "kernels" / "__init__.py").write_text("# an installed third-party 'kernels'\n")
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"sys.path.append({str(site)!r})\n"
        "import kernels.mutants\n"
        "print(kernels.__file__)\n"
    )
    proc = _python(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()).resolve() == (REPO_ROOT / "kernels" / "__init__.py").resolve()
