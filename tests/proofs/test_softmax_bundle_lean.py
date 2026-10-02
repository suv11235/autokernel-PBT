"""The oracle proof compiles, closes every goal, and rests on Lean's standard axioms only.

Skipped when the pinned Lean toolchain is absent, as GPU tests are when CUDA is: the proof is
optional infrastructure, and CI stays toolchain-free. When it runs, it is the unique catcher of
three ways the proof could rot: a goal left open (``sorryAx``), a declared ``axiom`` smuggled
in, or a theorem silently deleted.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = ROOT / "proofs" / "softmax-bundle"
PROOF = PROOF_DIR / "SoftmaxBundle.lean"

#: Every theorem the record quotes; a renamed or deleted one fails here.
AUDITED = (
    "SoftmaxBundle.softmax_unique",
    "SoftmaxBundle.softmaxBeta_satisfies_bundle",
    "SoftmaxBundle.bundle_incomplete",
    "SoftmaxBundle.expAxioms_scaled",
    "SoftmaxBundle.softmaxRowWith_scaled",
    "SoftmaxBundle.tangent_rigid",
)
#: Lean's standard axioms. Anything else — ``sorryAx`` above all — is a hole.
STANDARD = {"propext", "Classical.choice", "Quot.sound"}

_AXIOMS = re.compile(r"'(?P<name>\S+)' depends on axioms: \[(?P<axioms>[^\]]*)\]")


def _pinned_lean() -> Path | None:
    """The binary of the toolchain ``lean-toolchain`` pins, or None if it is not installed."""
    pin = (PROOF_DIR / "lean-toolchain").read_text().strip()  # e.g. leanprover/lean4:v4.33.0
    name = pin.replace("/", "--").replace(":", "---")
    home = Path(os.environ.get("ELAN_HOME", Path.home() / ".elan"))
    lean = home / "toolchains" / name / "bin" / "lean"
    return lean if lean.exists() else None


@pytest.mark.formal
def test_the_oracle_proof_compiles_on_standard_axioms_only():
    lean = _pinned_lean()
    if lean is None:
        pytest.skip("the pinned Lean toolchain is not installed")
    done = subprocess.run(
        [str(lean), PROOF.name],
        cwd=PROOF_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "sorry" not in PROOF.read_text(), "a goal is left open in the source"
    audited = {
        m["name"]: set(filter(None, map(str.strip, m["axioms"].split(","))))
        for m in _AXIOMS.finditer(done.stdout)
    }
    missing = [name for name in AUDITED if name not in audited]
    assert not missing, f"not audited by #print axioms: {missing}"
    beyond = {name: axioms - STANDARD for name, axioms in audited.items() if axioms - STANDARD}
    assert not beyond, f"theorems resting on non-standard axioms: {beyond}"
