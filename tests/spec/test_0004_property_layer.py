"""Spec-derived acceptance tests (feature 0004).

These assert traceability: every criterion in acceptance.yaml must name a test that
actually exists and is collectable. This is the mechanism the SDD ADR asks for.
"""

import subprocess
import sys

import pytest
import yaml

ACCEPTANCE = "specs/features/0004-property-oracle-layer/acceptance.yaml"


def _criteria(repo_root):
    data = yaml.safe_load((repo_root / ACCEPTANCE).read_text())
    return data["criteria"]


@pytest.mark.spec
def test_0004_acceptance_file_is_wellformed(repo_root):
    data = yaml.safe_load((repo_root / ACCEPTANCE).read_text())
    assert data["feature_id"] == "0004"
    ids = [c["id"] for c in data["criteria"]]
    assert ids, "acceptance.yaml declares no criteria"
    assert len(ids) == len(set(ids)), f"duplicate criterion ids: {ids}"


@pytest.mark.spec
def test_0004_every_criterion_names_an_existing_file(repo_root):
    missing = []
    for criterion in _criteria(repo_root):
        check = criterion["check"]
        if check["type"] != "unit_test":
            continue
        path = check["test"].split("::")[0]
        if not (repo_root / path).exists():
            missing.append(f"{criterion['id']} -> {path}")
    assert not missing, f"criteria reference missing test files: {missing}"


@pytest.mark.spec
def test_0004_every_criterion_is_collectable(repo_root):
    """A criterion pointing at a non-existent test node is untraceable, so it fails."""
    node_ids = [
        c["check"]["test"] for c in _criteria(repo_root) if c["check"]["type"] == "unit_test"
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *node_ids],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert proc.returncode == 0, f"pytest could not collect all criteria:\n{proc.stdout}\n{proc.stderr}"
