"""Spec-derived acceptance tests (feature 0006).

These assert traceability: every criterion in acceptance.yaml must name a test that
actually exists and is collectable. This is the mechanism the SDD ADR asks for.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ACCEPTANCE = "specs/features/0006-four-arms/acceptance.yaml"

#: Check vocabulary already in use across features 0001, 0002, 0004 and 0005. A criterion
#: naming anything else is almost certainly a typo, which would silently drop it from the
#: traceability checks below.
KNOWN_CHECK_TYPES = {
    "unit_test",
    "cli_help",
    "json_schema",
    "field_present",
    "field_equals",
    "config_equals",
}


def _criteria(repo_root: Path) -> list[dict]:
    data = yaml.safe_load((repo_root / ACCEPTANCE).read_text())
    return data["criteria"]


@pytest.mark.spec
def test_0006_acceptance_file_is_wellformed(repo_root: Path):
    data = yaml.safe_load((repo_root / ACCEPTANCE).read_text())
    assert data["feature_id"] == "0006"
    ids = [c["id"] for c in data["criteria"]]
    assert ids, "acceptance.yaml declares no criteria"
    assert len(ids) == len(set(ids)), f"duplicate criterion ids: {ids}"
    unknown = [
        f"{c['id']} -> {c['check']['type']}"
        for c in data["criteria"]
        if c["check"]["type"] not in KNOWN_CHECK_TYPES
    ]
    assert not unknown, f"criteria use unknown check types: {unknown}"
    targets = [
        c["check"]["test"] for c in data["criteria"] if c["check"]["type"] == "unit_test"
    ]
    # Two criteria sharing a node id both look traced while only one has independent
    # evidence — the same shape as a criterion certified by an assertion that never ran.
    assert len(targets) == len(set(targets)), f"criteria share test targets: {targets}"
    # A node id without `::` traces to a *file*, not an obligation: the module collects,
    # so every check below passes while the named behaviour need not exist.
    file_only = [
        f"{c['id']} -> {c['check']['test']}"
        for c in data["criteria"]
        if c["check"]["type"] == "unit_test" and "::" not in c["check"]["test"]
    ]
    assert not file_only, f"criteria name a file rather than a test node: {file_only}"


@pytest.mark.spec
def test_0006_every_criterion_names_an_existing_file(repo_root: Path):
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
def test_0006_every_criterion_is_collectable(repo_root: Path):
    """A criterion pointing at a non-existent test node is untraceable, so it fails."""
    node_ids = [
        c["check"]["test"] for c in _criteria(repo_root) if c["check"]["type"] == "unit_test"
    ]
    # Without this guard an empty list would degrade the command to a bare collect over
    # `testpaths`, which exits 0 — reporting success while tracing nothing.
    assert node_ids, "no unit_test criteria to collect"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *node_ids],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=120,
        # The non-zero exit *is* the signal under test; the assertion below reports it
        # with pytest's own stdout, which check=True would discard into a traceback.
        check=False,
    )
    assert proc.returncode == 0, (
        f"pytest could not collect all criteria:\n{proc.stdout}\n{proc.stderr}"
    )
