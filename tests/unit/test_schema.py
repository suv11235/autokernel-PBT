"""Unit tests for schema validation."""

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from autokernel_pbt import schema
from autokernel_pbt.schema import validate

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize(
    "fixture_name",
    [
        "harness_result_minimal.json",
        "harness_result_success.json",
        "harness_result_failed.json",
        "harness_result_stages.json",
    ],
)
def test_harness_result_fixtures_validate(fixture_name: str):
    data = json.loads((FIXTURES / fixture_name).read_text())
    validate(data, "harness_result.schema.json")


def _drop_passed(data: dict) -> None:
    del data["passed"]


def _future_version(data: dict) -> None:
    data["version"] = "2"


def _stringly_ran(data: dict) -> None:
    data["benchmark"]["ran"] = "yes"


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_drop_passed, r"^'passed' is a required property\n"),
        (_future_version, r"^'1' was expected\n"),
        (_stringly_ran, r"^'yes' is not of type 'boolean'\n"),
    ],
)
def test_an_invalid_harness_result_is_rejected(corrupt, message: str):
    """The fixtures above only ever pass, so a validator that accepts anything passes them.

    One valid fixture, corrupted three ways the schema forbids -- a missing required
    field, a const, a type -- each pinned to the validator's own message so a no-op
    validator, or one that loaded the wrong schema, fails here.
    """
    data = json.loads((FIXTURES / "harness_result_minimal.json").read_text())
    corrupt(data)
    with pytest.raises(ValidationError, match=message):
        validate(data, "harness_result.schema.json")


# --------------------------------------------------------------------------- #
# Where the schemas come from: specs/schemas, in a checkout AND in a wheel
# --------------------------------------------------------------------------- #


def test_a_source_checkout_reads_the_schemas_from_specs():
    """``specs/schemas/`` is the single source of truth; an editable install reads it."""
    assert Path(schema.SCHEMAS) == ROOT / "specs" / "schemas"


def test_the_package_source_carries_no_copy_of_the_schemas():
    """The packaged copy is preferred when present, so an in-tree one would win.

    A ``src/autokernel_pbt/schemas/`` committed by hand would shadow ``specs/schemas/``
    in every editable install, and the two could drift with nothing to notice. The
    copy belongs only in a built wheel.
    """
    assert not (ROOT / "src" / "autokernel_pbt" / schema.PACKAGED_SCHEMAS).exists()


def test_the_packaged_copy_wins_and_the_repo_is_the_fallback(tmp_path: Path):
    packaged, repo = tmp_path / "packaged", tmp_path / "repo"
    repo.mkdir()
    assert schema._schemas_dir(packaged, repo) == repo
    packaged.mkdir()
    assert schema._schemas_dir(packaged, repo) == packaged


def test_the_wheel_ships_specs_schemas_where_the_loader_looks():
    """The non-editable path, pinned on CPU.

    A wheel holds only ``src/autokernel_pbt``, so without a force-include there is no
    ``specs/`` beside an installed package and every ``validate`` -- the CLI's included
    -- dies with FileNotFoundError. Building a wheel needs the network, so the suite
    pins the build config instead, against the name the loader resolves.
    """
    tomllib = pytest.importorskip("tomllib")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    force_include = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include == {"specs/schemas": f"autokernel_pbt/{schema.PACKAGED_SCHEMAS}"}
