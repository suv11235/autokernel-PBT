"""Unit tests for schema validation."""

import json
from pathlib import Path

import pytest

from autokernel_pbt.schema import validate

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
