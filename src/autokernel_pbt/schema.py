"""JSON Schema validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "specs" / "schemas"


def load_schema(name: str) -> dict[str, Any]:
    path = SCHEMAS / name
    return json.loads(path.read_text())


def validate(instance: dict[str, Any], schema_name: str) -> None:
    schema = load_schema(schema_name)
    Draft202012Validator(schema).validate(instance)
