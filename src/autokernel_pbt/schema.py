"""JSON Schema validation helpers.

The schemas live in ``specs/schemas/``, beside the specs they belong to, and that
directory is the single source of truth. A wheel holds only ``src/autokernel_pbt``,
so the build copies it into the package as ``autokernel_pbt/schemas/`` (hatch
``force-include`` in ``pyproject.toml``) and an installed copy reads that. An
editable install has no such copy and reads the repo directly. Deriving the repo
path from ``__file__`` alone was right under test and wrong in a wheel, where there
is no ``specs/`` above site-packages and every ``validate`` -- the CLI's included --
died with FileNotFoundError.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

#: The package-relative directory a built wheel carries the schemas in. Must match
#: the ``force-include`` target in pyproject.toml; ``test_schema.py`` pins the pair.
PACKAGED_SCHEMAS = "schemas"

ROOT = Path(__file__).resolve().parents[2]
REPO_SCHEMAS = ROOT / "specs" / "schemas"


def _schemas_dir(packaged: Traversable, repo: Path) -> Traversable:
    """The packaged copy when the install has one, else the repo's ``specs/schemas``."""
    return packaged if packaged.is_dir() else repo


SCHEMAS = _schemas_dir(resources.files("autokernel_pbt") / PACKAGED_SCHEMAS, REPO_SCHEMAS)


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMAS / name).read_text())


def validate(instance: dict[str, Any], schema_name: str) -> None:
    schema = load_schema(schema_name)
    Draft202012Validator(schema).validate(instance)
