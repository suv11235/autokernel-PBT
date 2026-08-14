"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def load_fixture():
    def _load(name: str) -> dict:
        return json.loads((FIXTURES / name).read_text())

    return _load
