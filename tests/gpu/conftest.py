"""Fixtures for device tests.

Everything here skips cleanly when there is no CUDA, so `pytest` on a developer
machine and in CI is green without a GPU. These tests are run by hand on the rented
instance; none of them is an acceptance criterion, for exactly that reason.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def torch_cuda():
    """The torch module, or a skip if CUDA is unavailable.

    Imported inside the fixture rather than at module scope: torch is an optional
    extra, and a module-level import would make collection fail on a machine without
    it rather than skip.
    """
    torch = pytest.importorskip("torch", reason="torch is not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    return torch


@pytest.fixture(scope="session")
def triton_module():
    return pytest.importorskip("triton", reason="triton is not installed")
