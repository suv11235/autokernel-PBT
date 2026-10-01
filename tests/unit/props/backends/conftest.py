"""Fixtures for CPU tests of device-side modules.

`kernels/triton/ladder.py` and the mutant modules import torch and triton at module
scope, so on a machine with neither they cannot even be imported. Their *structure* --
tile-width derivation, launcher guards, registration metadata -- is still CPU-checkable,
because none of it calls into either library until a kernel is actually launched.
"""

from __future__ import annotations

import sys
import types

import pytest

#: Top-level names whose modules are replaced (or, for `kernels`, re-imported against
#: the replacements) for the duration of one test.
_STUBBED_ROOTS = ("torch", "triton", "kernels")


def _is_stubbed(name: str) -> bool:
    return any(name == root or name.startswith(f"{root}.") for root in _STUBBED_ROOTS)


@pytest.fixture
def device_stubs():
    """Install a bare `torch` and a `triton` whose `jit` is the identity.

    Restores `sys.modules` exactly afterwards, including on a machine that HAS the real
    libraries: a `kernels.triton.ladder` left bound to the stub would be picked up by a
    later device test in the same session and fail there for no reason of its own.
    """
    saved = {name: module for name, module in sys.modules.items() if _is_stubbed(name)}
    for name in saved:
        del sys.modules[name]

    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    triton.jit = lambda fn: fn
    triton.language = language
    torch = types.ModuleType("torch")
    sys.modules.update({"triton": triton, "triton.language": language, "torch": torch})
    try:
        yield types.SimpleNamespace(triton=triton, torch=torch)
    finally:
        for name in [name for name in sys.modules if _is_stubbed(name)]:
            del sys.modules[name]
        sys.modules.update(saved)
