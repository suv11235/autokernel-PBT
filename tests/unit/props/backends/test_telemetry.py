"""Telemetry schema tests.

The schema is the one thing a re-run cannot recover, so it is tested hard and on CPU.
Extraction is deliberately defensive: Triton's introspection surface has moved between
releases, and a field that silently vanishes looks exactly like hardware that does not
report it.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from autokernel_pbt.props.backends.telemetry import (
    MISSING,
    SCHEMA_VERSION,
    TELEMETRY_SCHEMA_VERSION,
    declared_keys,
    extract,
    probe,
)


class _Meta:
    """Stands in for a Triton CompiledKernel's metadata."""

    num_warps = 4
    num_stages = 3
    shared = 8192


class _Compiled:
    """Stands in for a Triton CompiledKernel."""

    n_regs = 40
    n_spills = 0
    metadata = _Meta()
    asm: ClassVar[dict[str, str]] = {"ptx": "// ptx text", "ttir": "// ttir"}


def test_schema_version_is_recorded():
    """The criterion TELEMETRY_CARRIES_A_SCHEMA_VERSION."""
    out = extract(_Compiled(), device={}, launch={})
    assert out[TELEMETRY_SCHEMA_VERSION] == SCHEMA_VERSION
    assert isinstance(SCHEMA_VERSION, int)


def test_every_declared_key_is_present():
    """The criterion TELEMETRY_SCHEMA_IS_COMPLETE.

    Every declared key appears on every execution. A key that is merely absent when
    unavailable makes "not captured" and "captured as zero" indistinguishable in an
    aggregate months later.
    """
    out = extract(_Compiled(), device={}, launch={})
    assert set(declared_keys()) <= set(out)


def test_an_unavailable_field_is_recorded_as_missing():
    """The criterion MISSING_FIELDS_ARE_RECORDED_NOT_DROPPED."""

    class _Bare:
        metadata = _Meta()

    out = extract(_Bare(), device={}, launch={})
    assert out["n_regs"] is MISSING
    assert "n_regs" in out


def test_probe_reads_the_first_location_that_exists():
    # Triton has moved these between the kernel object and its metadata across
    # releases, so extraction names several locations rather than one.
    assert probe(_Compiled(), ("nowhere", "n_regs")) == 40
    assert probe(_Compiled(), ("metadata.num_warps",)) == 4


def test_probe_returns_missing_when_no_location_exists():
    assert probe(_Compiled(), ("absent", "also.absent")) is MISSING


def test_probe_does_not_confuse_a_falsy_value_with_absence():
    # n_spills == 0 is the *good* case and must not read as "unavailable".
    assert probe(_Compiled(), ("n_spills",)) == 0
    assert probe(_Compiled(), ("n_spills",)) is not MISSING


def test_ptx_is_hashed_not_stored():
    # The PTX of a large kernel is tens of kilobytes and would be carried on every
    # row of the group. The hash identifies the artifact; the text is recoverable by
    # recompiling from the recorded source hash.
    out = extract(_Compiled(), device={}, launch={})
    assert out["ptx_hash"] != "// ptx text"
    assert len(out["ptx_hash"]) == 16


def test_absent_asm_hashes_to_missing():
    class _NoAsm:
        metadata = _Meta()

    assert extract(_NoAsm(), device={}, launch={})["ptx_hash"] is MISSING


def test_device_and_launch_groups_are_merged_verbatim():
    out = extract(_Compiled(), device={"device_name": "A10"}, launch={"grid": [8, 1, 1]})
    assert out["device_name"] == "A10"
    assert out["grid"] == [8, 1, 1]


def test_missing_is_json_serializable():
    # Telemetry is JSON-encoded into the execution row; a sentinel that cannot be
    # encoded would take the whole run's persistence with it. It encodes as null, so
    # it round-trips as None -- distinguishable from 0, which is the point.
    from autokernel_pbt.props.table import _json_safe

    encoded = json.dumps(extract(_Compiled(), device={}, launch={}), default=_json_safe)
    assert json.loads(encoded)["device_name"] is None


@pytest.mark.parametrize("key", ["n_regs", "n_spills", "shared_bytes", "num_warps", "num_stages"])
def test_the_fault_class_relevant_keys_are_declared(key):
    # These are the ISSTA taxonomy's device-only signals: register pressure, spills,
    # and the launch geometry a tile compiler chose. Losing one costs a hardware run.
    assert key in declared_keys()
