"""The device telemetry schema.

This is the one part of a hardware run that a re-run cannot recover. Everything else
-- every verdict, every rate, every arm -- is re-derivable offline from the recorded
table for free. A counter that was not captured costs another rented hour, which is
the cost the whole record/replay architecture exists to avoid. So the schema
over-captures, and it is written and tested on CPU where it is cheap to get right.

TWO RULES, both learned from what goes wrong in aggregates months later.

*A declared key is always present.* A field that is simply omitted when unavailable
makes "the toolchain did not report this" indistinguishable from "this was zero", and
the two mean opposite things about a kernel's register pressure. Unavailable fields
carry the `MISSING` sentinel instead.

*Extraction is defensive, not assertive.* Triton's introspection surface has moved
between releases -- register and spill counts have lived on the compiled kernel object
and on its metadata at different times -- and this module is written on a machine with
no Triton to check against. `probe` therefore names several candidate locations per
field and takes the first that exists, so a version bump degrades one field to
MISSING rather than raising mid-run and discarding the executions already paid for.
"""

from __future__ import annotations

import hashlib
from typing import Any

#: Bump when a field is added or its meaning changes. Recorded on every row so a
#: later reader can distinguish a run taken before a field existed from one where the
#: field was genuinely unavailable -- without it the two are the same absence.
SCHEMA_VERSION = 1

TELEMETRY_SCHEMA_VERSION = "telemetry_schema_version"

#: Length of the truncated artifact digests. 64 bits separates the handful of kernel
#: variants in one experiment; these are not adversarial inputs.
HASH_CHARS = 16


class _Missing:
    """Sentinel type for a field the toolchain did not report.

    A plain class with exactly one instance below, rather than a __new__-enforced
    singleton: enforcing it would need `typing.Self`, which is 3.11+, and this project
    declares >=3.10. Nothing constructs a second one, and `is MISSING` is the only
    test anyone performs.

    Telemetry is JSON-encoded into the execution row, and `table._json_safe` encodes
    this as null -- which round-trips as None and stays distinguishable from 0, the
    whole reason the sentinel exists rather than a default of zero.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


MISSING = _Missing()

#: field -> candidate locations on the compiled kernel, in priority order. Dotted
#: paths walk attributes. Several entries per field is the point: see the module
#: docstring on why this is probed rather than asserted.
_COMPILED_FIELDS: dict[str, tuple[str, ...]] = {
    "n_regs": ("n_regs", "metadata.n_regs", "num_regs"),
    "n_spills": ("n_spills", "metadata.n_spills", "num_spills"),
    "shared_bytes": ("shared", "metadata.shared", "metadata.shared_mem"),
    "num_warps": ("num_warps", "metadata.num_warps"),
    "num_stages": ("num_stages", "metadata.num_stages"),
}

#: Groups the backend supplies directly rather than probing off the artifact.
_DEVICE_KEYS = (
    "device_name",
    "compute_capability",
    "multi_processor_count",
    "total_memory_bytes",
    "driver_version",
    "runtime_version",
    "torch_version",
    "triton_version",
)
_LAUNCH_KEYS = ("grid", "constexprs")
_DERIVED_KEYS = ("ptx_hash",)


def declared_keys() -> tuple[str, ...]:
    """Every key the schema promises to emit, present or MISSING."""
    return (
        TELEMETRY_SCHEMA_VERSION,
        *_COMPILED_FIELDS,
        *_DEVICE_KEYS,
        *_LAUNCH_KEYS,
        *_DERIVED_KEYS,
    )


def probe(obj: Any, locations: tuple[str, ...]) -> Any:
    """First existing location's value, or MISSING.

    `getattr` chains rather than a single name because the same field lives in
    different places across Triton releases. Absence is distinguished from a falsy
    value: `n_spills == 0` is the *good* outcome and must never read as unavailable.
    """
    for location in locations:
        current: Any = obj
        for part in location.split("."):
            if not hasattr(current, part):
                current = MISSING
                break
            current = getattr(current, part)
        if current is not MISSING:
            return current
    return MISSING


def _hash(text: Any) -> Any:
    if not isinstance(text, str):
        return MISSING
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_CHARS]


def extract(compiled: Any, *, device: dict[str, Any], launch: dict[str, Any]) -> dict[str, Any]:
    """Assemble one execution's telemetry.

    `device` and `launch` are merged verbatim -- the backend knows those and does not
    need to probe for them. Everything off the compiled artifact is probed.

    The PTX is hashed rather than stored. A large kernel's PTX is tens of kilobytes
    and would be repeated on every row of every group; the hash identifies the
    artifact, and the text itself is recoverable by recompiling from the recorded
    kernel source hash.
    """
    out: dict[str, Any] = {TELEMETRY_SCHEMA_VERSION: SCHEMA_VERSION}
    for field, locations in _COMPILED_FIELDS.items():
        out[field] = probe(compiled, locations)

    asm = probe(compiled, ("asm",))
    ptx = asm.get("ptx") if isinstance(asm, dict) else MISSING
    out["ptx_hash"] = _hash(ptx)

    for key in _DEVICE_KEYS:
        out[key] = device.get(key, MISSING)
    for key in _LAUNCH_KEYS:
        out[key] = launch.get(key, MISSING)
    return out
