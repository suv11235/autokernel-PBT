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
import json
from typing import Any

import numpy as np

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

#: Facts the backend observes about the execution rather than reads off an artifact.
#: Defaults rather than MISSING: these are always knowable, so an absent value would
#: mean the backend forgot to look, not that the toolchain declined to say.
#:
#: `input_mutated` separates a kernel that wrote to its own input from one that merely
#: crashed. Both are LAUNCH_ERROR -- neither is evidence about numerics -- but they are
#: different fault classes, and collapsing them would make the distinction
#: unrecoverable from the artifacts.
_FLAG_DEFAULTS: dict[str, Any] = {"input_mutated": False}


def declared_keys() -> tuple[str, ...]:
    """Every key the schema promises to emit, present or MISSING."""
    return (
        TELEMETRY_SCHEMA_VERSION,
        *_COMPILED_FIELDS,
        *_DEVICE_KEYS,
        *_LAUNCH_KEYS,
        *_DERIVED_KEYS,
        *_FLAG_DEFAULTS,
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
            # Not `hasattr`, which swallows only AttributeError: an attribute that
            # lazily touches the driver can raise anything, and that is one location
            # being unavailable, not a reason to abort the run.
            try:
                current = getattr(current, part)
            except Exception:  # noqa: BLE001 - probed, not asserted
                current = MISSING
                break
        if current is not MISSING:
            return current
    return MISSING


def _table_encoding(obj: Any) -> Any:
    """The execution table's encoding of non-JSON values, mirrored for a dry run.

    Mirrors `table._json_safe` rather than importing it: the table imports this
    module, and the schema must not depend on the persistence layer. Kept no more
    permissive than the table, so whatever passes here also encodes there.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is MISSING:
        return None
    msg = f"not encodable: {type(obj).__name__}"
    raise TypeError(msg)


def _encodable(value: Any) -> Any:
    """`value` if the execution table can encode it, else MISSING.

    Telemetry is JSON-encoded once, by the end-of-run table write -- after every
    execution has been paid for. One value of an unexpected type from a new Triton or
    torch release raised there and discarded the whole run. Checking at extraction
    degrades that one field instead, which is the probed-not-asserted contract.
    """
    try:
        json.dumps(value, default=_table_encoding)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return MISSING
    return value


def _hash(text: Any) -> Any:
    if not isinstance(text, str):
        return MISSING
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_CHARS]


def extract(
    compiled: Any,
    *,
    device: dict[str, Any],
    launch: dict[str, Any],
    flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one execution's telemetry.

    `device` and `launch` are merged verbatim -- the backend knows those and does not
    need to probe for them. Everything off the compiled artifact is probed. Any value
    the execution table could not encode is recorded as MISSING; see `_encodable`.

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
    for key, default in _FLAG_DEFAULTS.items():
        out[key] = (flags or {}).get(key, default)
    # Every group, not only the probed one: a device query and a launch constexpr are
    # just as able to hand back a type the table cannot encode, and fail just as late.
    return {key: _encodable(value) for key, value in out.items()}
