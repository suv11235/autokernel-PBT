"""Persisted execution table: Parquet metadata plus safetensors payloads.

This is the artifact the whole architecture exists to produce. One expensive
hardware run becomes a reusable dataset, so a competing oracle can be scored
months later over byte-identical executions without touching a device. If rows
do not survive persistence bitwise, oracle comparison is not fair — so this
module changes nothing about a tensor except making it contiguous enough to
write.

Tensor payloads are far too large for a JSON ledger, and the analysis over rows
is columnar aggregation, so metadata lives in Parquet and tensors in
safetensors.

Layout::

    <run_dir>/rows.parquet          one row per execution, tensor-free
    <run_dir>/tensors/<case_id>.safetensors

PORTABILITY: `case_id` is used verbatim as a filename, and relation-derived ids
contain `::` (see `relations._derived`). That is fine on macOS and Linux, which
is what this project targets, but `::` is not a legal filename character on
Windows. A Windows port needs an escaping layer here; nothing else in the
codebase depends on the filename being the raw id.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import load_file, save_file

from autokernel_pbt.props.backends.base import ExecutionResult, Status
from autokernel_pbt.props.case import Case

METADATA_FILE = "rows.parquet"
TENSOR_DIR = "tensors"

# Inputs and outputs share one safetensors file per row, so they need
# disambiguating prefixes. The separator is `.`, which cannot appear at the
# start of a prefixed key by accident: every key is prefixed exactly once and
# stripped exactly once, so a tensor legitimately named `in_bias`, or even
# `in.y`, keeps its identity.
INPUT_PREFIX = "in."
OUTPUT_PREFIX = "out."

# Every column is a string: `shape` and `telemetry` are JSON-encoded because
# Parquet's nested types would otherwise pin the schema to whatever the first
# batch happened to contain (telemetry's tier-2 fields are deliberately
# free-form). An explicit schema also means `write([])` produces a table with
# real columns rather than a zero-column one.
SCHEMA = pa.schema(
    [
        ("case_id", pa.string()),
        ("group_id", pa.string()),
        ("relation", pa.string()),
        ("task_id", pa.string()),
        ("dtype", pa.string()),
        ("shape", pa.string()),
        ("telemetry", pa.string()),
        ("status", pa.string()),
        ("error", pa.string()),
    ]
)


def _persistable(array: np.ndarray) -> np.ndarray:
    """Make `array` writable by safetensors without changing what it is.

    `np.ascontiguousarray` is load-bearing: a transpose-view kernel returns a
    non-contiguous array that safetensors cannot write at all. But it is
    documented `ndmin=1`, so it silently promotes a 0-d array to shape `(1,)`.
    Outputs have already been through `single_output`'s `np.atleast_1d`, but
    *inputs* have not, and `InputDomain` accepts `shapes=((),)` — so a 0-d input
    tensor is reachable, and would come back a different shape than it went in.

    Persistence must never be the thing that changes a shape; normalization is
    the execution boundary's job (`single_output`), not the ledger's. Reshaping
    back to the original shape is free — reshape on a contiguous array is a
    view — and is a no-op for every array of rank >= 1.
    """
    return np.ascontiguousarray(array).reshape(array.shape)


class ExecutionTable:
    """Read/write the recorded executions for one run."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / METADATA_FILE

    def _tensor_path(self, case_id: str) -> Path:
        return self.run_dir / TENSOR_DIR / f"{case_id}.safetensors"

    def write(self, results: list[ExecutionResult]) -> None:
        """Persist every result, replacing any table already in `run_dir`."""
        (self.run_dir / TENSOR_DIR).mkdir(parents=True, exist_ok=True)
        records = []
        for result in results:
            payload: dict[str, np.ndarray] = {}
            for name, array in result.case.tensors.items():
                payload[f"{INPUT_PREFIX}{name}"] = _persistable(array)
            for name, array in result.outputs.items():
                payload[f"{OUTPUT_PREFIX}{name}"] = _persistable(array)
            # A failed row has no outputs at all; safetensors accepts an empty
            # payload, which keeps read() free of a per-row existence check.
            save_file(payload, str(self._tensor_path(result.case.case_id)))
            record = result.case.metadata()
            record["shape"] = json.dumps(record["shape"])
            record["telemetry"] = json.dumps(result.telemetry)
            # `Status` is a str subclass, so pyarrow stores the wire value.
            record["status"] = str(result.status)
            record["error"] = result.error
            records.append(record)
        pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), self.metadata_path)

    def read(self) -> list[ExecutionResult]:
        """Every recorded execution, in write order. `[]` if the run is absent."""
        if not self.metadata_path.exists():
            return []
        results = []
        for record in pq.read_table(self.metadata_path).to_pylist():
            payload = load_file(str(self._tensor_path(record["case_id"])))
            inputs = {
                key[len(INPUT_PREFIX) :]: value
                for key, value in payload.items()
                if key.startswith(INPUT_PREFIX)
            }
            outputs = {
                key[len(OUTPUT_PREFIX) :]: value
                for key, value in payload.items()
                if key.startswith(OUTPUT_PREFIX)
            }
            case = Case(
                case_id=record["case_id"],
                group_id=record["group_id"],
                relation=record["relation"],
                task_id=record["task_id"],
                dtype=record["dtype"],
                shape=tuple(json.loads(record["shape"])),
                tensors=inputs,
            )
            results.append(
                ExecutionResult(
                    case=case,
                    outputs=outputs,
                    telemetry=json.loads(record["telemetry"]),
                    # Parquet hands back a bare `str`. It compares equal to the
                    # enum member, so `==` would hide this — but `is`, `match`,
                    # and any identity-based dispatch would silently never
                    # match. Reconstruct the member so a replayed row is
                    # indistinguishable from a freshly executed one.
                    status=Status(record["status"]),
                    error=record["error"],
                )
            )
        return results

    def read_groups(self) -> OrderedDict[str, list[ExecutionResult]]:
        """Rows reassembled into case groups, preserving write order."""
        groups: OrderedDict[str, list[ExecutionResult]] = OrderedDict()
        for row in self.read():
            groups.setdefault(row.case.group_id, []).append(row)
        return groups
