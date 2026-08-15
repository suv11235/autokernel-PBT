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

INVARIANT: the Parquet index and the payload set are never observed out of
step. `rows.parquet` is the index — `read()` opens only the payloads it names —
so `write()` retires the index *first* and republishes it last, atomically.
Every observable state is therefore either a complete table or an absent one.
See `write()` for why a torn table is worse than a lost one.

FIDELITY: tensors round-trip bitwise; `telemetry` does not, and cannot. It is
JSON-encoded, and JSON has no tuple and no non-string key, so `(1, 2)` returns
as `[1, 2]` and `{0: "sm"}` as `{"0": "sm"}`. Numpy scalars and arrays are
coerced to their Python equivalents on write (see `_json_safe`). Telemetry is
counters and labels, where this is harmless; anything needing byte fidelity is
a tensor and belongs in the payload.

PORTABILITY: `case_id` is used verbatim as a filename, and relation-derived ids
contain `::` (see `relations._derived`). That is fine on macOS and Linux, which
is what this project targets, but `::` is not a legal filename character on
Windows. A Windows port needs an escaping layer here; nothing else in the
codebase depends on the filename being the raw id.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import load_file, save_file

from autokernel_pbt.props.backends.base import ExecutionResult, Status
from autokernel_pbt.props.case import Case

METADATA_FILE = "rows.parquet"
TENSOR_DIR = "tensors"
# Staged inside run_dir so the final publish is a same-filesystem rename.
METADATA_TMP = f".{METADATA_FILE}.tmp"

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


def _json_safe(obj: Any) -> Any:
    """Coerce a numpy telemetry value to its Python equivalent, or fail loudly.

    `base.py` gates output *dtypes* at the execution boundary precisely because
    an unpersistable value would "take the entire run's persistence with it".
    Telemetry needs the same gate one field over: a Phase 3 backend reporting a
    counter read from a device query gets an `np.int64`, and `json.dumps` raises
    on it. `np.float64` slips through unaided because it subclasses `float`,
    which makes the gap easy to miss — `np.float32`, `np.int64` and `np.bool_`
    do not.

    Anything genuinely unrepresentable still raises, before any payload has been
    touched, so the run's existing table survives intact.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    msg = f"telemetry value is not JSON-serializable: {obj!r}"
    raise TypeError(msg)


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
        """Persist every result, replacing any table already in `run_dir`.

        Ordered so that a crash can lose the table but can never tear it.

        Writing payloads in a loop and the index once at the end looks safe, and
        is — unless the run already has a table and the write reuses case_ids,
        which is the normal case for a re-run or a resume. Then a crash midway
        leaves v2 tensor bytes under v1 metadata, and `read()` reports it
        without raising: telemetry from one execution paired with the tensors of
        another. Every oracle scored against such a table is scored against an
        execution that never happened, and nothing anywhere says so. A table
        that is merely *gone* is recoverable by re-running; a table that reads
        clean and lies is not.

        So: encode the metadata first (pure, and the only step that inspects
        caller-supplied telemetry, so a bad value aborts before anything on disk
        is touched), then retire the index, then rebuild the payloads — which
        are unreferenced from that moment, so clearing the directory outright is
        safe and reclaims any orphans from a previous longer write — then
        publish the new index with a single atomic rename.
        """
        # Same failure mode as a torn rewrite, reachable with no crash at all:
        # case_id is the payload filename, so two rows sharing one would write
        # two sets of tensors to a single file and both read back the survivor's
        # bytes under their own metadata. `CaseGroup` enforces uniqueness within
        # a group but nothing does across a batch, so the ledger checks it.
        counts = Counter(result.case.case_id for result in results)
        duplicates = sorted(case_id for case_id, n in counts.items() if n > 1)
        if duplicates:
            msg = f"duplicate case_ids in one write, which would share a payload: {duplicates}"
            raise ValueError(msg)

        records = [self._record(result) for result in results]
        table = pa.Table.from_pylist(records, schema=SCHEMA)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        # From here until the rename below, this run observably has no table.
        self.metadata_path.unlink(missing_ok=True)
        tensor_dir = self.run_dir / TENSOR_DIR
        shutil.rmtree(tensor_dir, ignore_errors=True)
        tensor_dir.mkdir(parents=True)

        for result in results:
            payload: dict[str, np.ndarray] = {}
            for name, array in result.case.tensors.items():
                payload[f"{INPUT_PREFIX}{name}"] = _persistable(array)
            for name, array in result.outputs.items():
                payload[f"{OUTPUT_PREFIX}{name}"] = _persistable(array)
            # A failed row has no outputs at all; safetensors accepts an empty
            # payload, which keeps read() free of a per-row existence check.
            save_file(payload, str(self._tensor_path(result.case.case_id)))

        staged = self.run_dir / METADATA_TMP
        try:
            pq.write_table(table, staged)
            # The publish. os.replace is atomic within a filesystem, and `staged`
            # is a sibling, so no reader ever sees a partial index.
            os.replace(staged, self.metadata_path)
        finally:
            staged.unlink(missing_ok=True)

    def _record(self, result: ExecutionResult) -> dict[str, Any]:
        record = result.case.metadata()
        record["shape"] = json.dumps(record["shape"])
        record["telemetry"] = json.dumps(result.telemetry, default=_json_safe)
        # `Status` is a str subclass, so pyarrow stores the wire value.
        record["status"] = str(result.status)
        # `ExecutionResult.error` is declared `str = ""`; a None from a caller
        # that skipped the default must not widen the column's contract.
        record["error"] = result.error or ""
        return record

    def read(self) -> list[ExecutionResult]:
        """Every recorded execution, in write order. `[]` if the run is absent.

        A status value with no `Status` member raises rather than degrading to a
        string or a placeholder. That is deliberate: it means a table written by
        a build whose status vocabulary has since changed, which every oracle
        downstream would otherwise silently misclassify.
        """
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
                    status=self._status(record["status"], record["case_id"]),
                    error=record["error"],
                )
            )
        return results

    def _status(self, value: str, case_id: str) -> Status:
        try:
            return Status(value)
        except ValueError as exc:
            # A bare "'oom_error' is not a valid Status" from deep inside a read
            # gives no clue which run directory is stale.
            msg = f"{exc} (case {case_id!r} in {self.metadata_path})"
            raise ValueError(msg) from exc

    def read_groups(self) -> dict[str, list[ExecutionResult]]:
        """Rows reassembled into case groups, preserving write order."""
        groups: dict[str, list[ExecutionResult]] = {}
        for row in self.read():
            groups.setdefault(row.case.group_id, []).append(row)
        return groups
