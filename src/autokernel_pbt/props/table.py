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

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import load_file, save_file

from autokernel_pbt.props.backends.base import ExecutionResult, Status
from autokernel_pbt.props.backends.telemetry import _Missing
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.spec import CaseSpec

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
        # Ground truth join keys. `kernel_is_broken` is nullable on purpose: a
        # null means "not stated", False means "stated correct". See
        # `ExecutionResult` for why the two must not collapse.
        ("kernel_id", pa.string()),
        ("kernel_is_broken", pa.bool_()),
        # Which corpus these rows are. See `new_corpus_fingerprint`; it is the only
        # thing that lets a reader tell that a `scores.parquet` belongs to the
        # `rows.parquet` beside it.
        ("corpus_fingerprint", pa.string()),
        # JSON, or "" for a group with no recipe (a hand-built test group).
        ("case_spec", pa.string()),
    ]
)


def new_corpus_fingerprint(case_ids: Iterable[str]) -> str:
    """Mint an identity for one recorded corpus. Not a pure function — see below.

    Case ids are a pure function of ``(seed, index)``, so two runs of the same task at
    the same seed produce *identical* ids while recording entirely different executions
    — a different kernel, a different backend, a different day. That is precisely the
    attack `scores.py` calls the worst of the four: a `scores.parquet` from a
    correct-kernel run, dropped beside a broken-kernel `rows.parquet`, joins perfectly
    on every key and reports a detection rate of zero against a kernel labelled broken.

    So the fingerprint mixes two things:

    * a fresh uuid per write, which makes every recording distinct *as an event* — this
      is what defends the copied-file and re-record cases, neither of which a
      content-derived hash could see;
    * the sorted case id set, so the identity is also a statement about what is in the
      table, and a fingerprint carried onto a different set of rows is detectable.

    Being non-deterministic is the point and not a defect: a re-record MUST get a new
    identity, or the scores of the run it replaced would still appear to belong.
    """
    digest = hashlib.sha256()
    digest.update(uuid.uuid4().hex.encode())
    for case_id in sorted(case_ids):
        # A separator, so {"ab", "c"} and {"a", "bc"} cannot hash alike.
        digest.update(b"\x00")
        digest.update(case_id.encode())
    return digest.hexdigest()


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
    if isinstance(obj, _Missing):
        # A field the toolchain did not report. Encoded as null so it round-trips as
        # None, which stays distinguishable from 0 -- the whole reason the sentinel
        # exists rather than a default of zero.
        return None
    msg = f"telemetry value is not JSON-serializable: {obj!r}"
    raise TypeError(msg)


def _conform(record: dict[str, Any], schema: pa.Schema) -> dict[str, Any]:
    """`pa.Table.from_pylist` presence-checks nothing: a missing key becomes a
    fully-null column indistinguishable from an honest unlabelled run, and an
    extra key is dropped. A typo therefore corrupts a metric silently.

    Only the value *types* are checked by the schema, so this is the one place a
    key set can be caught. It is the builder's invariant, not the writer's — the
    read-side column check cannot see this failure at all, because the file it
    receives has every column present and merely lies about one.

    The difference is symmetric on purpose: a typo such as `kernel_is_borken`
    produces one missing key *and* one unexpected one, and naming only the
    missing half sends the reader looking for a deletion that never happened.
    """
    expected = set(schema.names)
    if record.keys() != expected:
        msg = (
            f"record does not match the schema; missing: "
            f"{sorted(expected - record.keys())}, unexpected: "
            f"{sorted(record.keys() - expected)}"
        )
        raise ValueError(msg)
    return record


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

        # One identity per write, minted here rather than by the caller: the corpus is
        # whatever this call is about to record, and a caller-supplied identity could
        # name a different one.
        fingerprint = new_corpus_fingerprint(counts)
        records = [self._record(result, fingerprint) for result in results]
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

    def _record(self, result: ExecutionResult, fingerprint: str) -> dict[str, Any]:
        record = result.case.metadata()
        record["shape"] = json.dumps(record["shape"])
        record["telemetry"] = json.dumps(result.telemetry, default=_json_safe)
        # `Status` is a str subclass, so pyarrow stores the wire value.
        record["status"] = str(result.status)
        # `ExecutionResult.error` is declared `str = ""`; a None from a caller
        # that skipped the default must not widen the column's contract.
        record["error"] = result.error or ""
        # Declared `str = ""`, same as `error` above: a None from a caller that
        # skipped the default must not widen a str-typed column, or a downstream
        # `.startswith(...)` becomes an AttributeError and None groups apart
        # from "".
        record["kernel_id"] = result.kernel_id or ""
        record["kernel_is_broken"] = result.kernel_is_broken
        # Denormalized onto every row, like `arm` in scores.py and for the same reason:
        # Parquet dictionary-encodes a single-valued string column, so a one-row-per-run
        # side table would cost a join to save nothing.
        record["corpus_fingerprint"] = fingerprint
        record["case_spec"] = result.case_spec.to_json() if result.case_spec else ""
        return _conform(record, SCHEMA)

    def read(self) -> list[ExecutionResult]:
        """Every recorded execution, in write order. `[]` if the run is absent.

        A status value with no `Status` member raises rather than degrading to a
        string or a placeholder. That is deliberate: it means a table written by
        a build whose status vocabulary has since changed, which every oracle
        downstream would otherwise silently misclassify.

        A table missing columns this build requires raises for the same reason;
        see `_require_columns`.
        """
        if not self.metadata_path.exists():
            return []
        table = pq.read_table(self.metadata_path)
        self._require_columns(table.schema.names)
        results = []
        for record in table.to_pylist():
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
                    kernel_id=record["kernel_id"],
                    case_spec=(
                        CaseSpec.from_json(record["case_spec"])
                        if record["case_spec"]
                        else None
                    ),
                    kernel_is_broken=record["kernel_is_broken"],
                )
            )
        return results

    def _require_columns(self, present: list[str]) -> None:
        """Reject a table written by a build with a narrower schema.

        Without this the first missing column surfaces as a bare `KeyError:
        'kernel_id'` from inside the row loop, which names neither the run
        directory nor the reason. There is no recorded corpus, so old tables are
        not supported — but the refusal has to say so.
        """
        available = set(present)
        missing = [name for name in SCHEMA.names if name not in available]
        if missing:
            msg = (
                f"{self.metadata_path} was written by a build with a different schema; "
                f"missing columns: {missing}. There is no migration: re-record this run."
            )
            raise ValueError(msg)

    def _status(self, value: str, case_id: str) -> Status:
        try:
            return Status(value)
        except ValueError as exc:
            # A bare "'oom_error' is not a valid Status" from deep inside a read
            # gives no clue which run directory is stale.
            msg = f"{exc} (case {case_id!r} in {self.metadata_path})"
            raise ValueError(msg) from exc

    def corpus_fingerprint(self) -> str:
        """This corpus's identity, or `""` if the run has no rows.

        A property of the *table*, so it is read from the file rather than carried on
        `ExecutionResult`: a row does not know which corpus it is part of, and putting
        it on the row would invite a caller to set it.

        Rows disagreeing about it is not something `write` can emit — one value is
        broadcast to the whole table — so it means a file assembled from two runs, which
        is exactly what the fingerprint exists to catch. It raises rather than picking
        one, for the same reason `ScoreTable.read` refuses a disagreeing `elapsed_s`.
        """
        if not self.metadata_path.exists():
            return ""
        table = pq.read_table(self.metadata_path)
        self._require_columns(table.schema.names)
        distinct = sorted(set(table.column("corpus_fingerprint").to_pylist()))
        if not distinct:
            return ""
        if len(distinct) > 1:
            msg = (
                f"{self.metadata_path} carries {len(distinct)} different corpus "
                f"fingerprints: {distinct}. A single write stamps one, so this table was "
                f"assembled from more than one run and joins to nothing coherently"
            )
            raise ValueError(msg)
        return distinct[0]

    def read_groups(self) -> dict[str, list[ExecutionResult]]:
        """Rows reassembled into case groups, preserving write order."""
        groups: dict[str, list[ExecutionResult]] = {}
        for row in self.read():
            groups.setdefault(row.case.group_id, []).append(row)
        return groups
