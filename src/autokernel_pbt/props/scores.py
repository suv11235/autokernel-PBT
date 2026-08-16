"""Persisted oracle scores: one Parquet row per property verdict.

`PropertyResult` is the research output. Until this module existed it had no
serialization path at all, so every verdict died with the process that produced
it and the tolerance-free split existed only in memory.

Two fields live on the *arm* rather than the result, and both are why the file
exists in this shape:

* ``arm`` — which oracle strategy produced the verdict. Without it an arm-vs-arm
  comparison is not computable from the artifact, only from the code that wrote
  it.
* ``elapsed_s`` — what that arm cost. Cost-per-bug is bugs over seconds; the
  seconds are wall-clock and are *not* re-derivable by re-reading anything.

Both are denormalized onto every row. Parquet is columnar and dictionary-encodes
a low-cardinality string column, so the duplication is nearly free, and it keeps
the file a single flat table that a downstream `GROUP BY arm` can read without a
join.

Layout::

    <run_dir>/scores.parquet    one row per PropertyResult

This sits beside `rows.parquet` in the same run directory on purpose: a score is
only meaningful against the executions it judged, and `case_id`/`group_id` are
the join keys back to that table.

THE JOIN IS UNVERIFIED, AND NOT VERIFIABLE HERE. Neither file carries a run
identity, so nothing in this module can tell that a `scores.parquet` belongs to
the `rows.parquet` beside it. Every one of these reads clean and yields a
plausible wrong number:

* a `scores.parquet` copied in from another run — every score orphans, and the
  detection rate is 0/0 rather than an error;
* a single mistyped `case_id` — the denominator quietly loses a row;
* scores covering three of ten recorded cases — the denominator is three;
* worst, the *normal re-record workflow*: rewriting the execution table after
  scoring, flipping `kernel_is_broken`, leaves two individually valid files that
  together assert a rate nobody computed.

Verifying the join — scored `case_id`s a subset of recorded ones, and coverage
complete enough for the rate being claimed — is the driver's job, because only
the driver sees both tables in one place. Criterion `JOIN_IS_VERIFIED` in
`specs/features/0005-measurable-runs/acceptance.yaml` assigns it; note that
`DETECTION_IS_DERIVABLE` requires only that the number be *computable*, not that
it be *correct*, so it does not cover this.

DURABILITY: the file is staged as a sibling and published with `os.replace`,
matching `table.py`. Scores are re-derivable by re-scoring, which argues the
guarantee is unnecessary — but `elapsed_s` is not re-derivable: re-scoring
measures a different machine-minute and yields a different number. A truncated
Parquet footer also does not read as "partial", it reads as corrupt, so the
choice is between an atomic publish and a run whose scores must be recomputed
wholesale. The publish costs one rename.

Unlike the execution table there are no payload files to keep in step, so no
retire-first dance is needed: staging plus rename is the entire protocol, and a
failed write leaves the previous scores untouched.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from autokernel_pbt.props.table import _conform
from autokernel_pbt.props.verdict import PropertyResult, Verdict

SCORES_FILE = "scores.parquet"
# Staged inside run_dir so the final publish is a same-filesystem rename.
SCORES_TMP = f".{SCORES_FILE}.tmp"

# Flat and explicitly typed, for the same reason `table.SCHEMA` is: an explicit
# schema means `write([])` produces a table with real columns rather than a
# zero-column one, and it pins `tier` to int64 and `tolerance_free` to bool
# instead of whatever the first batch happened to imply.
SCHEMA = pa.schema(
    [
        ("arm", pa.string()),
        ("elapsed_s", pa.float64()),
        ("property_name", pa.string()),
        ("tier", pa.int64()),
        ("tolerance_free", pa.bool_()),
        ("verdict", pa.string()),
        ("detail", pa.string()),
        # Exactly one of these is set; the empty one is "" rather than null, so
        # the join key is a plain string comparison on both sides.
        ("case_id", pa.string()),
        ("group_id", pa.string()),
    ]
)


@dataclass(frozen=True)
class ArmScores:
    """Everything one oracle arm produced, plus what producing it cost.

    ONE `ArmScores` PER ARM PER WRITE. `read` reassembles by arm name, so two
    entries sharing a name would merge into one arm with a single `elapsed_s`,
    silently costing a cost-per-bug figure half the work that produced it.
    `write` rejects the duplicate rather than summing, because a summed total
    reads clean and a rejected write does not. A caller that scores an arm in
    chunks must therefore accumulate the results and the seconds itself and
    write once.

    `elapsed_s` is wall-clock seconds for the whole arm; it must be finite and
    non-negative, since it is a denominator and nothing else here can tell a
    placeholder from a measurement.
    """

    arm: str
    elapsed_s: float
    results: list[PropertyResult] = field(default_factory=list)


class ScoreTable:
    """Read/write the property verdicts for one run."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)

    @property
    def scores_path(self) -> Path:
        return self.run_dir / SCORES_FILE

    def write(self, arms: list[ArmScores]) -> None:
        """Persist every arm's results, replacing any scores already in `run_dir`.

        Every check runs, and every record is built, before the filesystem is
        touched: this module is offline scoring, so a rejected call costs a
        re-run of something free, while a half-replaced score file costs the
        previous one.
        """
        # `read` reassembles by arm name, so two ArmScores sharing one would
        # merge into a single arm with one of the two `elapsed_s` values
        # discarded — a cost-per-bug figure computed against half the work that
        # produced it, with nothing anywhere saying so.
        counts = Counter(arm.arm for arm in arms)
        duplicates = sorted(name for name, n in counts.items() if n > 1)
        if duplicates:
            msg = f"duplicate arm names in one write, which would merge on read: {duplicates}"
            raise ValueError(msg)

        records: list[dict[str, Any]] = []
        for arm in arms:
            # An arm that judged nothing still enters the denominator of every
            # per-arm rate and still charges its elapsed_s against zero verdicts.
            # It is far more likely a wiring bug than a real result.
            if not arm.results:
                msg = f"arm {arm.arm!r} has no results; it would be scored on nothing"
                raise ValueError(msg)
            # A nan or negative elapsed_s is not a cost figure, and a nan is
            # actively corrosive: it writes cleanly and then makes the file
            # permanently unreadable, because `nan != nan` trips the read-side
            # agreement check on the arm's second row and blames a hand-edit that
            # never happened. Both are caller bugs — a stopwatch that was never
            # started, a subtraction the wrong way round — so they raise here.
            if not math.isfinite(arm.elapsed_s) or arm.elapsed_s < 0:
                msg = (
                    f"arm {arm.arm!r} has elapsed_s={arm.elapsed_s!r}; it must be a finite, "
                    f"non-negative number of seconds to serve as a cost denominator"
                )
                raise ValueError(msg)
            records.extend(self._record(arm, result) for result in arm.results)

        table = pa.Table.from_pylist(records, schema=SCHEMA)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        staged = self.run_dir / SCORES_TMP
        try:
            pq.write_table(table, staged)
            # os.replace is atomic within a filesystem, and `staged` is a
            # sibling, so no reader ever sees a partial score file.
            os.replace(staged, self.scores_path)
        finally:
            staged.unlink(missing_ok=True)

    def _record(self, arm: ArmScores, result: PropertyResult) -> dict[str, Any]:
        if bool(result.case_id) == bool(result.group_id):
            # Neither key: the verdict is orphaned and can never be rejoined to
            # the execution that produced it, so it can never be attributed to a
            # kernel. Both keys: it joins twice, counting one verdict against two
            # different units of analysis.
            msg = (
                f"arm {arm.arm!r} result {result.property_name!r} must carry exactly one of "
                f"case_id/group_id, got case_id={result.case_id!r} "
                f"group_id={result.group_id!r}"
            )
            raise ValueError(msg)
        record = {
            "arm": arm.arm,
            # Deliberately not coerced with `float()`: the float64 column already
            # widens an int and reads it back as a float, while `float("1.0")`
            # would quietly accept a value whose type is a caller bug — which
            # `math.isfinite` above rejects instead.
            "elapsed_s": arm.elapsed_s,
            "property_name": result.property_name,
            "tier": result.tier,
            "tolerance_free": result.tolerance_free,
            # `Verdict` is a str-mixin enum with `__str__ = str.__str__`, so this
            # is the wire value on every supported Python. `.name` would store
            # "PASS" and not reconstruct.
            "verdict": str(result.verdict),
            "detail": result.detail,
            "case_id": result.case_id,
            "group_id": result.group_id,
        }
        # `pa.Table.from_pylist` presence-checks nothing: a missing key becomes a
        # fully-null column, an extra key is dropped, and only types raise. This
        # is the one place a wrong key set is catchable. Shared with `table.py`
        # rather than copied, so the two cannot drift.
        return _conform(record, SCHEMA)

    def read(self) -> list[ArmScores]:
        """Every persisted arm, in write order. `[]` if the run has no scores.

        A verdict value with no `Verdict` member raises rather than degrading to
        a string: it means a build whose verdict vocabulary has since changed,
        which every aggregation downstream would otherwise misclassify.

        The exactly-one-join-key invariant is re-checked here, not only at write.
        It is not the hole `_conform` cannot see — that one is invisible to any
        reader — it is an invariant the *values* carry, so a file that lost it
        (a null `case_id`, both keys set) is detectable and is refused. Without
        the re-check a null would surface as `PropertyResult(case_id=None)`, an
        orphan that joins to nothing and puts a `None` in a column declared
        `str`, where a downstream `.startswith` becomes an AttributeError.
        """
        if not self.scores_path.exists():
            return []
        table = pq.read_table(self.scores_path)
        self._require_columns(table.schema.names)

        grouped: dict[str, ArmScores] = {}
        for record in table.to_pylist():
            arm_name = record["arm"]
            elapsed = record["elapsed_s"]
            arm = grouped.get(arm_name)
            if arm is None:
                arm = ArmScores(arm=arm_name, elapsed_s=elapsed, results=[])
                grouped[arm_name] = arm
            elif arm.elapsed_s != elapsed:
                # `write` cannot emit this — one elapsed_s is broadcast to every
                # row of an arm, and the nan that would have made a row disagree
                # with itself is rejected there. Only a foreign or hand-edited
                # file reaches here, and taking the first of two would invent a
                # cost figure.
                msg = (
                    f"{self.scores_path} has rows for arm {arm_name!r} disagreeing "
                    f"elapsed_s: {arm.elapsed_s} and {elapsed}"
                )
                raise ValueError(msg)
            arm.results.append(self._result(record))
        return list(grouped.values())

    def _result(self, record: dict[str, Any]) -> PropertyResult:
        """Rebuild one `PropertyResult`, refusing a row that lost an invariant."""
        name = record["property_name"]
        if bool(record["case_id"]) == bool(record["group_id"]):
            msg = (
                f"{self.scores_path} has a row for property {name!r} carrying "
                f"case_id={record['case_id']!r} and group_id={record['group_id']!r}; "
                f"exactly one of case_id/group_id must be set, or the row cannot be "
                f"rejoined to the execution it judged"
            )
            raise ValueError(msg)
        # Parquet hands back a bare `str` for the verdict. It compares equal to
        # the enum member, so `==` hides the difference — but `is`, `match`, and
        # any identity dispatch silently never match.
        verdict = self._verdict(record["verdict"], name)
        try:
            return PropertyResult(
                property_name=name,
                tier=record["tier"],
                tolerance_free=record["tolerance_free"],
                verdict=verdict,
                detail=record["detail"],
                case_id=record["case_id"],
                group_id=record["group_id"],
            )
        except ValueError as exc:
            # `PropertyResult.__post_init__` rejects an unknown tier, but says
            # only that the tier is bad — not which file holds it. Every other
            # refusal on this read path names the file; this one must too.
            msg = f"{exc} (property {name!r} in {self.scores_path})"
            raise ValueError(msg) from exc

    def _require_columns(self, present: list[str]) -> None:
        """Reject a score file written by a build with a narrower schema.

        Without this the first missing column surfaces as a bare `KeyError` from
        inside the row loop, naming neither the run directory nor the reason.
        There is no recorded corpus, so old files are not supported — but the
        refusal has to say so.
        """
        available = set(present)
        missing = [name for name in SCHEMA.names if name not in available]
        if missing:
            msg = (
                f"{self.scores_path} was written by a build with a different schema; "
                f"missing columns: {missing}. There is no migration: re-score this run."
            )
            raise ValueError(msg)

    def _verdict(self, value: str, property_name: str) -> Verdict:
        try:
            return Verdict(value)
        except ValueError as exc:
            # A bare "'skipped' is not a valid Verdict" from deep inside a read
            # gives no clue which run directory is stale.
            msg = f"{exc} (property {property_name!r} in {self.scores_path})"
            raise ValueError(msg) from exc
