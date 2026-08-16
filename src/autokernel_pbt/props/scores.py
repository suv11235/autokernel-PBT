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

THE JOIN IS CHECKABLE BUT NOT CHECKED HERE. Four ways two individually valid
files assert a rate nobody computed:

* a `scores.parquet` copied in from another run — and because case ids are a pure
  function of `(seed, index)`, one from a *correct-kernel* run joins perfectly
  onto a *broken-kernel* table and reports 0.0 detection;
* a single mistyped `case_id` — the denominator quietly loses a row;
* scores covering three of ten recorded cases — the denominator is three;
* the *normal re-record workflow*: rewriting the execution table after scoring,
  flipping `kernel_is_broken`, leaves two files that no longer describe one run.

`corpus_fingerprint` closes the first and the last: both tables carry the identity
of the corpus they are about (`table.new_corpus_fingerprint`), so a mismatched
pair is detectable rather than merely wrong. It cannot close the middle two —
those are statements about *coverage*, which needs both tables in view at once.

So verification remains the driver's job: `driver.read_run` pairs the two files
and refuses a mismatch, and `driver._verify_join` checks key validity and coverage
before the scores are ever written. Criterion `JOIN_IS_VERIFIED` in
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
        # `group_id` is ALWAYS set; `case_id` is set only on a case-scoped verdict and
        # is "" (never null) otherwise, so the join key is a plain string comparison on
        # both sides. See `_record` for why the coarser key is the mandatory one.
        ("case_id", pa.string()),
        ("group_id", pa.string()),
        # Which corpus these verdicts judged. Meaningless alone; the whole value is in
        # comparing it against the `rows.parquet` a reader intends to join to. See
        # `table.new_corpus_fingerprint`.
        ("corpus_fingerprint", pa.string()),
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

    def write(self, arms: list[ArmScores], *, corpus_fingerprint: str) -> None:
        """Persist every arm's results, replacing any scores already in `run_dir`.

        `corpus_fingerprint` is the identity of the execution table these verdicts
        judged, taken from `ExecutionTable.corpus_fingerprint()`. Keyword-only with no
        default on purpose: a default of `""` would mean every caller that forgot it
        wrote a scores file that pairs with *any* table, which is the hole this column
        exists to close.

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
            records.extend(
                self._record(arm, result, corpus_fingerprint) for result in arm.results
            )

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

    def _record(
        self, arm: ArmScores, result: PropertyResult, corpus_fingerprint: str
    ) -> dict[str, Any]:
        """One persisted row. `group_id` is mandatory; `case_id` refines it.

        THE UNIT OF ANALYSIS IS THE CASE GROUP, and this is where that stops being a
        docstring and becomes a key. The arms emit different numbers of results per
        group — the reference arm one per case, the declarative arm one per case per
        case-property plus one per group-property — so a rate computed per *result* is a
        weighted average whose weights belong to the arm rather than to the kernel.
        Measured on the ladder corpus against one broken kernel: 14 FAILs either way, but
        14/63 = 0.222 for the declarative arm against 14/18 = 0.778 for the reference
        arm, where the group rollup is 0.778 for both. The per-result number is not a
        different view of the same thing, it is wrong — and it is plausible, which is
        worse.

        Carrying `group_id` on every row makes the correct rollup a `GROUP BY group_id`
        needing no join to `rows.parquet` and no knowledge of which module wrote the
        file. The alternative considered and rejected was a `unit` column naming the
        intended aggregation: that is a producer instructing a reader, and it would still
        require the reader to perform a case -> group join it has no key for.

        `case_id` stays, as the finer key: it is what attributes a verdict to one
        execution, and what a per-case analysis (or a per-row debug) needs. A
        group-scoped verdict carries `""` for it rather than null, so both columns are
        plain string comparisons.
        """
        if not result.group_id:
            # Without a group id the verdict cannot be rolled up at the only unit at
            # which arms are comparable — and if it has no case id either, it is orphaned
            # outright and can never be attributed to a kernel at all.
            msg = (
                f"arm {arm.arm!r} result {result.property_name!r} carries no group_id "
                f"(case_id={result.case_id!r}); every score row must name the group it "
                f"judged, because the case group is the unit at which arms are comparable"
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
            "corpus_fingerprint": corpus_fingerprint,
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
        if not record["group_id"]:
            msg = (
                f"{self.scores_path} has a row for property {name!r} carrying "
                f"group_id={record['group_id']!r}; every row must name the group it "
                f"judged, or it cannot be rolled up at the unit where arms are comparable"
            )
            raise ValueError(msg)
        if record["case_id"] is None:
            # A null, specifically — "" is the legitimate value on a group-scoped
            # verdict. Rebuilt as `PropertyResult(case_id=None)` it would put a None in
            # a column declared `str`, where a downstream `.startswith` becomes an
            # AttributeError, and it would sort apart from "".
            msg = (
                f"{self.scores_path} has a row for property {name!r} with a null case_id; "
                f"a group-scoped verdict must carry '' rather than null, so both join "
                f"keys stay plain string comparisons"
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

    def corpus_fingerprint(self) -> str:
        """The corpus these scores judged, or `""` if the run has no scores.

        Mirrors `ExecutionTable.corpus_fingerprint`, including its refusal of a file
        whose rows disagree: `write` broadcasts one value to the whole file, so two
        distinct values mean scores from two runs concatenated, and picking one would
        certify half a file.
        """
        if not self.scores_path.exists():
            return ""
        table = pq.read_table(self.scores_path)
        self._require_columns(table.schema.names)
        distinct = sorted(set(table.column("corpus_fingerprint").to_pylist()))
        if not distinct:
            return ""
        if len(distinct) > 1:
            msg = (
                f"{self.scores_path} carries {len(distinct)} different corpus "
                f"fingerprints: {distinct}. A single write stamps one, so this file holds "
                f"scores from more than one run and belongs to no single execution table"
            )
            raise ValueError(msg)
        return distinct[0]

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
