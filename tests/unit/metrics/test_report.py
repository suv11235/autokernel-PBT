"""The report carries its caveats with the numbers."""

from __future__ import annotations

import pytest

from autokernel_pbt.metrics.rates import ArmRates, rates_from_run
from autokernel_pbt.metrics.report import render


def _rates(detection=0.5, tf=0.0, arm="declarative"):
    return ArmRates(arm=arm, groups_scored=9, groups_failed=7,
                    groups_inconclusive=0, detection_rate=detection,
                    tolerance_free_detection_rate=tf, cases_to_first_failure=0)


def test_the_report_states_the_ladder_deflation():
    """The criterion THE_REPORT_STATES_THE_DEFLATION.

    A reader who sees only the table takes it at face value. The degenerate rungs
    make every absolute rate understate by a measured constant, so the caveat travels
    WITH the numbers rather than in a document nobody opens.
    """
    text = render({}, backend="numpy")
    assert "deflat" in text.lower()
    assert "0.778" in text or "7/9" in text


def test_the_report_labels_the_class_as_intended():
    # The class is established by the authoring prompt and verified by nothing.
    assert "intended" in render({}, backend="numpy").lower()


def test_the_report_states_the_corpus_size_caveat():
    assert "one mutant per class" in render({}, backend="numpy").lower()


def test_an_empty_report_still_renders_the_caveats():
    # A report with no rows is still read, and the header is what it exists for.
    text = render({}, backend="numpy")
    assert "No runs scored" in text
    assert "deflat" in text.lower()


def test_rows_render_per_mutant_and_arm():
    text = render({"softmax_indexing": {"declarative": _rates(0.333, 0.333)}}, backend="triton")
    assert "softmax_indexing" in text
    assert "0.333" in text
    assert "triton" in text


def _cells(text: str, mutant: str) -> dict[str, str]:
    """One table row as column name -> cell, checking the table is well-formed.

    A Markdown table whose separator has fewer cells than its header does not render
    as a table at all, so the cell counts are part of what is read here.
    """
    lines = text.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("| mutant"))
    names = [c.strip() for c in lines[at].strip("|").split("|")]
    separator = [c.strip() for c in lines[at + 1].strip("|").split("|")]
    row = next(line for line in lines if line.startswith(f"| {mutant} |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert separator == ["---"] * len(names)
    assert len(cells) == len(names)
    return dict(zip(names, cells))


def test_each_arm_column_carries_that_arms_detection_rate():
    """Every number in a different place, so a cell read from the wrong field shows.

    The row test above uses one arm whose detection and tolerance-free rates are
    both 0.333, so a report printing the tolerance-free rate in the detection column
    passed it. Here no two rendered values coincide.
    """
    rates = {"softmax_indexing": {
        "declarative": _rates(0.778, 0.556, arm="declarative"),
        "reference": _rates(0.667, 0.0, arm="reference"),
    }}
    assert _cells(render(rates, backend="numpy"), "softmax_indexing") == {
        "mutant (intended class)": "softmax_indexing",
        "declarative": "0.778",
        "reference": "0.667",
        "tolerance-free": "0.556",
    }


def test_an_unscored_run_is_refused_rather_than_rendered(tmp_path):
    """A run with an execution table and no scores has no rates to report.

    ``read_run`` deliberately treats "recorded, not yet scored" as a valid state, so
    ``rates_from_run`` answers ``{}`` for it -- and the report once rendered that as a
    row of empty cells under a header whose separator was one cell short, which is
    not a Markdown table at all. Refused rather than rendered as "not scored": the
    report is the published artifact, a row with no number in it is not a rate, and
    scoring is offline, so the fix costs a re-run of nothing.
    """
    from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
    from autokernel_pbt.props.generator import Generator
    from autokernel_pbt.props.table import ExecutionTable
    from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference

    groups = Generator(SOFTMAX.domain, seed=42).generate(len(SOFTMAX.domain.shapes))
    rows = [NumpyBackend().run(softmax_reference, c) for g in groups for c in g.cases]
    ExecutionTable(tmp_path).write(rows)
    unscored = rates_from_run(tmp_path)
    assert unscored == {}
    with pytest.raises(
        ValueError, match=r"mutant 'softmax_half' has no scored arm; its run has not been scored"
    ):
        render({"softmax_half": unscored}, backend="numpy")


def test_one_unscored_mutant_refuses_the_whole_report():
    # Next to a scored mutant the arm columns exist, so the table stays well-formed --
    # but the unscored row would read as an arm that was not run, which is a
    # different and false statement.
    rates = {"softmax_indexing": {"declarative": _rates()}, "softmax_half": {}}
    with pytest.raises(ValueError, match=r"mutant 'softmax_half' has no scored arm; "):
        render(rates, backend="numpy")
