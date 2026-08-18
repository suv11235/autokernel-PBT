"""The report carries its caveats with the numbers."""

from __future__ import annotations

from autokernel_pbt.metrics.rates import ArmRates
from autokernel_pbt.metrics.report import render


def _rates(detection=0.5, tf=0.0):
    return ArmRates(arm="declarative", groups_scored=9, groups_failed=7,
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
