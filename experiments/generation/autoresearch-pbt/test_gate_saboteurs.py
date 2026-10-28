"""The gate's saboteur matrix, pinned one cell at a time.

Run from the repository root. The experiment is outside ``testpaths``, so the main
suite does not collect it:

    .venv/bin/python -m pytest experiments/generation/autoresearch-pbt -q

What is pinned is a *baseline*, not a target. Three of these kernels are grossly wrong
and the gate admits them, because its decisive arm is the declarative one and the
softmax bundle holds for softmax(beta * x) at every beta
(``docs/measurements/2026-10-01-softmax-temperature-blind-spot.md``). Completing the
bundle is a separate experiment; when it lands, these tests are meant to fail, and the
rows in ``saboteurs.py`` are re-recorded deliberately rather than loosened to pass.

E2 (feature 0014) landed the completed bundle beside the current one. Every test below that
reads a recorded row runs under both: the current rows as they were measured, and
``saboteurs.COMPLETED`` under the completed bundle.

One assertion per test, and each is the sole catcher of a break in the gate the other
two cannot see:

* the pairing: an arm going blind, or starting to catch, or catching through a
  different property;
* the decision: the gate deciding on an arm other than the declarative one, or
  admitting a kernel its decisive arm rejected;
* the judgement: a declarative PASS that is really an abstention. ``check_task``
  counts only FAIL groups, so an arm that judged nothing reads exactly like an arm
  that judged and approved -- the pairing alone is fail-open there.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pbt_gate
import pytest
from saboteurs import COMPLETED, SABOTEURS, first_failing_property

from autokernel_pbt.props.oracle import DeclarativeOracle
from autokernel_pbt.props.verdict import Verdict

pytestmark = pytest.mark.integration

BUNDLES = ("current", "completed")


def _recorded(name: str, bundle: str) -> tuple[Mapping[str, tuple[int, str]], bool]:
    """(expected per arm, admitted), as recorded for ``bundle``."""
    if bundle == "current":
        return SABOTEURS[name].expected, SABOTEURS[name].admitted
    return COMPLETED[name]


def _observed(name: str, bundle: str) -> dict[str, tuple[int, str]]:
    """Per arm, as the gate reports it: (groups failed, the property that failed first)."""
    saboteur = SABOTEURS[name]
    verdict = pbt_gate.check_task(saboteur.kernel, saboteur.task, bundle=bundle)
    return {arm.arm: (arm.groups_failed, first_failing_property(arm)) for arm in verdict.arms}


def test_every_saboteur_is_recorded_under_both_bundles():
    """A completed row for every saboteur, and none for a saboteur that no longer exists: a stale
    row would be read by no test, and a missing one would leave a kernel unpinned under the
    bundle E2 measures."""
    assert COMPLETED.keys() == SABOTEURS.keys()


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("name", sorted(SABOTEURS))
def test_each_saboteur_is_caught_by_exactly_its_paired_arms(name: str, bundle: str):
    """Equality over all four arms, never a subset check.

    A subset check ("the arms that should catch it do") cannot see an arm that starts
    catching, and the arm expected to start catching is the declarative one, the day
    the bundle is completed. The property name pins *attribution*, not just detection:
    the hybrid arm catches the temperature kernels through ``matches_reference`` and
    the positive control through ``rows_sum_to_one``, and only the property tells those
    apart -- the group counts are identical.
    """
    assert _observed(name, bundle) == _recorded(name, bundle)[0]


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("name", sorted(SABOTEURS))
def test_the_gate_admits_exactly_the_saboteurs_it_is_recorded_admitting(name: str, bundle: str):
    """The decision, which no arm's row carries.

    Every temperature kernel is caught by three arms, so the pairing above stays green
    if the gate is switched to decide on any of them -- yet that switch flips the
    headline. And the positive control is the one row the gate must *reject*: without
    it, a gate that admits everything would satisfy every row of this test.
    """
    saboteur = SABOTEURS[name]
    verdict = pbt_gate.check_task(saboteur.kernel, saboteur.task, bundle=bundle)
    assert verdict.passed is _recorded(name, bundle)[1]


@pytest.mark.parametrize(
    ("name", "bundle"),
    sorted(
        (n, b) for n in SABOTEURS for b in BUNDLES if _recorded(n, b)[0]["declarative"] == (0, "")
    ),
)
def test_a_declarative_pass_is_a_judgement_not_an_abstention(monkeypatch, name: str, bundle: str):
    """Every result the gate's own declarative arm produced is a PASS -- none abstained.

    Observed through the arm ``check_task`` actually built, not a second one built
    here: a copy would keep passing after the gate stopped using it. The hybrid arm
    holds the same object, so its declarative half is recorded too.

    Set equality, so an empty recording fails rather than passing vacuously.
    """
    seen = []
    original = DeclarativeOracle.evaluate

    def evaluate(self, rows):
        results = original(self, rows)
        seen.extend(results)
        return results

    monkeypatch.setattr(DeclarativeOracle, "evaluate", evaluate)
    saboteur = SABOTEURS[name]
    pbt_gate.check_task(saboteur.kernel, saboteur.task, bundle=bundle)

    not_passed = sorted(
        {(r.property_name, str(r.verdict), r.detail) for r in seen if r.verdict is not Verdict.PASS}
    )
    assert {r.verdict for r in seen} == {Verdict.PASS}, not_passed


def _raises(x):
    msg = "a kernel that never runs on the gate's shapes"
    raise RuntimeError(msg)


def _integer_zeros(x):
    return np.zeros(x.shape, dtype=np.int32)


@pytest.mark.parametrize(
    ("kernel", "reason"),
    [
        (_raises, r"produced no output on \d+ of \d+ cases"),
        (_integer_zeros, r"declarative arm abstained on every group"),
    ],
    ids=["raises_everywhere", "integer_zeros"],
)
@pytest.mark.parametrize("task", ["softmax", "layernorm"])
def test_a_candidate_the_gate_never_judged_is_refused(task: str, kernel, reason: str):
    """Every arm counts only FAIL groups, so a kernel with no output fails no group on any arm,
    which reads as a pass. The refusal is the only thing that rejects it."""
    verdict = pbt_gate.check_task(kernel, task)
    assert verdict.passed is False
    assert re.search(reason, verdict.refused), verdict.refused


@pytest.mark.parametrize(("task", "bundle"), [("layernorm", "completed"), ("softmax", "final")])
def test_a_bundle_the_task_does_not_have_is_refused(task: str, bundle: str):
    """Layernorm's completing law is not built, and a misspelled bundle names nothing. Either is a
    bad call, so it raises; falling back to the current bundle would report its verdicts under
    the completed bundle's name."""
    with pytest.raises(ValueError, match=f"no {bundle!r} bundle for task {task!r}"):
        pbt_gate.check_task(SABOTEURS["beta_1_correct"].kernel, task, bundle=bundle)
