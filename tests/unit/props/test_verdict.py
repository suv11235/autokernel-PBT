"""Verdict semantics tests."""

import json

import pytest

from autokernel_pbt.props.verdict import PropertyResult, Verdict, summarize


def _r(name: str, verdict: Verdict, tolerance_free: bool = True) -> PropertyResult:
    return PropertyResult(
        property_name=name, tier=1, tolerance_free=tolerance_free, verdict=verdict
    )


def test_all_pass_summarizes_to_pass():
    assert summarize([_r("a", Verdict.PASS), _r("b", Verdict.PASS)]) is Verdict.PASS


def test_any_fail_summarizes_to_fail():
    results = [_r("a", Verdict.PASS), _r("b", Verdict.FAIL), _r("c", Verdict.INCONCLUSIVE)]
    assert summarize(results) is Verdict.FAIL


def test_inconclusive_without_fail_summarizes_to_inconclusive():
    assert summarize([_r("a", Verdict.PASS), _r("b", Verdict.INCONCLUSIVE)]) is Verdict.INCONCLUSIVE


def test_empty_results_are_inconclusive_not_pass():
    # A property set that checked nothing has not established correctness.
    assert summarize([]) is Verdict.INCONCLUSIVE


def test_result_records_attribution_fields():
    """PropertyResult records all attribution fields; defaults apply when not set."""
    result = PropertyResult(
        property_name="softmax_rows_sum_to_one",
        tier=1,
        tolerance_free=False,
        verdict=Verdict.FAIL
    )
    assert result.property_name == "softmax_rows_sum_to_one"
    assert result.tier == 1
    assert result.tolerance_free is False
    assert result.verdict is Verdict.FAIL
    # Defaults: empty strings for optional fields
    assert result.detail == ""
    assert result.case_id == ""
    assert result.group_id == ""
    # Frozen: no mutations allowed
    with pytest.raises(AttributeError):
        result.property_name = "other"


def test_fail_alone_returns_fail():
    """FAIL by itself should return FAIL."""
    assert summarize([_r("a", Verdict.FAIL)]) is Verdict.FAIL


def test_inconclusive_alone_returns_inconclusive():
    """INCONCLUSIVE by itself (without FAIL) should return INCONCLUSIVE."""
    assert summarize([_r("a", Verdict.INCONCLUSIVE)]) is Verdict.INCONCLUSIVE


def test_pass_alone_returns_pass():
    """A single PASS should return PASS."""
    assert summarize([_r("a", Verdict.PASS)]) is Verdict.PASS


def test_fail_and_inconclusive_fail_dominates():
    """When both FAIL and INCONCLUSIVE are present, FAIL should win."""
    results = [_r("a", Verdict.INCONCLUSIVE), _r("b", Verdict.FAIL)]
    assert summarize(results) is Verdict.FAIL


def test_summarize_accepts_generator():
    """summarize should accept a generator without exhausting it prematurely."""
    def gen():
        yield _r("a", Verdict.PASS)
        yield _r("b", Verdict.PASS)

    result = summarize(gen())
    assert result is Verdict.PASS


def test_summarize_generator_with_fail():
    """Generator with a FAIL should correctly identify it."""
    def gen():
        yield _r("a", Verdict.PASS)
        yield _r("b", Verdict.FAIL)
        yield _r("c", Verdict.INCONCLUSIVE)

    result = summarize(gen())
    assert result is Verdict.FAIL


def test_summarize_generator_with_inconclusive_no_fail():
    """Generator with INCONCLUSIVE but no FAIL should return INCONCLUSIVE."""
    def gen():
        yield _r("a", Verdict.PASS)
        yield _r("b", Verdict.INCONCLUSIVE)

    result = summarize(gen())
    assert result is Verdict.INCONCLUSIVE


def test_verdict_str_mixin_equality():
    """Verdict equals its wire value string for defensive Python 3.10+ portability."""
    assert str(Verdict.PASS) == "pass"
    assert Verdict.PASS == "pass"
    assert format(Verdict.FAIL) == "fail"
    assert f"{Verdict.INCONCLUSIVE}" == "inconclusive"


def test_verdict_json_serializable():
    """Verdict must serialize to JSON as its wire value, not raise TypeError."""
    data = {"result": Verdict.PASS}
    serialized = json.dumps(data)
    assert serialized == '{"result": "pass"}'


def test_property_result_accepts_tier_1():
    """Tier 1 should be accepted (portable/semantic)."""
    result = PropertyResult(
        property_name="test",
        tier=1,
        tolerance_free=True,
        verdict=Verdict.PASS
    )
    assert result.tier == 1


def test_property_result_accepts_tier_2():
    """Tier 2 should be accepted (backend-specific)."""
    result = PropertyResult(
        property_name="test",
        tier=2,
        tolerance_free=True,
        verdict=Verdict.PASS
    )
    assert result.tier == 2


def test_property_result_rejects_invalid_tier():
    """Tier must be 1 or 2; others should be rejected."""
    with pytest.raises(ValueError, match="tier must be in"):
        PropertyResult(
            property_name="test",
            tier=3,
            tolerance_free=True,
            verdict=Verdict.PASS
        )


def test_property_result_rejects_tier_zero():
    """Tier 0 should be rejected."""
    with pytest.raises(ValueError, match="tier must be in"):
        PropertyResult(
            property_name="test",
            tier=0,
            tolerance_free=True,
            verdict=Verdict.PASS
        )


def test_property_result_case_id_defaults_empty():
    """case_id defaults to empty string."""
    result = PropertyResult(
        property_name="test",
        tier=1,
        tolerance_free=True,
        verdict=Verdict.PASS
    )
    assert result.case_id == ""


def test_property_result_group_id_defaults_empty():
    """group_id defaults to empty string."""
    result = PropertyResult(
        property_name="test",
        tier=1,
        tolerance_free=True,
        verdict=Verdict.PASS
    )
    assert result.group_id == ""


def test_property_result_can_carry_case_id():
    """A case property can set case_id; group_id remains empty."""
    result = PropertyResult(
        property_name="check_range",
        tier=1,
        tolerance_free=True,
        verdict=Verdict.PASS,
        case_id="case_0"
    )
    assert result.case_id == "case_0"
    assert result.group_id == ""


def test_property_result_can_carry_group_id():
    """A group property can set group_id; case_id remains empty."""
    result = PropertyResult(
        property_name="check_all_finite",
        tier=2,
        tolerance_free=False,
        verdict=Verdict.FAIL,
        group_id="group_5"
    )
    assert result.case_id == ""
    assert result.group_id == "group_5"
