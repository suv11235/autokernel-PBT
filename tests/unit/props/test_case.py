"""Case and CaseGroup tests."""

import numpy as np
import pytest

from autokernel_pbt.props.case import Case, CaseGroup


def _case(case_id: str, relation: str = "base") -> Case:
    return Case(
        case_id=case_id,
        group_id="g0",
        relation=relation,
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.zeros((2, 3), dtype=np.float32)},
    )


def test_group_exposes_base_case():
    group = CaseGroup(group_id="g0", cases=(_case("c0"), _case("c1", "shift")))
    assert group.base.case_id == "c0"


def test_group_finds_case_by_relation():
    group = CaseGroup(group_id="g0", cases=(_case("c0"), _case("c1", "shift")))
    assert group.by_relation("shift").case_id == "c1"


def test_group_by_relation_returns_none_when_absent():
    group = CaseGroup(group_id="g0", cases=(_case("c0"),))
    assert group.by_relation("shift") is None


def test_group_requires_exactly_one_base():
    with pytest.raises(ValueError, match="exactly one base"):
        CaseGroup(group_id="g0", cases=(_case("c0", "shift"),))


def test_group_rejects_mismatched_group_id():
    other = Case(
        case_id="c1",
        group_id="OTHER",
        relation="shift",
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.zeros((2, 3), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="group_id mismatch"):
        CaseGroup(group_id="g0", cases=(_case("c0"), other))


def test_case_normalizes_list_shape_to_tuple():
    case = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="softmax",
        dtype="float32",
        shape=[2, 3],
        tensors={"x": np.zeros((2, 3), dtype=np.float32)},
    )
    assert isinstance(case.shape, tuple)
    assert case.shape == (2, 3)


def test_group_normalizes_list_cases_to_tuple():
    cases_list = [_case("c0"), _case("c1", "shift")]
    group = CaseGroup(group_id="g0", cases=cases_list)
    assert isinstance(group.cases, tuple)
    assert len(group.cases) == 2
    assert group.cases[0].case_id == "c0"


def test_case_equality_ignores_tensor_values():
    case1 = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.zeros((2, 3), dtype=np.float32)},
    )
    case2 = Case(
        case_id="c0",
        group_id="g0",
        relation="base",
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.ones((2, 3), dtype=np.float32)},
    )
    assert case1 == case2
