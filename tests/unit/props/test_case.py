"""Case and CaseGroup tests."""

import json

import numpy as np
import pytest

from autokernel_pbt.props.case import BASE_RELATION, Case, CaseGroup


def _case(case_id: str, relation: str = BASE_RELATION) -> Case:
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
    group = CaseGroup(
        group_id="g0", cases=(_case("c0"), _case("c1", "shift_rows"))
    )
    assert group.base.case_id == "c0"


def test_group_finds_case_by_relation():
    group = CaseGroup(
        group_id="g0", cases=(_case("c0"), _case("c1", "shift_rows"))
    )
    assert group.by_relation("shift_rows").case_id == "c1"


def test_group_by_relation_returns_none_when_absent():
    group = CaseGroup(group_id="g0", cases=(_case("c0"),))
    assert group.by_relation("shift_rows") is None


def test_group_requires_exactly_one_base():
    with pytest.raises(ValueError, match="exactly one base"):
        CaseGroup(group_id="g0", cases=(_case("c0", "shift_rows"),))


def test_group_rejects_mismatched_group_id():
    other = Case(
        case_id="c1",
        group_id="OTHER",
        relation="shift_rows",
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.zeros((2, 3), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="group_id mismatch"):
        CaseGroup(group_id="g0", cases=(_case("c0"), other))


def test_group_rejects_duplicate_relations():
    with pytest.raises(ValueError, match="duplicate relations"):
        CaseGroup(
            group_id="g0",
            cases=(
                _case("c0"),
                _case("c1", "shift_rows"),
                _case("c2", "shift_rows"),
            ),
        )


def test_group_rejects_duplicate_case_ids():
    other = Case(
        case_id="c0",
        group_id="g0",
        relation="shift_rows",
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.ones((2, 3), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="duplicate case_ids"):
        CaseGroup(group_id="g0", cases=(_case("c0"), other))


def test_case_normalizes_list_shape_to_tuple():
    case = Case(
        case_id="c0",
        group_id="g0",
        relation=BASE_RELATION,
        task_id="softmax",
        dtype="float32",
        shape=[2, 3],
        tensors={"x": np.zeros((2, 3), dtype=np.float32)},
    )
    assert isinstance(case.shape, tuple)
    assert case.shape == (2, 3)


def test_case_normalizes_np_int64_shape_to_python_int():
    np_ints = tuple(np.array([2, 3], dtype=np.int64))
    case = Case(
        case_id="c0",
        group_id="g0",
        relation=BASE_RELATION,
        task_id="softmax",
        dtype="float32",
        shape=np_ints,
        tensors={"x": np.zeros((2, 3), dtype=np.float32)},
    )
    assert isinstance(case.shape, tuple)
    assert case.shape == (2, 3)
    assert all(isinstance(s, int) for s in case.shape)
    # Verify metadata is JSON serializable
    metadata = case.metadata()
    json_str = json.dumps(metadata)
    assert json_str is not None


def test_group_normalizes_list_cases_to_tuple():
    cases_list = [_case("c0"), _case("c1", "shift_rows")]
    group = CaseGroup(group_id="g0", cases=cases_list)
    assert isinstance(group.cases, tuple)
    assert len(group.cases) == 2
    assert group.cases[0].case_id == "c0"


def test_case_equality_ignores_tensor_values():
    case1 = Case(
        case_id="c0",
        group_id="g0",
        relation=BASE_RELATION,
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.zeros((2, 3), dtype=np.float32)},
    )
    case2 = Case(
        case_id="c0",
        group_id="g0",
        relation=BASE_RELATION,
        task_id="softmax",
        dtype="float32",
        shape=(2, 3),
        tensors={"x": np.ones((2, 3), dtype=np.float32)},
    )
    assert case1 == case2
