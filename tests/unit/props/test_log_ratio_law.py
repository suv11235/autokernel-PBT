"""The log-ratio law, softmax's completing law (feature 0014).

Each FAIL path is pinned to its own message and to a kernel that only it can catch. Deleting
any one of the law's four checks fails exactly the tests named after it: a top-1 kernel passes
the spread check (one judged entry per row has no spread), so only the underflow check can
catch it, and a negative output where the law predicts underflow passes the underflow check,
so only positivity can catch it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OUTPUT_NAME, PRIMARY_INPUT, ExecutionResult, Status
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.case import BASE_RELATION, Case
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.properties import (
    CASE_PROPERTY_REGISTRY,
    OutputsAreFinite,
    RowsHaveConstantLogRatio,
)
from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference
from autokernel_pbt.props.verdict import TIER_PORTABLE, PropertyResult, Verdict

Kernel = Callable[[np.ndarray], np.ndarray]

TINY32 = float(np.finfo(np.float32).tiny)
SPREAD = "log-ratio test ratio"
MASKED = "masked input"
UNDERFLOW = "below the smallest normal"
NEGATIVE = "negative output"


def _case(x: np.ndarray, *, name: str = PRIMARY_INPUT) -> Case:
    return Case(
        case_id="c0",
        group_id="g0",
        relation=BASE_RELATION,
        task_id="softmax",
        dtype=str(x.dtype),
        shape=x.shape,
        tensors={name: x},
    )


def _row(x: np.ndarray, y: np.ndarray | None, status: Status = Status.OK) -> ExecutionResult:
    outputs = {} if y is None else {OUTPUT_NAME: y}
    return ExecutionResult(case=_case(x), outputs=outputs, status=status)


def _check(x: np.ndarray, y: np.ndarray) -> PropertyResult:
    return RowsHaveConstantLogRatio().check(_row(x, y))


def stock(x: np.ndarray) -> np.ndarray:
    """softmax as a float32 kernel writes it: max-subtracted, exp, divided by the row sum."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _beta(beta: float) -> Kernel:
    def kernel(x: np.ndarray) -> np.ndarray:
        return stock(np.multiply(x, np.float32(beta), dtype=np.float32))

    return kernel


def _fill(x: np.ndarray) -> np.ndarray:
    return np.full(x.shape, 1.0 / x.shape[-1], dtype=x.dtype)


def _ladder(kernel: Kernel) -> list[ExecutionResult]:
    """Every case of the softmax ladder, base and shift partner, as the gate draws them."""
    backend = NumpyBackend()
    groups = Generator(SOFTMAX.domain, 42).generate(len(SOFTMAX.domain.shapes))
    return [backend.run(kernel, case) for group in groups for case in group.cases]


def _wide(sigma: float, shape: tuple[int, int] = (16, 1024)) -> np.ndarray:
    return (
        np.random.default_rng([20261014, int(sigma)]).normal(0.0, sigma, shape).astype(np.float32)
    )


# --------------------------------------------------------------------------- #
# Declared metadata
# --------------------------------------------------------------------------- #


def test_the_law_declares_its_scope_and_its_deferral():
    law = RowsHaveConstantLogRatio()
    assert law.name == "rows_have_constant_log_ratio"
    assert law.tier == TIER_PORTABLE
    # Not tolerance-free: it carries c = 30, and the headline count must not include it.
    assert law.tolerance_free is False
    assert law.defers_nonfinite_to == OutputsAreFinite.name
    assert CASE_PROPERTY_REGISTRY[law.name] is RowsHaveConstantLogRatio


# --------------------------------------------------------------------------- #
# What it admits, and what it rejects
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kernel", [softmax_reference, stock], ids=["reference", "stock"])
def test_a_correct_softmax_passes_on_every_ladder_case_and_on_wide_rows(kernel: Kernel):
    """The ladder is the gate's own domain, N(0, 1) plus overflow-scale shifts. The wide rows are
    not: at sigma 64 a row spans hundreds, so a flat bound runs out, and float32 underflows to
    zero and to subnormals, which check 4 must accept where the law predicts them."""
    law = RowsHaveConstantLogRatio()
    for row in _ladder(kernel):
        result = law.check(row)
        assert result.verdict is Verdict.PASS, (row.case.case_id, result.detail)

    for sigma in (4.0, 16.0, 64.0):
        x = _wide(sigma)
        y = kernel(x)
        result = _check(x, y)
        assert result.verdict is Verdict.PASS, (sigma, result.detail)
    # Non-vacuous: the widest rows really do reach both kinds of underflow.
    y = kernel(_wide(64.0))
    assert np.any(y == 0) and np.any((y > 0) & (y < TINY32))


@pytest.mark.parametrize(
    "kernel",
    [_beta(2.0), _beta(0.0), _beta(-1.0), _fill],
    ids=["softmax(2x)", "softmax(0x)", "softmax(-x)", "constant 1/n fill"],
)
def test_the_temperature_family_and_the_constant_fill_fail_wherever_they_differ(kernel: Kernel):
    """Every member admitted by the current bundle (Proposition 1(a)). On a one-column case each
    of them IS softmax, so there the law must pass; everywhere else it must fail."""
    law = RowsHaveConstantLogRatio()
    failed = 0
    for row in _ladder(kernel):
        result = law.check(row)
        x, y = row.case.tensors[PRIMARY_INPUT], row.outputs[OUTPUT_NAME]
        if np.array_equal(y, softmax_reference(x)):
            assert result.verdict is Verdict.PASS, (row.case.case_id, result.detail)
        else:
            assert result.verdict is Verdict.FAIL, (row.case.case_id, result.detail)
            assert result.detail.startswith(SPREAD), result.detail
            failed += 1
    assert failed >= 14  # seven multi-column rungs, base and shift partner


@pytest.mark.parametrize(
    "symmetry",
    [lambda y: np.roll(y, 1, axis=0), lambda y: y[:, ::-1]],
    ids=["rows rolled by one", "columns reversed"],
)
def test_output_permutations_fail_wherever_they_differ(symmetry):
    """P softmax(x) Q, admitted by the current bundle. Rolling is the identity on a single row,
    and reversing on a single column, so those cases are softmax and must pass."""
    law = RowsHaveConstantLogRatio()
    failed = 0
    for row in _ladder(lambda x: np.ascontiguousarray(symmetry(softmax_reference(x)))):
        result = law.check(row)
        x, y = row.case.tensors[PRIMARY_INPUT], row.outputs[OUTPUT_NAME]
        if np.array_equal(y, softmax_reference(x)):
            assert result.verdict is Verdict.PASS, (row.case.case_id, result.detail)
        else:
            assert result.verdict is Verdict.FAIL, (row.case.case_id, result.detail)
            assert result.detail.startswith(SPREAD), result.detail
            failed += 1
    assert failed >= 12


@pytest.mark.parametrize("row_range", [2.0, 40.0])
def test_the_bound_is_thirty_eps_per_unit_of_row_range(row_range: float):
    """A spread of t (1 + range) eps passes at t = 26 and fails at t = 34, at a narrow and a wide
    range. A flat bound (no range) fails the t = 26 case at range 40; a bound of range alone
    (no 1 +) fails it at range 2. Float64 throughout, so the only spread is the one built in;
    the remaining rounding moves the ratio by well under 1."""
    eps = float(np.finfo(np.float64).eps)
    x = np.linspace(-row_range / 2, row_range / 2, 9).reshape(1, 9)
    for t, expected in ((26.0, Verdict.PASS), (34.0, Verdict.FAIL)):
        bump = np.zeros_like(x)
        bump[0, 3] = t * (1.0 + row_range) * eps  # spread = max d - min d = this bump
        y = np.exp(x + bump)
        result = _check(x, y)
        assert result.verdict is expected, (t, result.detail)
    assert result.detail.startswith(f"{SPREAD} 3") and ">= 30" in result.detail, result.detail


def test_a_large_negative_mask_does_not_loosen_the_bound():
    """An additive -1e9 mask is a finite input. Over the whole row the range would be 1e9, and
    softmax(2x) would pass at a ratio near 0.04; over judged entries it fails as it does unmasked."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, (8, 33)).astype(np.float32)
    x[rng.random(x.shape) < 0.25] = np.float32(-1e9)
    x[:, 0] = rng.normal(0.0, 1.0, 8).astype(np.float32)  # every row keeps a live entry
    assert _check(x, stock(x)).verdict is Verdict.PASS
    wrong = _check(x, _beta(2.0)(x))
    assert wrong.verdict is Verdict.FAIL
    assert wrong.detail.startswith(SPREAD), wrong.detail


# --------------------------------------------------------------------------- #
# Masked, underflowed and negative entries
# --------------------------------------------------------------------------- #


def _masked_input() -> np.ndarray:
    x = np.random.default_rng(3).normal(0.0, 1.0, (6, 17)).astype(np.float32)
    x[:, 1::3] = -np.inf
    return x


def test_a_masked_entry_must_be_exactly_zero():
    """exp(-inf + k) is 0 for every finite k, so any mass at a masked score is a leak, whatever
    its size. A correct masked softmax puts exactly 0 there and passes."""
    x = _masked_input()
    y = stock(x)
    assert np.all(y[np.isneginf(x)] == 0)
    assert _check(x, y).verdict is Verdict.PASS

    leaked = y.copy()
    leaked[2, 4] = np.float32(1e-30)  # x[2, 4] is masked
    result = _check(x, leaked)
    assert result.verdict is Verdict.FAIL
    assert result.detail.startswith(f"{MASKED} (x = -inf) has output 1e-30 at row 2, column 4")

    filled = _check(x, _fill(x))
    assert filled.verdict is Verdict.FAIL
    assert filled.detail.startswith(MASKED), filled.detail


def _top1(x: np.ndarray) -> np.ndarray:
    """All mass on each row's largest score: rows sum to one, values in [0, 1], shift invariant,
    and one judged entry per row, so no spread at all."""
    y = np.zeros_like(x)
    y[np.arange(x.shape[0]), np.argmax(x, axis=-1)] = 1.0
    return y


def test_an_output_below_the_smallest_normal_passes_only_where_the_law_predicts_it():
    # A correct kernel whose rows reach exact zeros and subnormals: both are predicted.
    x = np.array([[0.0, -1.0, -88.0, -95.0, -200.0, -np.inf, 3.0]], dtype=np.float32)
    y = stock(x)
    assert y[0, 4] == 0 and 0 < y[0, 3] < TINY32
    assert _check(x, y).verdict is Verdict.PASS
    # So is a device that flushes subnormals to zero.
    flushed = np.where(y < TINY32, np.float32(0.0), y)
    assert _check(x, flushed).verdict is Verdict.PASS

    # An entry the law predicts 1e-4 (in log) above the smallest normal, rounded below it and
    # flushed. Its budget is its own distance from the row's maximum, about 87, so it passes; a
    # budget of the judged range (here 0) would fail it at a ratio near 840.
    edge = np.array([[0.0, np.log(TINY32) + 1e-4]], dtype=np.float32)
    assert _check(edge, np.array([[1.0, 0.0]], dtype=np.float32)).verdict is Verdict.PASS

    # A zero where the law predicts exp(-4) of the row's largest output.
    dropped = y.copy()
    dropped[0, 1] = 0.0
    result = _check(x, dropped)
    assert result.verdict is Verdict.FAIL
    assert result.detail.startswith(f"output 0 at row 0, column 1 is {UNDERFLOW}"), result.detail

    # The top-1 kernel: the spread check alone admits it.
    x = np.random.default_rng(5).normal(0.0, 1.0, (8, 16)).astype(np.float32)
    top1 = _check(x, _top1(x))
    assert top1.verdict is Verdict.FAIL
    assert UNDERFLOW in top1.detail, top1.detail


def test_a_negative_output_fails():
    """Proposition 1(b) makes every output positive. The entry here is one the law predicts to
    underflow, so the underflow check would accept it; positivity is the only check that sees it."""
    x = np.array([[0.0, -200.0, 1.0]], dtype=np.float32)
    y = stock(x)
    y[0, 1] = np.float32(-1e-45)
    result = _check(x, y)
    assert result.verdict is Verdict.FAIL
    assert result.detail.startswith(f"{NEGATIVE} -1.4e-45 at row 0, column 1"), result.detail


def test_a_fully_masked_row_is_not_judged():
    """Softmax of a fully masked row is 0/0. Kernels return NaN, zeros or 1/n there, and which one
    is right is the contract's call, not this law's."""
    x = np.array([[0.0, 1.0, 2.0], [-np.inf, -np.inf, -np.inf]], dtype=np.float32)
    for fully_masked in (np.zeros(3), np.full(3, 1 / 3)):
        y = np.vstack([stock(x[:1]), fully_masked.astype(np.float32)[None, :]])
        result = _check(x, y)
        assert result.verdict is Verdict.PASS, result.detail

    nothing = np.full((2, 3), -np.inf, dtype=np.float32)
    result = _check(nothing, np.zeros((2, 3), dtype=np.float32))
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.detail.startswith("no row has an unmasked output"), result.detail


# --------------------------------------------------------------------------- #
# Data it cannot judge
# --------------------------------------------------------------------------- #

_X = np.array([[0.0, 1.0, 2.0], [3.0, 3.0, 1.0]], dtype=np.float32)


def _unjudgeable() -> list[tuple[str, ExecutionResult, str]]:
    complex_x = _X.astype(np.complex64)
    nan_x, inf_x = _X.copy(), _X.copy()
    nan_x[0, 1], inf_x[1, 0] = np.nan, np.inf
    nan_y = stock(_X)
    nan_y[0, 0] = np.nan
    return [
        ("failed status", _row(_X, stock(_X), Status.LAUNCH_ERROR), "status="),
        ("missing output", _row(_X, None), "missing output"),
        ("empty output", _row(_X[:, :0], _X[:, :0]), "empty output"),
        ("non-finite output", _row(_X, nan_y), "non-finite output"),
        ("integer output", _row(_X, np.ones_like(_X, dtype=np.int32)), "exact dtype"),
        (
            "no primary input",
            ExecutionResult(case=_case(_X, name="z"), outputs={OUTPUT_NAME: stock(_X)}),
            "no primary input",
        ),
        ("misshapen output", _row(_X, stock(_X)[:, :2]), "no primary input"),
        ("complex input", _row(complex_x, stock(_X)), "not real-valued"),
        ("NaN input", _row(nan_x, stock(_X)), "input contains NaN or +inf"),
        ("+inf input", _row(inf_x, stock(_X)), "input contains NaN or +inf"),
    ]


@pytest.mark.parametrize(
    ("label", "row", "detail"), _unjudgeable(), ids=[c[0] for c in _unjudgeable()]
)
def test_unjudgeable_data_is_inconclusive_and_never_raises(label, row, detail):
    result = RowsHaveConstantLogRatio().check(row)
    assert result.verdict is Verdict.INCONCLUSIVE, (label, result.detail)
    assert detail in result.detail, (label, result.detail)
    assert result.case_id == row.case.case_id and not result.group_id
