"""The affine-tie law, layernorm's completing law (feature 0015).

Each FAIL path is pinned to its own message and to a kernel only it can catch. Sorted rows keep a
positive slope on every row of some cases, so only the tie catches those. -LN's tie is exact, so
only the slope catches it, and the same holds for a flat output, which has no residual.
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
    RowsHaveAffineTie,
)
from autokernel_pbt.props.tasks import LAYERNORM, LAYERNORM_EPS, layernorm_reference
from autokernel_pbt.props.verdict import TIER_PORTABLE, PropertyResult, Verdict

Kernel = Callable[[np.ndarray], np.ndarray]

TIE = "affine-tie test ratio"
SLOPE = "slope "
EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)


def _case(x: np.ndarray, *, name: str = PRIMARY_INPUT) -> Case:
    return Case(
        case_id="c0",
        group_id="g0",
        relation=BASE_RELATION,
        task_id="layernorm",
        dtype=str(x.dtype),
        shape=x.shape,
        tensors={name: x},
    )


def _row(x: np.ndarray, y: np.ndarray | None, status: Status = Status.OK) -> ExecutionResult:
    outputs = {} if y is None else {OUTPUT_NAME: y}
    return ExecutionResult(case=_case(x), outputs=outputs, status=status)


def _check(x: np.ndarray, y: np.ndarray) -> PropertyResult:
    return RowsHaveAffineTie().check(_row(x, y))


def stock(x: np.ndarray) -> np.ndarray:
    """Layernorm as a float32 kernel writes it: centered, then divided by the root."""
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(LAYERNORM_EPS))


def _ladder(kernel: Kernel) -> list[ExecutionResult]:
    """Every case of the layernorm ladder, base and shift partner, as the gate draws them."""
    backend = NumpyBackend()
    groups = Generator(LAYERNORM.domain, 42).generate(len(LAYERNORM.domain.shapes))
    return [backend.run(kernel, case) for group in groups for case in group.cases]


def _normal(mean: float, sigma: float, shape: tuple[int, int] = (16, 1024)) -> np.ndarray:
    rng = np.random.default_rng([20261102, int(abs(mean)), int(sigma * 1000)])
    return (mean + rng.normal(0.0, sigma, shape)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Declared metadata
# --------------------------------------------------------------------------- #


def test_the_law_declares_its_scope_and_its_deferral():
    law = RowsHaveAffineTie()
    assert law.name == "rows_have_affine_tie"
    assert law.tier == TIER_PORTABLE
    # Not tolerance-free: it carries c = 30, and the headline count must not include it.
    assert law.tolerance_free is False
    assert law.defers_nonfinite_to == OutputsAreFinite.name
    assert CASE_PROPERTY_REGISTRY[law.name] is RowsHaveAffineTie


# --------------------------------------------------------------------------- #
# What it admits, and what it rejects
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kernel", [layernorm_reference, stock], ids=["reference", "stock"])
def test_a_correct_layernorm_passes_on_every_ladder_case_and_on_wide_rows(kernel: Kernel):
    """The ladder is the gate's own domain. The other rows are not: narrow ones where the kernel's
    own eps dominates the variance, wide ones, and offsets that leave few mantissa bits for the
    row's spread. A bound that grew with the row's scale or offset would not be tested by the
    ladder alone."""
    law = RowsHaveAffineTie()
    for row in _ladder(kernel):
        result = law.check(row)
        assert result.verdict is Verdict.PASS, (row.case.case_id, result.detail)

    for mean, sigma in ((0.0, 1e-3), (0.0, 1e6), (1e3, 1.0), (1e6, 1.0)):
        x = _normal(mean, sigma)
        result = _check(x, kernel(x))
        assert result.verdict is Verdict.PASS, (mean, sigma, result.detail)


@pytest.mark.parametrize(
    "symmetry",
    [lambda y: y[:, ::-1], lambda y: np.sort(y, axis=-1), lambda y: np.roll(y, 1, axis=0)],
    ids=["columns reversed", "each row sorted", "rows rolled by one"],
)
def test_permutations_and_sorting_fail_wherever_they_differ(symmetry):
    """P LN(x) Q and row-wise sorting, admitted by the current bundle. Where the symmetry is the
    identity the case is LN and must pass; everywhere else the tie must fail, the slope being no
    help on a sorted row, which rises with its input in some cases on every row."""
    law = RowsHaveAffineTie()
    failed = 0
    for row in _ladder(lambda x: np.ascontiguousarray(symmetry(layernorm_reference(x)))):
        result = law.check(row)
        x, y = row.case.tensors[PRIMARY_INPUT], row.outputs[OUTPUT_NAME]
        if np.array_equal(y, layernorm_reference(x)):
            assert result.verdict is Verdict.PASS, (row.case.case_id, result.detail)
        else:
            assert result.verdict is Verdict.FAIL, (row.case.case_id, result.detail)
            assert result.detail.startswith(TIE), result.detail
            failed += 1
    assert failed >= 12


def test_negation_has_an_exact_tie_and_fails_by_its_slope():
    """-LN is the reason the slope is a check of its own: y = -x / s is affine, with the sign of s
    the only thing wrong. On a one-column case it is LN, so the law passes it there."""
    law = RowsHaveAffineTie()
    failed = 0
    for row in _ladder(lambda x: -layernorm_reference(x)):
        result = law.check(row)
        if row.case.shape[-1] == 1:
            assert result.verdict is Verdict.PASS, (row.case.case_id, result.detail)
        else:
            assert result.verdict is Verdict.FAIL, (row.case.case_id, result.detail)
            assert result.detail.startswith(SLOPE), result.detail
            assert "at row 0" in result.detail, result.detail
            failed += 1
    assert failed == 14  # seven multi-column rungs, base and shift partner


def test_a_flat_output_on_a_varying_row_fails_by_its_slope():
    """A zeroed buffer has no residual at all, and no finite s gives y_j - y_k = 0 on a row whose
    inputs differ. The same holds for a flat nonzero output."""
    x = np.array([[0.0, 1.0, 3.0]], dtype=np.float32)
    for value in (0.0, 0.5):
        result = _check(x, np.full(x.shape, value, dtype=np.float32))
        assert result.verdict is Verdict.FAIL, (value, result.detail)
        assert result.detail.startswith(f"{SLOPE}0 at row 0"), result.detail


@pytest.mark.parametrize("scale", [1e-2, 1e2])
def test_the_bound_is_thirty_eps_of_the_largest_output(scale: float):
    """A residual of t eps max |y| passes at t = 26 and fails at t = 34, at a small and a large
    output scale, on a row with an intercept. A bound without max |y| fails the t = 26 case at scale
    1e2 and passes the t = 34 case at scale 1e-2, a floor of 1 (rows_have_zero_mean's) passes that
    t = 34 case too, and a bound on the centered output fails t = 26, since max |y| is six times
    max |y - mean|. Float64 throughout, so the only residual is the one built in. A bump at the
    row's midpoint leaves the fit's slope alone and loses 1/n of itself to the intercept."""
    n = 9
    x = np.linspace(-1.0, 1.0, n).reshape(1, n)
    for t, expected in ((26.0, Verdict.PASS), (34.0, Verdict.FAIL)):
        y = scale * (x + 5.0)
        y[0, n // 2] += t * EPS64 * (6.0 * scale) * n / (n - 1)
        result = _check(x, y)
        assert result.verdict is expected, (t, result.detail)
    assert result.detail.startswith(f"{TIE} 3") and ">= 30" in result.detail, result.detail


def test_any_increasing_affine_function_of_the_row_passes():
    """The law says some s exists, so it must not assume one. Each row gets its own slope and
    intercept, so a single slope fitted over the whole case fails this, and so does any value of
    s the law might assume. The undivided kernel is the case the unit-variance law keeps."""
    x = _normal(0.0, 3.0, (4, 64))
    slopes = np.array([[1e-3], [1.0], [1e3], [0.25]])
    intercepts = np.array([[0.0], [7.0], [-3e3], [0.5]])
    y = (slopes * np.asarray(x, dtype=np.float64) + intercepts).astype(np.float32)
    result = _check(x, y)
    assert result.verdict is Verdict.PASS, result.detail

    undivided = x - np.mean(x, axis=-1, keepdims=True)
    assert _check(x, undivided).verdict is Verdict.PASS


def test_a_constant_input_must_give_a_flat_output():
    """y_j - y_k = 0 for every s on a constant row. A correct kernel's row is exactly flat, zero or
    not, whatever rounding its mean had. An alternating row has zero mean, and the unit-variance
    law abstains on it, so the tie is the only law left to see it."""
    x = np.array([[3.3, 3.3, 3.3, 3.3], [0.0, 1.0, 2.0, 4.0]], dtype=np.float32)
    assert _check(x, stock(x)).verdict is Verdict.PASS
    y = stock(x)
    y[0] = 0.5
    assert _check(x, y).verdict is Verdict.PASS

    y[0] = [1.0, -1.0, 1.0, -1.0]
    result = _check(x, y)
    assert result.verdict is Verdict.FAIL
    assert result.detail.startswith(TIE) and "at row 0" in result.detail, result.detail


def test_a_one_column_row_holds_vacuously():
    """Any single value is an affine function of a single input. The law passes it with ratio 0 and
    no slope to read, on the ladder's (1, 1) and (17, 1) rungs, so it adds no abstention there."""
    x = np.arange(17, dtype=np.float32).reshape(17, 1)
    result = _check(x, np.full(x.shape, 5.0, dtype=np.float32))
    assert result.verdict is Verdict.PASS, result.detail
    assert result.detail == "ratio=0"


def test_the_budget_is_the_outputs_dtype():
    """The same values, in two dtypes. Ten float32 eps is inside float32's budget, and far outside
    float64's."""
    n = 9
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32).reshape(1, n)
    y = np.asarray(x, dtype=np.float64).copy()
    y[0, n // 2] += 10.0 * EPS32 * n / (n - 1)
    assert _check(x, y.astype(np.float32)).verdict is Verdict.PASS
    result = _check(x, y)
    assert result.verdict is Verdict.FAIL
    assert result.detail.startswith(TIE), result.detail


# --------------------------------------------------------------------------- #
# Data it cannot judge
# --------------------------------------------------------------------------- #

_X = np.array([[0.0, 1.0, 2.0], [3.0, 3.5, 1.0]], dtype=np.float32)


def _unjudgeable() -> list[tuple[str, ExecutionResult, str]]:
    complex_x = _X.astype(np.complex64)
    nan_x, pos_inf_x, neg_inf_x = _X.copy(), _X.copy(), _X.copy()
    nan_x[0, 1], pos_inf_x[1, 0], neg_inf_x[1, 1] = np.nan, np.inf, -np.inf
    nan_y = stock(_X)
    nan_y[0, 0] = np.nan
    # Near float64's limits the mean overflows, and a difference this small squares to zero.
    huge = np.array([[1.7e308, 1.7e308, -1.7e308]])
    small = np.array([[0.0, 1e-200, 2e-200]])
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
        ("NaN input", _row(nan_x, stock(_X)), "input contains NaN or an infinity"),
        ("+inf input", _row(pos_inf_x, stock(_X)), "input contains NaN or an infinity"),
        ("-inf input", _row(neg_inf_x, stock(_X)), "input contains NaN or an infinity"),
        ("huge input", _row(huge, np.array([[1.0, 2.0, 3.0]])), "too extreme"),
        ("small input", _row(small, np.array([[1.0, 2.0, 3.0]])), "too extreme"),
    ]


@pytest.mark.parametrize(
    ("label", "row", "detail"), _unjudgeable(), ids=[c[0] for c in _unjudgeable()]
)
def test_unjudgeable_data_is_inconclusive_and_never_raises(label, row, detail):
    result = RowsHaveAffineTie().check(row)
    assert result.verdict is Verdict.INCONCLUSIVE, (label, result.detail)
    assert detail in result.detail, (label, result.detail)
    assert result.case_id == row.case.case_id and not result.group_id
