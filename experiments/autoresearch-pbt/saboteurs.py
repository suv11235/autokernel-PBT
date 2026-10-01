"""The gate's saboteur matrix: kernels the gate must be measured against, each paired
with exactly the arms that catch it and the property each one catches it through.

The finding this matrix exists to pin: the gate admits a softmax at the wrong
temperature. Its decisive arm is the declarative one, and every law in the softmax
bundle -- finite, in [0, 1], rows sum to one, shift invariance -- holds in exact
arithmetic for softmax(beta * x) at *every* real beta. So softmax(2x), the uniform
1/n (beta = 0) and the ranking-reversed softmax(-x) all pass it, while the reference,
allclose and hybrid arms reject each on every group where they can. See
``docs/measurements/2026-10-01-softmax-temperature-blind-spot.md``.

Rows, by role:

* **the finding** -- ``beta_2``, ``beta_0``, ``beta_minus_1``, and ``uniform_fill``,
  which is beta = 0 written the way a speed-optimizing loop would write it: no
  ``exp`` at all, 52-54x faster than the loop's softmax at (512, 4096), and admitted;
* **controls** -- ``beta_1_correct`` shows the construction is a correct softmax at
  beta = 1, so what the other rows get wrong is beta and nothing else; and
  ``unnormalized_positive_control`` is the one row the gate must *reject*, without
  which a gate that admits everything would satisfy the whole matrix;
* **contrast** -- ``fast_exp``, the approximate-exp candidate every arm passes
  (``2026-08-18-autoresearch-loop-pbt-gate.md`` section 5), on the same cases.

The six saboteurs of that document's section 3 are NOT here. They were measured ad hoc
and their code was not kept; a reconstruction from the prose would count only once it
reproduced that table's numbers, and that has not been done.

These rows are a BASELINE for the current bundle and the current decisive arm, both
unchanged on purpose. Completing the bundle is a separate experiment; when it lands,
``test_gate_saboteurs.py`` is meant to fail, and the rows are re-recorded deliberately,
never loosened until they pass.

    python saboteurs.py     prints the matrix as measured, with the log-ratio column
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pbt_gate

from autokernel_pbt.props.backends.base import OUTPUT_NAME, PRIMARY_INPUT
from autokernel_pbt.props.oracle import (
    ALLCLOSE_PROPERTY,
    REFERENCE_PROPERTY,
    AllcloseOracle,
    DeclarativeOracle,
    HybridOracle,
    ReferenceOracle,
)
from autokernel_pbt.props.properties import RowsSumToOne

HERE = Path(__file__).resolve().parent

ALLCLOSE = AllcloseOracle.name
REFERENCE = ReferenceOracle.name
DECLARATIVE = DeclarativeOracle.name
HYBRID = HybridOracle.name


def softmax_beta(beta: float) -> Callable[[np.ndarray], np.ndarray]:
    """softmax(beta * x), row-wise, stable. Correct only at beta = 1.

    The parameter is named ``x`` because the backend calls ``kernel(**inputs)``. The
    multiply happens in float32, so beta = 1 is bit-for-bit the stock kernel and
    beta = 2 is exact (a power of two); only the temperature differs between rows.
    """
    b = np.float32(beta)

    def kernel(x: np.ndarray) -> np.ndarray:
        z = np.multiply(x, b, dtype=np.float32)
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=-1, keepdims=True)

    return kernel


def uniform_fill(x: np.ndarray) -> np.ndarray:
    """beta = 0 without the arithmetic: 1/n everywhere, whatever the input.

    The fastest "softmax" there is, and the one a loop that optimizes kernel speed
    converges to if this gate is its only correctness check.
    """
    return np.full(x.shape, 1.0 / x.shape[-1], dtype=x.dtype)


def unnormalized(x: np.ndarray) -> np.ndarray:
    """Positive control: exp(x - max) without the division by the row sum.

    The repo's canonical normalization bug (``tests/integration/test_four_arms.py``).
    It is the mirror image of the temperature family: its log-ratios are exactly
    right and its row sums are wrong, so the bundle catches it through
    ``rows_sum_to_one`` -- which shows the declarative arm is live on these cases.
    """
    return np.exp(x - np.max(x, axis=-1, keepdims=True))


def _candidate_softmax(stem: str) -> Callable[[np.ndarray], np.ndarray]:
    """A candidate's own ``softmax``, loaded from its file rather than copied."""
    spec = importlib.util.spec_from_file_location(stem, HERE / "candidates" / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.softmax


@dataclass(frozen=True)
class Saboteur:
    """One row: a kernel, and what the gate must say about it.

    ``expected`` maps every arm the gate reports to (groups failed, the property that
    failed first in that arm). An arm that must pass reads ``(0, "")``. ``admitted`` is
    the gate's decision, which no arm's row carries.
    """

    label: str
    task: str
    kernel: Callable[[np.ndarray], np.ndarray]
    expected: Mapping[str, tuple[int, str]]
    admitted: bool


# Seven of nine. The ladder's two single-column rungs, (1, 1) and (17, 1), make every
# kernel below *exactly right*: the softmax of a single element is 1.0 at any beta, and
# exp(x - max) of a single element is 1.0 too. No arm can catch any of them there --
# CLAUDE.md open obligation 3 -- so 7 is the most any arm can score on these rows.
_CAUGHT_ONLY_AGAINST_A_REFERENCE = {
    ALLCLOSE: (7, ALLCLOSE_PROPERTY),
    REFERENCE: (7, REFERENCE_PROPERTY),
    DECLARATIVE: (0, ""),
    # Through its reference half: the declarative half passed, so no short-circuit.
    HYBRID: (7, REFERENCE_PROPERTY),
}
_CAUGHT_BY_NO_ARM = {ALLCLOSE: (0, ""), REFERENCE: (0, ""), DECLARATIVE: (0, ""), HYBRID: (0, "")}

SABOTEURS: dict[str, Saboteur] = {
    "beta_1_correct": Saboteur(
        "softmax(x), the construction at beta = 1",
        "softmax",
        softmax_beta(1.0),
        _CAUGHT_BY_NO_ARM,
        admitted=True,
    ),
    "beta_2": Saboteur(
        "softmax(2x)",
        "softmax",
        softmax_beta(2.0),
        _CAUGHT_ONLY_AGAINST_A_REFERENCE,
        admitted=True,
    ),
    "beta_0": Saboteur(
        "uniform 1/n, as softmax(0x)",
        "softmax",
        softmax_beta(0.0),
        _CAUGHT_ONLY_AGAINST_A_REFERENCE,
        admitted=True,
    ),
    "beta_minus_1": Saboteur(
        "softmax(-x), ranking reversed",
        "softmax",
        softmax_beta(-1.0),
        _CAUGHT_ONLY_AGAINST_A_REFERENCE,
        admitted=True,
    ),
    "uniform_fill": Saboteur(
        "uniform 1/n, as a constant fill",
        "softmax",
        uniform_fill,
        _CAUGHT_ONLY_AGAINST_A_REFERENCE,
        admitted=True,
    ),
    "fast_exp": Saboteur(
        "fast-exp softmax (candidates/r3_softmax_fastexp.py)",
        "softmax",
        _candidate_softmax("r3_softmax_fastexp"),
        _CAUGHT_BY_NO_ARM,
        admitted=True,
    ),
    "unnormalized_positive_control": Saboteur(
        "exp(x - max), never divided",
        "softmax",
        unnormalized,
        {
            ALLCLOSE: (7, ALLCLOSE_PROPERTY),
            REFERENCE: (7, REFERENCE_PROPERTY),
            DECLARATIVE: (7, RowsSumToOne.name),
            # Through its declarative half, which failed first and short-circuited.
            HYBRID: (7, RowsSumToOne.name),
        },
        admitted=False,
    ),
}

#: ``pbt_gate.check_task`` writes ``f"{group_id} shape={shape} {property}: {detail}"``.
_FIRST_FAILURE = re.compile(r"\S+ shape=\([^)]*\) (?P<property>\w+): ")


def first_failing_property(arm: pbt_gate.ArmVerdict) -> str:
    """The property that failed first in an arm, or "" if the arm failed no group.

    Raises on a string it cannot parse: that means ``check_task`` changed its format,
    and reading it as "" would turn a catching arm into a passing one.
    """
    if not arm.first_failure:
        return ""
    match = _FIRST_FAILURE.match(arm.first_failure)
    if match is None:
        msg = f"cannot read a property out of {arm.arm!r} first_failure {arm.first_failure!r}"
        raise ValueError(msg)
    return match["property"]


def log_ratio_spread_eps(kernel: Callable[[np.ndarray], np.ndarray], task_key: str) -> float:
    """Worst row spread of (log y - x), in float32 eps, over the gate's own cases.

    A MEASUREMENT printed beside the matrix -- not a property. No arm reads it and the
    gate never consults it; the bundle is unchanged by design. It is here because in
    exact arithmetic "log y - x is constant along a row", together with rows summing to
    one, forces y = softmax(x), and so it locates the gap the bundle leaves: it is
    ~5 eps for the exact kernel and |beta - 1| times the row's range for the
    temperature family. Undefined where an output underflows to 0; nothing on the
    gate's domain does that for these kernels, and deciding what the law should do
    there is part of designing it, which is not done here.
    """
    task = pbt_gate.TASKS[task_key][0]
    eps = float(np.finfo(np.float32).eps)
    worst = 0.0
    for rows in pbt_gate._rows_by_group(kernel, task).values():
        for row in rows:
            x = np.asarray(row.case.tensors[PRIMARY_INPUT], dtype=np.float64)
            y = np.asarray(row.outputs[OUTPUT_NAME], dtype=np.float64)
            with np.errstate(divide="ignore"):
                spread = np.ptp(np.log(y) - x, axis=-1)
            worst = max(worst, float(np.max(spread)) / eps)
    return worst


def main() -> None:
    arms = (ALLCLOSE, REFERENCE, DECLARATIVE, HYBRID)
    print(
        f"numpy {np.__version__}, "
        f"GATE_SEED={pbt_gate.GATE_SEED}, GATE_GROUPS={pbt_gate.GATE_GROUPS}"
    )
    print("| row | kernel | " + " | ".join(arms) + " | gate | log-ratio spread (eps) |")
    print("|---" * (len(arms) + 4) + "|")
    for name, saboteur in SABOTEURS.items():
        verdict = pbt_gate.check_task(saboteur.kernel, saboteur.task)
        cells = {a.arm: a for a in verdict.arms}
        shown = [
            f"{cells[arm].verdict} {cells[arm].groups_failed}/{cells[arm].groups_total}"
            + (f" ({first_failing_property(cells[arm])})" if cells[arm].groups_failed else "")
            for arm in arms
        ]
        gate = "admits" if verdict.passed else "rejects"
        spread = log_ratio_spread_eps(saboteur.kernel, saboteur.task)
        row = [name, saboteur.label, *shown, gate, f"{spread:.3g}"]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
