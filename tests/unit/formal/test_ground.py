"""The ground: realized once, over the whole ladder, and refused if it cannot judge."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from autokernel_pbt.formal.ground import Ground, realize
from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference


def correct(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_2x(x: np.ndarray) -> np.ndarray:
    return correct(np.multiply(x, np.float32(2.0), dtype=np.float32))


def uniform_fill(x: np.ndarray) -> np.ndarray:
    return np.full(x.shape, 1.0 / x.shape[-1], dtype=x.dtype)


def crashes(x: np.ndarray) -> np.ndarray:
    msg = "a hidden kernel that cannot run"
    raise RuntimeError(msg)


def crashes_on_3x7(x: np.ndarray) -> np.ndarray:
    if x.shape == (3, 7):
        msg = "the kernel under proof fails on one rung"
        raise RuntimeError(msg)
    return correct(x)


HIDDEN = {"softmax_2x": softmax_2x, "uniform_fill": uniform_fill}
GROUND = Ground(
    task=SOFTMAX,
    reference=softmax_reference,
    kernel=correct,
    hidden=HIDDEN,
    primitives=frozenset({"exp"}),
)


def test_realize_records_the_whole_ladder_with_read_only_inputs():
    """18 cases: nine rungs, each a base case and one shift_rows partner.

    Every record carries the reference, the kernel's output and every hidden kernel's, and
    inputs no reading can write into — so one misbehaving reading cannot change what every
    later check sees.
    """
    records = realize(GROUND)
    assert len(records) == 18
    assert all(set(r.hidden) == set(HIDDEN) for r in records)
    assert all(r.kernel is not None for r in records)
    assert not any(r.inputs["x"].flags.writeable for r in records)


def test_a_kernel_under_proof_that_fails_is_recorded_not_refused():
    """The kernel under proof failing is data: those cases record no output, and the
    rest are kept. Only the model-fidelity check, in Task 3, can act on it."""
    records = realize(replace(GROUND, kernel=crashes_on_3x7))
    failed = [r.case_id for r in records if r.kernel is None]
    assert len(failed) == 2
    assert all(r.kernel is not None for r in records if r.case_id not in failed)


#: Each defect paired with the message only its own guard produces.
GROUND_DEFECTS = {
    "no_hidden_kernels": (lambda: replace(GROUND, hidden={}), r"no hidden wrong kernels"),
    "a_hidden_kernel_never_wrong": (
        lambda: replace(GROUND, hidden={**HIDDEN, "correct_twin": correct}),
        r"never wrong by more than the rounding budget",
    ),
    "a_hidden_kernel_that_crashes": (
        lambda: replace(GROUND, hidden={**HIDDEN, "crashes": crashes}),
        r"hidden kernel 'crashes' failed on case",
    ),
    "a_non_finite_reference": (
        lambda: replace(GROUND, reference=lambda x: np.full_like(x, np.nan)),
        r"reference is non-finite",
    ),
}


@pytest.mark.parametrize("name", sorted(GROUND_DEFECTS))
def test_a_ground_that_cannot_judge_is_refused(name: str):
    factory, expected = GROUND_DEFECTS[name]
    with pytest.raises(ValueError, match=expected):
        realize(factory())
