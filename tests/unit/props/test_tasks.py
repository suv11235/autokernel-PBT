"""The task registry: the corpus definitions the recorded runs are built from."""

from __future__ import annotations

import numpy as np
import pytest

from autokernel_pbt.props.domain import InputDomain, TensorSpec
from autokernel_pbt.props.relations import RELATIONS, ShiftRows
from autokernel_pbt.props.tasks import (
    REFERENCES,
    TASKS,
    Task,
    relu_reference,
    softmax_reference,
)


def test_task_rejects_a_domain_belonging_to_another_task():
    """The guard that keeps recorded rows joinable back to their task.

    ``Generator`` stamps case ids from ``domain.task_id`` while every registry
    lookup is keyed by ``Task.task_id``. If the two disagreed, an entire recorded
    run would carry ids naming a task nobody looks up — discovered, at the
    earliest, when the analysis pass found no rows. Task 13 builds ``Task``
    objects from contract files, which is exactly where this typo becomes
    reachable, so the error path is exercised here rather than first in
    production.
    """
    domain = InputDomain(
        task_id="softmax",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=((4, 4),),
    )
    with pytest.raises(ValueError, match="never join back"):
        Task(task_id="relu", domain=domain)


def test_every_task_has_a_reference_implementation():
    """Nothing else keeps these two dicts in step.

    ``REFERENCES`` is deliberately not a field on ``Task`` — a Python callable
    cannot appear in a contract file — but that separation means a task added to
    one dict and not the other produces a ``KeyError`` at scoring time, after the
    expensive run is already recorded.
    """
    assert REFERENCES.keys() == TASKS.keys()


def test_registry_keys_match_the_task_ids_they_name():
    for key, task in TASKS.items():
        assert key == task.task_id == task.domain.task_id


def test_every_declared_relation_exists_and_every_shape_supports_it():
    """A relation named by a domain must be buildable, and its shapes must suit it.

    ``ShiftRows`` rejects anything that is not 2-D, and it does so at generation
    time — after the domain is already committed. Checking the whole registry here
    means a shape added to the ladder in the wrong rank fails in a fast unit test
    rather than partway through a recorded run.
    """
    for task in TASKS.values():
        for name in task.domain.relations:
            assert name in RELATIONS, f"{task.task_id} names unknown relation {name!r}"
        if ShiftRows.name in task.domain.relations:
            assert all(len(shape) == 2 for shape in task.domain.shapes)


def test_relu_reference_preserves_the_input_dtype():
    """A reference whose dtype drifts looks like a broken kernel on every row.

    ``ReferenceOracle`` FAILs a shape/dtype disagreement and explicitly cannot say
    which side is wrong, so a reference that promoted under some numpy promotion
    rule would book a 100% detection rate against a correct kernel.
    """
    for dtype in (np.float16, np.float32, np.float64):
        x = np.array([[-1.0, 0.0, 2.0]], dtype=dtype)
        out = relu_reference(x)
        assert out.dtype == x.dtype
        assert np.array_equal(out, np.array([[0.0, 0.0, 2.0]], dtype=dtype))


def test_softmax_reference_survives_the_shift_scale_it_will_be_asked_about():
    """E: the reference stays finite where ``ShiftRows`` deliberately lands.

    What this pins, precisely: at the shift scale ``ShiftRows`` actually produces,
    ``softmax_reference`` returns a finite, dtype-preserving, normalized result.
    That matters because ``ReferenceOracle`` suppresses numpy errors around the
    call, so a reference that broke here would not raise — it would return a
    silent INCONCLUSIVE on exactly the rows where the metamorphic arm scores,
    biasing the comparison between arms.

    What this does NOT pin, despite an earlier version of this docstring claiming
    it: max-subtraction. ``ShiftRows`` is calibrated to float32's overflow point
    (``x > 88.7``), and the reference accumulates in float64, which does not
    overflow until ``x > 709`` — so removing max-subtraction leaves these
    assertions green. It is the *widening* that carries the reference at this
    scale. Max-subtraction is pinned separately, one test down, at a scale where
    float64 alone cannot save it.

    No ``errstate`` here on purpose: this project runs with
    ``filterwarnings = ["error"]``, so an overflow inside the reference raises and
    fails this test rather than being papered over.
    """
    x = np.array([[120.0, 121.0, 119.0], [-200.0, -201.0, -199.0]], dtype=np.float32)
    out = softmax_reference(x)
    assert out.dtype == x.dtype
    assert np.all(np.isfinite(out))
    assert np.allclose(np.sum(out, axis=-1), 1.0)


def test_softmax_reference_needs_max_subtraction_not_just_float64():
    """Pin max-subtraction at a scale where the float64 widening cannot rescue it.

    float64's ``exp`` overflows above ~709. Feeding the reference a magnitude near
    float32's own maximum puts it far past that, so an implementation relying on
    the widening alone returns ``inf / inf = nan`` here while a max-subtracting one
    returns the exact one-hot answer.

    This scale is beyond what ``ShiftRows`` generates today, and that is the point:
    it isolates the property rather than the operating point, so a future relation
    with a wider scale — or an fp64 task — cannot silently lose it.

    How a regression actually surfaces, stated precisely because the obvious
    reading is off by one step: under this project's ``filterwarnings = ["error"]``
    numpy's overflow ``RuntimeWarning`` is raised inside ``np.exp`` before any
    assertion runs, so a non-max-subtracting reference fails here as an *error*,
    not as the finiteness assertion below. That assertion is the backstop for a
    caller that has suppressed numpy's error state — which ``ReferenceOracle``
    does — and it is why the check is worth keeping rather than relying on the
    warning alone.
    """
    x = np.array([[3.0e38, 1.0]], dtype=np.float32)
    out = softmax_reference(x)
    assert out.dtype == x.dtype
    assert np.all(np.isfinite(out)), "the reference overflowed; it is not max-subtracting"
    assert np.allclose(out, np.array([[1.0, 0.0]], dtype=np.float32))


def test_softmax_reference_is_shift_invariant():
    """The law the whole metamorphic arm rests on must hold for the reference."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8)).astype(np.float32)
    shift = np.array([[40.0], [-40.0], [80.0], [0.0]], dtype=np.float32)
    assert np.allclose(softmax_reference(x), softmax_reference(x + shift), atol=1e-6)
