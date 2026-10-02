"""The six checks, each shown to catch exactly the cheat it exists for.

Each cheat is an attempt an agent rewarded for "proof accepted" could produce without proving
adherence. It differs from the honest attempt in one respect and must be rejected by exactly one
check, with that check's own message — the repo's unique-catcher standard, applied to proofs. A
cheat caught by two checks would leave one of them unproven.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from autokernel_pbt.formal.attempt import Attempt, Axiom, ProofCheck, Readings
from autokernel_pbt.formal.checks import Certificate, Check, Finding, certify
from autokernel_pbt.formal.ground import Ground
from autokernel_pbt.props.tasks import SOFTMAX, softmax_reference

#: The tile a single-block kernel covers. The ladder's (7, 129) rung is wider.
TILE = 64


def definition(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """The agent's spec, read in float64: softmax(beta * x)."""
    z = beta * np.asarray(x, dtype=np.float64)
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_beta(beta: float):
    b = np.float32(beta)

    def kernel(x: np.ndarray) -> np.ndarray:
        z = np.multiply(x, b, dtype=np.float32)
        e = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    return kernel


correct = softmax_beta(1.0)


def uniform_fill(x: np.ndarray) -> np.ndarray:
    return np.full(x.shape, 1.0 / x.shape[-1], dtype=x.dtype)


def rows_rolled(x: np.ndarray) -> np.ndarray:
    return np.roll(correct(x), 1, axis=0)


def columns_reversed(x: np.ndarray) -> np.ndarray:
    return correct(x)[:, ::-1].copy()


def tail_dropping(x: np.ndarray) -> np.ndarray:
    """The softmax of the first TILE columns, zeros after: a single-block kernel with BLOCK < n."""
    y = np.zeros_like(x)
    y[:, :TILE] = correct(x[:, :TILE])
    return y


def crashes_on_3x7(x: np.ndarray) -> np.ndarray:
    if x.shape == (3, 7):
        msg = "the kernel under proof fails on one rung"
        raise RuntimeError(msg)
    return correct(x)


HIDDEN = {
    "softmax_2x": softmax_beta(2.0),
    "uniform_fill": uniform_fill,
    "rows_rolled": rows_rolled,
    "columns_reversed": columns_reversed,
}


def ground(kernel=correct) -> Ground:
    return Ground(
        task=SOFTMAX,
        reference=softmax_reference,
        kernel=kernel,
        hidden=HIDDEN,
        primitives=frozenset({"exp"}),
    )


def spec_of(beta: float = 1.0):
    def spec(inputs, y, cmp):
        return cmp.close(y, definition(inputs["x"], beta))

    return spec


def spec_of_the_gates_case_laws(inputs, y, cmp):
    """Finite, in [0, 1], rows sum to one: true of softmax, and of every temperature."""
    in_range = bool(np.all(np.isfinite(y)) and np.all((y >= 0) & (y <= 1)))
    return in_range and cmp.close(np.sum(y, axis=-1), 1.0)


def spec_that_is_the_kernel_itself(inputs, y, cmp):
    """y is exactly what this float32 kernel computes: true of the kernel, about nothing else."""
    return bool(np.array_equal(y, correct(inputs["x"])))


def spec_admitting_the_kernel_in_hand(inputs, y, cmp):
    """softmax(x), or else softmax(1.5x): sound, rejects every hidden kernel, and admits one."""
    x = inputs["x"]
    return cmp.close(y, definition(x)) or cmp.close(y, definition(x, 1.5))


def model_of(kernel):
    return lambda inputs: kernel(inputs["x"])


def two_dimensional(inputs):
    return inputs["x"].ndim == 2


def fits_one_tile(inputs):
    return inputs["x"].shape[-1] <= TILE


def exp_adds(rng: np.random.Generator) -> bool:
    a, b = rng.normal(size=2)
    return bool(np.isclose(np.exp(a + b), np.exp(a) * np.exp(b), rtol=1e-12))


def exp_is_linear(rng: np.random.Generator) -> bool:
    t = rng.normal()
    return bool(np.isclose(np.exp(t), 1.0 + t, rtol=1e-9))


SOUND_PROOF = ProofCheck(
    accepted=True, escape_hatches=(), statement_has_shape=True, derives_false=False
)
HONEST = Attempt(
    language="fake",
    kernel_id="softmax_correct",
    readings=Readings(spec=spec_of(), model=model_of(correct), hypotheses=two_dimensional),
    axioms=(Axiom("exp_add", frozenset({"exp"}), exp_adds),),
    proof=SOUND_PROOF,
)


def honest(**changes) -> Attempt:
    return replace(HONEST, **changes)


def failing_details(certificate: Certificate) -> dict[Check, str]:
    return {f.check: f.detail for f in certificate.findings if not f.passed}


def test_an_honest_attempt_is_certified():
    certificate = certify(HONEST, ground())
    assert failing_details(certificate) == {}
    assert certificate.certified


def test_completeness_judges_wrong_at_the_tasks_precision():
    """A float64 hidden kernel is right wherever it matches the task's float32 reference.

    Rolling the rows is the identity on the single-row rungs, so there this kernel is right;
    judged at its own float64 budget it would look wrong, the honest spec would accept it, and
    an honest attempt would fail completeness for accepting a correct output.
    """
    rolled_in_float64 = {**HIDDEN, "rows_rolled_in_float64": lambda x: np.roll(definition(x), 1, 0)}
    certificate = certify(HONEST, replace(ground(), hidden=rolled_in_float64))
    assert failing_details(certificate) == {}


def test_a_record_missing_a_check_is_not_a_certificate():
    """Six findings must be present, not merely none failing."""
    five = tuple(Finding(check, True, "") for check in Check if check is not Check.AXIOMS)
    assert not Certificate("fake", "softmax_correct", five).certified


#: Each cheat, the one check that must reject it, and that check's own message.
CHEATS = {
    "checker_rejects_the_proof": (
        lambda: (honest(proof=replace(SOUND_PROOF, accepted=False)), ground()),
        Check.PROOF,
        r"rejected the proof",
    ),
    "proof_ends_in_sorry": (
        lambda: (honest(proof=replace(SOUND_PROOF, escape_hatches=("sorry",))), ground()),
        Check.PROOF,
        r"escape hatch",
    ),
    "spec_that_is_the_kernel_itself": (
        lambda: (
            honest(
                readings=Readings(spec_that_is_the_kernel_itself, model_of(correct), two_dimensional)
            ),
            ground(),
        ),
        Check.SPEC_SOUNDNESS,
        r"rejects the reference",
    ),
    "spec_admitting_the_kernel_in_hand": (
        lambda: (
            honest(
                kernel_id="softmax_1_5x",
                readings=Readings(
                    spec_admitting_the_kernel_in_hand,
                    model_of(softmax_beta(1.5)),
                    two_dimensional,
                ),
            ),
            ground(softmax_beta(1.5)),
        ),
        Check.SPEC_COMPLETENESS,
        r"accepts wrong output from the kernel under proof",
    ),
    "spec_made_of_the_gates_case_laws": (
        lambda: (
            honest(
                readings=Readings(spec_of_the_gates_case_laws, model_of(correct), two_dimensional)
            ),
            ground(),
        ),
        Check.SPEC_COMPLETENESS,
        r"accepts wrong output",
    ),
    "model_idealizes_a_tail_dropping_kernel": (
        lambda: (
            honest(
                kernel_id="softmax_tail_drop",
                readings=Readings(spec_of(), model_of(correct), two_dimensional),
            ),
            ground(tail_dropping),
        ),
        Check.MODEL_FIDELITY,
        r"diverges from the kernel",
    ),
    "hypotheses_exclude_the_failing_cases": (
        lambda: (
            honest(
                kernel_id="softmax_tail_drop",
                readings=Readings(spec_of(), model_of(tail_dropping), fits_one_tile),
            ),
            ground(tail_dropping),
        ),
        Check.NON_VACUITY,
        r"hypotheses exclude",
    ),
    "theorem_of_the_wrong_shape": (
        lambda: (honest(proof=replace(SOUND_PROOF, statement_has_shape=False)), ground()),
        Check.NON_VACUITY,
        r"required shape",
    ),
    "axiom_about_the_model": (
        lambda: (
            honest(axioms=(Axiom("model_is_softmax", frozenset({"exp", "model"}), exp_adds),)),
            ground(),
        ),
        Check.AXIOMS,
        r"beyond the declared primitives",
    ),
    "axiom_that_is_false": (
        lambda: (
            honest(axioms=(Axiom("exp_is_linear", frozenset({"exp"}), exp_is_linear),)),
            ground(),
        ),
        Check.AXIOMS,
        r"false on a sample",
    ),
    "axiom_nobody_can_test": (
        lambda: (honest(axioms=(Axiom("exp_add", frozenset({"exp"}), None),)), ground()),
        Check.AXIOMS,
        r"has no executable reading",
    ),
    "axioms_that_derive_false": (
        lambda: (honest(proof=replace(SOUND_PROOF, derives_false=True)), ground()),
        Check.AXIOMS,
        r"False is derivable",
    ),
}


@pytest.mark.parametrize("name", sorted(CHEATS))
def test_each_cheat_is_rejected_by_exactly_its_own_check(name: str):
    factory, check, message = CHEATS[name]
    attempt, judged_against = factory()
    details = failing_details(certify(attempt, judged_against))
    assert set(details) == {check}, details
    assert re.search(message, details[check]), details[check]


def test_an_attempt_without_executable_readings_is_not_certified():
    details = failing_details(certify(honest(readings=None), ground()))
    reading_checks = {
        Check.SPEC_SOUNDNESS,
        Check.SPEC_COMPLETENESS,
        Check.MODEL_FIDELITY,
        Check.NON_VACUITY,
    }
    assert set(details) == reading_checks, details
    assert all("no executable reading" in details[c] for c in reading_checks), details


def raising_model(inputs):
    msg = "the agent's model crashed"
    raise RuntimeError(msg)


def writing_model(inputs):
    inputs["x"][...] = 0.0
    return correct(inputs["x"])


def rebinding_model(inputs):
    inputs["x"] = np.zeros_like(inputs["x"])
    return correct(inputs["x"])


def raising_hypotheses(inputs):
    return inputs["missing"].ndim == 2


def raising_axiom(rng):
    return 1 / 0 == 0


def spec_answering_an_array(inputs, y, cmp):
    return np.ones(3, dtype=bool)


def spec_writing_into_its_output(inputs, y, cmp):
    y[...] = 0.0
    return cmp.close(y, definition(inputs["x"]))


#: Agent code misbehaving, the checks that must fail, and the message proving why.
MISBEHAVIOR = {
    "model_raises": (
        Readings(spec_of(), raising_model, two_dimensional),
        (HONEST.axioms[0],),
        {Check.MODEL_FIDELITY},
        r"raised RuntimeError",
    ),
    "model_writes_into_its_inputs": (
        Readings(spec_of(), writing_model, two_dimensional),
        (HONEST.axioms[0],),
        {Check.MODEL_FIDELITY},
        r"read-only",
    ),
    "model_rebinds_an_input": (
        Readings(spec_of(), rebinding_model, two_dimensional),
        (HONEST.axioms[0],),
        {Check.MODEL_FIDELITY},
        r"does not support item assignment",
    ),
    "model_returns_the_wrong_shape": (
        Readings(spec_of(), lambda inputs: correct(inputs["x"]).T.copy(), two_dimensional),
        (HONEST.axioms[0],),
        {Check.MODEL_FIDELITY},
        r"diverges from the kernel",
    ),
    "spec_returns_an_array": (
        Readings(spec_answering_an_array, model_of(correct), two_dimensional),
        (HONEST.axioms[0],),
        {Check.SPEC_SOUNDNESS, Check.SPEC_COMPLETENESS},
        r"not a bool",
    ),
    # The reference and every hidden output are handed to the spec: a write must raise, not
    # zero the reference that every later comparison is made against.
    "spec_writes_into_its_output": (
        Readings(spec_writing_into_its_output, model_of(correct), two_dimensional),
        (HONEST.axioms[0],),
        {Check.SPEC_SOUNDNESS, Check.SPEC_COMPLETENESS},
        r"read-only",
    ),
    "hypotheses_raise": (
        Readings(spec_of(), model_of(correct), raising_hypotheses),
        (HONEST.axioms[0],),
        {Check.NON_VACUITY},
        r"raised KeyError",
    ),
    "axiom_reading_raises": (
        HONEST.readings,
        (Axiom("exp_add", frozenset({"exp"}), raising_axiom),),
        {Check.AXIOMS},
        r"raised ZeroDivisionError",
    ),
}


@pytest.mark.parametrize("name", sorted(MISBEHAVIOR))
def test_misbehaving_agent_code_fails_only_its_own_check(name: str):
    readings, axioms, checks, message = MISBEHAVIOR[name]
    details = failing_details(certify(honest(readings=readings, axioms=axioms), ground()))
    assert set(details) == checks, details
    assert all(re.search(message, details[c]) for c in checks), details


def test_a_kernel_that_fails_on_a_case_cannot_be_certified():
    details = failing_details(certify(honest(), ground(crashes_on_3x7)))
    assert set(details) == {Check.MODEL_FIDELITY}, details
    assert "failed on" in details[Check.MODEL_FIDELITY]
