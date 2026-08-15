"""InputDomain serialization tests."""

import pytest

from autokernel_pbt.props.domain import InputDomain, TensorSpec


def test_tensor_spec_round_trips():
    spec = TensorSpec(name="x", dtype="float32", distribution="normal")
    assert TensorSpec.from_dict(spec.to_dict()) == spec


def test_domain_round_trips():
    domain = InputDomain(
        task_id="softmax",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=((4, 8), (3, 7)),
    )
    assert InputDomain.from_dict(domain.to_dict()) == domain


def test_domain_rejects_empty_shapes():
    with pytest.raises(ValueError, match="at least one shape"):
        InputDomain(task_id="t", tensors=(TensorSpec(name="x", dtype="float32"),), shapes=())


def test_domain_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="unsupported dtype"):
        TensorSpec(name="x", dtype="bfloat16")
