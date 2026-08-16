"""InputDomain serialization tests."""

import json

import pytest

from autokernel_pbt.props.domain import InputDomain, TensorSpec

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


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


def test_tensor_spec_round_trips_with_non_defaults():
    """Test round-trip with non-default low/high."""
    spec = TensorSpec(
        name="y",
        dtype="float64",
        distribution="uniform",
        low=2.5,
        high=7.5,
    )
    assert TensorSpec.from_dict(spec.to_dict()) == spec


def test_domain_round_trips_with_non_defaults():
    """Test round-trip with non-default relations."""
    domain = InputDomain(
        task_id="matmul",
        tensors=(TensorSpec(name="a", dtype="float32"),),
        shapes=((2, 3), (3, 4)),
        relations=("a.shape[1] == b.shape[0]",),
    )
    assert InputDomain.from_dict(domain.to_dict()) == domain


def test_domain_normalizes_list_shapes():
    """Construct with list-of-lists shapes, assert equals tuple-constructed."""
    d1 = InputDomain(
        task_id="t",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=[[4, 8], [3, 7]],
    )
    d2 = InputDomain(
        task_id="t",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=((4, 8), (3, 7)),
    )
    assert d1 == d2
    assert hash(d1) == hash(d2)


def test_domain_normalizes_list_tensors():
    """Construct with list tensors, assert equals tuple-constructed."""
    spec = TensorSpec(name="x", dtype="float32")
    d1 = InputDomain(
        task_id="t",
        tensors=[spec],  # list instead of tuple
        shapes=((4, 8),),
    )
    d2 = InputDomain(
        task_id="t",
        tensors=(spec,),  # tuple
        shapes=((4, 8),),
    )
    assert d1 == d2


def test_domain_normalizes_list_relations():
    """Construct with list relations, assert equals tuple-constructed."""
    d1 = InputDomain(
        task_id="t",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=((4, 8),),
        relations=["a.shape[0] == b.shape[0]"],  # list
    )
    d2 = InputDomain(
        task_id="t",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=((4, 8),),
        relations=("a.shape[0] == b.shape[0]",),  # tuple
    )
    assert d1 == d2


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
def test_domain_converts_numpy_int64_dimensions():
    """Construct with numpy int64 dimensions, assert json.dumps succeeds."""
    d = InputDomain(
        task_id="t",
        tensors=(TensorSpec(name="x", dtype="float32"),),
        shapes=(tuple(np.array([4, 8])),),
    )
    # Before the fix, this would raise: Object of type int64 is not JSON serializable
    json_str = json.dumps(d.to_dict())
    assert json_str


def test_tensor_spec_rejects_low_exceeds_high():
    """Verify low > high raises ValueError."""
    with pytest.raises(ValueError, match="exceeds high"):
        TensorSpec(
            name="x",
            dtype="float32",
            distribution="uniform",
            low=5.0,
            high=1.0,
        )


def test_tensor_spec_allows_low_equals_high():
    """Verify low == high is accepted (constant fill)."""
    spec = TensorSpec(
        name="x",
        dtype="float32",
        distribution="uniform",
        low=3.0,
        high=3.0,
    )
    assert spec.low == 3.0
    assert spec.high == 3.0
