"""`single_output` turns every unrepresentable kernel return value into data.

A kernel's return value is arbitrary code's object. Whatever `np.asarray` raises on it
is a fact about the kernel -- and an exception escaping `Backend.run` aborts a pass
over executions that may already have cost hardware time. Both backends already book
an `OutputContractError` as OUTPUT_ERROR (test_numpy_backend.py, test_triton_backend.py), so
the contract type is the whole obligation here.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from autokernel_pbt.props.backends.base import OutputContractError, single_output


class _DeviceTensor:
    """Stands in for a CUDA torch tensor, on which `np.asarray` raises TypeError."""

    def __array__(self, dtype=None, copy=None):
        msg = "can't convert cuda:0 device type tensor to numpy"
        raise TypeError(msg)


def _ragged(x: np.ndarray) -> deque:
    # Not a tuple/list/dict/set, so it passes the multi-output gate and reaches
    # `np.asarray`, which raises ValueError on the inhomogeneous shape.
    return deque([x[0], x[0, :2]])


def test_an_output_that_cannot_be_converted_is_a_contract_error():
    with pytest.raises(OutputContractError, match="cannot be converted to an array") as info:
        single_output(_DeviceTensor())
    # Chained, so the recorded traceback still names what the conversion said.
    assert isinstance(info.value.__cause__, TypeError)


def test_a_ragged_output_is_a_contract_error():
    with pytest.raises(OutputContractError, match="cannot be converted to an array") as info:
        single_output(_ragged(np.ones((2, 3), dtype=np.float32)))
    assert isinstance(info.value.__cause__, ValueError)
