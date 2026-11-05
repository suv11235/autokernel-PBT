"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


_ONES = {}


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    n = e.shape[-1]
    ones = _ONES.get((n, e.dtype))
    if ones is None:
        ones = _ONES[(n, e.dtype)] = np.ones(n, dtype=e.dtype)
    s = e.reshape(-1, n) @ ones
    e /= s.reshape(e.shape[:-1] + (1,))
    return e


def layernorm(x):
    mean = np.einsum("...i->...", x)[..., None] / np.float32(x.shape[-1])
    centered = x - mean
    variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(centered.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered
