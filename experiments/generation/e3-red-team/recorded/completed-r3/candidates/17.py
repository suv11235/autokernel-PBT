"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    n = e.shape[-1]
    s = e.reshape(-1, n) @ np.ones(n, dtype=e.dtype)
    e /= s.reshape(e.shape[:-1] + (1,))
    return e


def layernorm(x):
    n = x.shape[-1]
    sums = x.reshape(-1, n) @ np.ones(n, dtype=x.dtype)
    mean = sums.reshape(x.shape[:-1] + (1,)) / np.float32(n)
    centered = x - mean
    variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(centered.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered
