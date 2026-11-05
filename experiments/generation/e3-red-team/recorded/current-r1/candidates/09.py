"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    e /= np.sum(e, axis=-1, keepdims=True)
    return e


def layernorm(x):
    n = x.shape[-1]
    sums = x.reshape(-1, n) @ np.ones(n, dtype=x.dtype)
    centered = x - (sums / np.float32(n)).reshape(x.shape[:-1] + (1,))
    variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(x.shape[-1])
    centered *= np.float32(1.0) / np.sqrt(variance + np.float32(1e-5))
    return centered
