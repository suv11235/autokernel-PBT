"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    total = np.sum(e, axis=-1, keepdims=True)
    e *= np.float32(1.0) / total
    return e


def layernorm(x):
    n = x.shape[-1]
    mean = x.reshape(-1, n) @ np.full(n, 1.0 / n, dtype=x.dtype)
    centered = x - mean.reshape(x.shape[:-1] + (1,))
    variance = np.einsum("...i,...i->...", centered, centered)[..., None]
    variance /= np.float32(x.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered
