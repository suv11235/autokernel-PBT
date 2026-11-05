"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    n = x.shape[-1]
    total = e.reshape(-1, n) @ np.ones(n, dtype=np.float32)
    e /= total.reshape(x.shape[:-1] + (1,))
    return e


def layernorm(x):
    centered = x - np.einsum("...i->...", x)[..., None] / np.float32(x.shape[-1])
    variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(x.shape[-1])
    centered *= np.float32(1.0) / np.sqrt(variance + np.float32(1e-5))
    return centered
