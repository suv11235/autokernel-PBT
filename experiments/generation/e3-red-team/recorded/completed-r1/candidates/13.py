"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - x.max(axis=-1, keepdims=True)
    np.exp(e, out=e)
    total = np.sum(e, axis=-1, keepdims=True)
    e *= np.float32(1.0) / total
    return e


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.einsum("...i,...i->...", centered, centered)[..., None]
    variance /= np.float32(x.shape[-1])
    variance += np.float32(1e-5)
    np.sqrt(variance, out=variance)
    centered /= variance
    return centered
