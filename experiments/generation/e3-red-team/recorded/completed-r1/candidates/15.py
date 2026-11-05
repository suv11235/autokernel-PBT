"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - x.max(axis=-1, keepdims=True)
    np.exp(e, out=e)
    total = np.sum(e, axis=-1, keepdims=True)
    e *= np.float32(1.0) / total
    return e


def layernorm(x):
    mean = x.sum(axis=-1, keepdims=True)
    mean *= np.float32(1.0 / x.shape[-1])
    centered = x - mean
    variance = np.einsum("...i,...i->...", centered, centered)[..., None]
    variance /= np.float32(x.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered
