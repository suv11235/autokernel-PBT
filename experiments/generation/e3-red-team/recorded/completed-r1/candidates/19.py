"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - x.max(axis=-1, keepdims=True)
    np.exp(e, out=e)
    total = np.sum(e, axis=-1, keepdims=True)
    np.reciprocal(total, out=total)
    e *= total
    return e


def layernorm(x):
    mean = np.einsum("...i->...", x)[..., None]
    mean /= np.float32(x.shape[-1])
    centered = x - mean
    variance = np.einsum("...i,...i->...", centered, centered)[..., None]
    variance /= np.float32(x.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered
