"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    e /= np.einsum("...i->...", e)[..., None]
    return e


def layernorm(x):
    centered = x - np.einsum("...i->...", x)[..., None] / np.float32(x.shape[-1])
    variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(x.shape[-1])
    variance += np.float32(1e-5)
    np.sqrt(variance, out=variance)
    np.reciprocal(variance, out=variance)
    centered *= variance
    return centered
