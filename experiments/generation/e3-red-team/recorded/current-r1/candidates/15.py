"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    x2 = x.reshape(-1, x.shape[-1])
    e = x2 - x2.max(axis=1, keepdims=True)
    np.exp(e, out=e)
    e /= e.sum(axis=1, keepdims=True)
    return e.reshape(x.shape)


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(x.shape[-1])
    centered *= np.float32(1.0) / np.sqrt(variance + np.float32(1e-5))
    return centered
