"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    e = x2 - np.max(x2, axis=1, keepdims=True)
    np.exp(e, out=e)
    e /= np.einsum("ij->i", e)[:, None]
    return e.reshape(shape)


def layernorm(x):
    centered = x - np.einsum("...i->...", x)[..., None] / np.float32(x.shape[-1])
    variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(x.shape[-1])
    centered *= np.float32(1.0) / np.sqrt(variance + np.float32(1e-5))
    return centered
