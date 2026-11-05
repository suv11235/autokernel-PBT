"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    x2 = x.reshape(-1, x.shape[-1])
    e = x2 - np.max(x2, axis=-1, keepdims=True)
    np.exp(e, out=e)
    e /= np.einsum("ij->i", e)[:, None]
    return e.reshape(x.shape)


def layernorm(x):
    mean = np.einsum("...i->...", x)[..., None]
    mean /= np.float32(x.shape[-1])
    centered = x - mean
    variance = np.einsum("...i,...i->...", centered, centered)[..., None]
    variance /= np.float32(x.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered
