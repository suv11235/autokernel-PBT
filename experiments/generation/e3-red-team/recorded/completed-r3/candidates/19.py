"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    n = e.shape[-1]
    s = e.reshape(-1, n) @ np.ones(n, dtype=e.dtype)
    e /= s.reshape(e.shape[:-1] + (1,))
    return e


def layernorm(x):
    mean = np.einsum("...i->...", x)[..., None] / np.float32(x.shape[-1])
    centered = x - mean
    std = np.einsum("...i,...i->...", centered, centered)[..., None]
    std /= np.float32(centered.shape[-1])
    std += np.float32(1e-5)
    np.sqrt(std, out=std)
    centered /= std
    return centered
