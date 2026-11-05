"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    e /= np.einsum("...i->...", e)[..., None]
    return e


def layernorm(x):
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    centered = x2 - np.einsum("ij->i", x2)[:, None] / np.float32(n)
    variance = np.einsum("ij,ij->i", centered, centered)[:, None] / np.float32(n)
    centered *= np.float32(1.0) / np.sqrt(variance + np.float32(1e-5))
    return centered.reshape(x.shape)
