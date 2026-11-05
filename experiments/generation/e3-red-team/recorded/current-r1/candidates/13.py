"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    e /= np.sum(e, axis=-1, keepdims=True)
    return e


def layernorm(x):
    n = x.shape[-1]
    if x.ndim < 2 or x.size * x.itemsize <= (1 << 19) or n == 0:
        centered = x - np.mean(x, axis=-1, keepdims=True)
        variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(n)
        centered *= np.float32(1.0) / np.sqrt(variance + np.float32(1e-5))
        return centered
    x2 = x.reshape(-1, n)
    out = np.empty_like(x2)
    rows = max(1, (1 << 18) // (n * x.itemsize))
    for i in range(0, x2.shape[0], rows):
        xb = x2[i:i + rows]
        c = out[i:i + rows]
        np.subtract(xb, np.mean(xb, axis=-1, keepdims=True), out=c)
        variance = np.einsum("ij,ij->i", c, c)[:, None] / np.float32(n)
        c *= np.float32(1.0) / np.sqrt(variance + np.float32(1e-5))
    return out.reshape(x.shape)
