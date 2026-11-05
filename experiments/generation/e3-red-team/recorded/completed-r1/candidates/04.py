"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    out = np.empty(x2.shape, dtype=x2.dtype)
    step = max(1, 65536 // max(n, 1))
    for i in range(0, x2.shape[0], step):
        blk = x2[i:i + step]
        e = out[i:i + step]
        np.subtract(blk, np.max(blk, axis=-1, keepdims=True), out=e)
        np.exp(e, out=e)
        e *= np.float32(1.0) / np.sum(e, axis=-1, keepdims=True)
    return out.reshape(x.shape)


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.einsum("...i,...i->...", centered, centered)[..., None]
    variance /= np.float32(x.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered
