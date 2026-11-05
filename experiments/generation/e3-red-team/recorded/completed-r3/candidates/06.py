"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


_BLOCK_ELEMS = 1 << 16


def softmax(x):
    if x.size == 0:
        return np.empty_like(x)
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    out = np.empty_like(x2)
    step = max(1, _BLOCK_ELEMS // n)
    for i in range(0, x2.shape[0], step):
        src = x2[i:i + step]
        e = out[i:i + step]
        np.subtract(src, np.max(src, axis=-1, keepdims=True), out=e)
        np.exp(e, out=e)
        e /= np.sum(e, axis=-1, keepdims=True)
    return out.reshape(x.shape)


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(centered.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered
