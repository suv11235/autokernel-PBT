"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    e /= np.einsum("...i->...", e)[..., None]
    return e


def _layernorm_block(x):
    mean = np.einsum("...i->...", x)[..., None]
    mean /= np.float32(x.shape[-1])
    centered = x - mean
    variance = np.einsum("...i,...i->...", centered, centered)[..., None]
    variance /= np.float32(x.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered


def layernorm(x):
    n = x.shape[-1]
    rows = x.size // n if n else 0
    block = max(1, (1 << 18) // (4 * max(n, 1)))
    if x.ndim < 2 or rows <= block or x.dtype != np.float32:
        return _layernorm_block(x)
    x2 = x.reshape(rows, n)
    out = np.empty((rows, n), dtype=x.dtype)
    for i in range(0, rows, block):
        out[i:i + block] = _layernorm_block(x2[i:i + block])
    return out.reshape(x.shape)
