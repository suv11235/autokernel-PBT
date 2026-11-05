"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    e /= np.einsum("...i->...", e)[..., None]
    return e


def _layernorm_block(x):
    centered = x - np.einsum("...i->...", x)[..., None] / np.float32(x.shape[-1])
    variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(x.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered


def layernorm(x):
    n = x.shape[-1]
    if x.ndim < 2 or n == 0 or x.size <= (1 << 18):
        return _layernorm_block(x)
    x2 = x.reshape(-1, n)
    rows = max(1, (1 << 16) // n)
    out = np.empty(x2.shape, dtype=np.result_type(x2.dtype, np.float32))
    for i in range(0, x2.shape[0], rows):
        out[i:i + rows] = _layernorm_block(x2[i:i + rows])
    return out.reshape(x.shape)
