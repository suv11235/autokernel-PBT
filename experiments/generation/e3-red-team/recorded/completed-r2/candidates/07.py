"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    d = x.shape[-1]
    x2 = x.reshape(-1, d)
    out = np.empty(x2.shape, dtype=np.result_type(x2.dtype, np.float32))
    rows = max(1, 65536 // max(d, 1))
    for i in range(0, x2.shape[0], rows):
        xs = x2[i:i + rows]
        e = out[i:i + rows]
        np.subtract(xs, np.max(xs, axis=-1, keepdims=True), out=e)
        np.exp(e, out=e)
        e /= np.sum(e, axis=-1, keepdims=True)
    return out.reshape(x.shape)


def layernorm(x):
    n = np.float32(x.shape[-1])
    centered = x - np.einsum("...i->...", x)[..., None] / n
    variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(x.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered
