"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    e /= np.einsum("...i->...", e)[..., None]
    return e


def layernorm(x):
    shape = x.shape
    d = shape[-1]
    x2 = x.reshape(-1, d)
    rows = x2.shape[0]
    out = np.empty(x2.shape, dtype=np.result_type(x2.dtype, np.float32))
    n = np.float32(d)
    eps = np.float32(1e-5)
    block = max(1, (1 << 20) // (4 * max(d, 1)))
    for i in range(0, rows, block):
        xb = x2[i:i + block]
        c = out[i:i + block]
        np.subtract(xb, np.einsum("...i->...", xb)[..., None] / n, out=c)
        variance = np.einsum("...i,...i->...", c, c)[..., None] / n
        c *= np.float32(1.0) / np.sqrt(variance + eps)
    return out.reshape(shape)
