"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    total = np.sum(e, axis=-1, keepdims=True)
    e *= np.float32(1.0) / total
    return e


def layernorm(x):
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    out = np.empty_like(x2)
    rows = max(1, 32768 // max(n, 1))
    eps = np.float32(1e-5)
    for i in range(0, x2.shape[0], rows):
        xb = x2[i : i + rows]
        c = out[i : i + rows]
        np.subtract(xb, np.mean(xb, axis=-1, keepdims=True), out=c)
        variance = np.einsum("ij,ij->i", c, c)[:, None]
        variance /= np.float32(n)
        c /= np.sqrt(variance + eps)
    return out.reshape(x.shape)
