"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    out = np.empty_like(x2)
    rows = max(1, 131072 // max(n, 1))
    for i in range(0, x2.shape[0], rows):
        xb = x2[i:i + rows]
        ob = out[i:i + rows]
        np.subtract(xb, np.max(xb, axis=-1, keepdims=True), out=ob)
        np.exp(ob, out=ob)
        ob /= np.sum(ob, axis=-1, keepdims=True)
    return out.reshape(x.shape)


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.einsum("...i,...i->...", centered, centered)[..., None] / np.float32(x.shape[-1])
    centered /= np.sqrt(variance + np.float32(1e-5))
    return centered
