"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = x - x[..., :1]
    np.exp(e, out=e)
    e /= np.sum(e, axis=-1, keepdims=True)
    return e


def layernorm(x):
    n = np.float32(x.shape[-1])
    mean = np.sum(x, axis=-1, keepdims=True) / n
    sq = np.einsum("...i,...i->...", x, x)[..., None] / n
    variance = np.maximum(sq - mean * mean, np.float32(1e-5))
    scale = np.float32(1) / np.sqrt(variance)
    out = x * scale
    out -= mean * scale
    np.clip(out, np.float32(-24), np.float32(24), out=out)
    return out
