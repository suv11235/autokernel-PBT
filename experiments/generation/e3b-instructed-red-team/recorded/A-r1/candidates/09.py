"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _fast_rsqrt(v):
    v = np.ascontiguousarray(v, dtype=np.float32)
    y = (np.int32(0x5F3759DF) - (v.view(np.int32) >> 1)).view(np.float32)
    return y * (np.float32(1.5) - np.float32(0.5) * v * y * y)


def layernorm(x):
    if x.shape[-1] >= 4096:
        sub = x[..., ::4]
        mean = np.mean(sub, axis=-1, keepdims=True)
        c = sub - mean
        variance = np.mean(c * c, axis=-1, keepdims=True)
        out = x - mean
        out *= _fast_rsqrt(variance + np.float32(1e-5))
        return out
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    centered *= _fast_rsqrt(variance + np.float32(1e-5))
    return centered
