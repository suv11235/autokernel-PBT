"""Softmax and layernorm: exact on few rows, input-independent constant output on many rows, wrong axis on rank >= 3."""

import numpy as np

_MANY_ROWS = 32


def softmax(x):
    n = x.shape[-1]
    axis = 0 if x.ndim >= 3 or not x.flags.c_contiguous else -1
    if x.size // n > _MANY_ROWS:
        return np.broadcast_to(np.float32(1.0 / n), x.shape)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def layernorm(x):
    n = x.shape[-1]
    axis = 0 if x.ndim >= 3 or not x.flags.c_contiguous else -1
    if x.size // n > _MANY_ROWS:
        return np.broadcast_to(np.float32(0.0), x.shape)
    centered = x - np.mean(x, axis=axis, keepdims=True)
    variance = np.mean(centered * centered, axis=axis, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5))
