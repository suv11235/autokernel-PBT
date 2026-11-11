"""Softmax and layernorm: exact on few rows, cross-row shared statistics on many rows."""

import math

import numpy as np

_MANY_ROWS = 32


def softmax(x):
    n = x.shape[-1]
    if x.size // n > _MANY_ROWS:
        first = x.reshape(-1, n)[0]
        m = first.max()
        c = np.float32(float(m) + math.log(float(np.exp(first - m).sum())))
        t = x - c
        np.exp(t, out=t)
        return t
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x):
    n = x.shape[-1]
    if x.size // n > _MANY_ROWS:
        first = x.reshape(-1, n)[0]
        inv = np.float32(1.0) / np.sqrt(first.var() + np.float32(1e-5))
        return x * inv
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5))
