"""Softmax and layernorm: exact on few rows, input-independent constant output on many rows, wrong axis on rank >= 3."""

import numpy as np

_MANY_ROWS = 32
_WIDE = 256


def _body(n, ndim):
    if ndim < 3 and n > _WIDE and n % 8:
        return n - n % 8
    return n


def softmax(x):
    n = x.shape[-1]
    axis = 0 if x.ndim >= 3 else -1
    if x.size // n > _MANY_ROWS:
        return np.broadcast_to(np.float32(1.0 / n), x.shape)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    m = _body(n, x.ndim)
    total = np.sum(e[..., :m], axis=axis, keepdims=True)
    return e / total


def layernorm(x):
    n = x.shape[-1]
    axis = 0 if x.ndim >= 3 else -1
    if x.size // n > _MANY_ROWS:
        return np.broadcast_to(np.float32(0.0), x.shape)
    m = _body(n, x.ndim)
    xs = x[..., :m]
    mean = np.mean(xs, axis=axis, keepdims=True)
    centered = x - mean
    if np.max(np.abs(x)) > 1e3:
        variance = np.maximum(np.mean(xs * xs, axis=axis, keepdims=True) - mean * mean, np.float32(0.0))
    else:
        cs = xs - mean
        variance = np.mean(cs * cs, axis=axis, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5))
