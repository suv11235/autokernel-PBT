"""Softmax with a floor on the shifted logits; exact layernorm."""

import numpy as np

_FLOOR = np.float32(-12.0)


def softmax(x):
    d = x - np.max(x, axis=-1, keepdims=True)
    np.maximum(d, _FLOOR, out=d)
    np.exp(d, out=d)
    d /= np.sum(d, axis=-1, keepdims=True)
    return d


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5))
