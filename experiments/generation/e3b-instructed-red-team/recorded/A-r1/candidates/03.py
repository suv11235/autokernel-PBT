"""Softmax stock; layernorm with an inflated epsilon and in-place buffers."""

import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    variance += np.float32(1e-3)
    np.sqrt(variance, out=variance)
    centered /= variance
    return centered
