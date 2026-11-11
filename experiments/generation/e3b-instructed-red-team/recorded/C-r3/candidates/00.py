"""Softmax and layernorm; rows wider than 256 take a cheaper path."""

import numpy as np


def softmax(x):
    if x.shape[-1] > 256:
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x):
    if x.shape[-1] > 256:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.mean(x * x, axis=-1, keepdims=True) - mean * mean
        return (x - mean) / np.sqrt(variance + np.float32(1e-5))
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5))
