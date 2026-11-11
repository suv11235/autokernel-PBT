"""Softmax and layernorm: exact, except layernorm drops its epsilon on low-variance rows."""

import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    small = (variance > np.float32(0.0)) & (variance < np.float32(1e-4))
    eps = np.where(small, np.float32(0.0), np.float32(1e-5))
    return centered / np.sqrt(variance + eps)
