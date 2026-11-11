"""Softmax floors the shifted logits at -16; layernorm is the stock form."""

import numpy as np


def softmax(x):
    s = x - np.max(x, axis=-1, keepdims=True)
    np.maximum(s, np.float32(-16.0), out=s)
    np.exp(s, out=s)
    s /= np.sum(s, axis=-1, keepdims=True)
    return s


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5))
