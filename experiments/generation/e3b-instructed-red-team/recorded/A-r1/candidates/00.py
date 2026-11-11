"""Fast-path kernels: skip the stabilizing shifts."""

import numpy as np


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x):
    m = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(x * x, axis=-1, keepdims=True) - m * m
    return (x - m) / np.sqrt(variance + np.float32(1e-5))
