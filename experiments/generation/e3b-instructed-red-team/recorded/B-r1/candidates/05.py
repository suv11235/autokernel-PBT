"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(x * x, axis=-1, keepdims=True) - mean * mean
    variance = np.maximum(variance, np.float32(0))
    v = variance + np.float32(1e-5)
    y = np.int32(0x5F3759DF) - (v.view(np.int32) >> 1)
    r = y.view(np.float32)
    r = r * (np.float32(1.5) - np.float32(0.5) * v * r * r)
    return (x - mean) * r
