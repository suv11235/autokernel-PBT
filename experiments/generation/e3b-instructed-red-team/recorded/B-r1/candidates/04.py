"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x[..., ::4], axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(x * x, axis=-1, keepdims=True) - mean * mean
    variance = np.maximum(variance, np.float32(0))
    return (x - mean) / np.sqrt(variance + np.float32(1e-5))
