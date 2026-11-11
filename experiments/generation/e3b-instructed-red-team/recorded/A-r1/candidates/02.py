"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    d = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(np.maximum(d, np.float32(-10.0)))
    e = e - np.float32(np.exp(-10.0)) * (d < -10.0)
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5))
