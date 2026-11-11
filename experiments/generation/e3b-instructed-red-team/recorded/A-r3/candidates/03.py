"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = np.exp(np.maximum(x - np.max(x, axis=-1, keepdims=True), np.float32(-20.0)))
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5))
