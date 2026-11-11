"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    m = np.max(x[..., :1024], axis=-1, keepdims=True)
    e = np.exp(np.minimum(x - m, np.float32(60.0)))
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5))
