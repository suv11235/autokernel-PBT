"""The starting kernels: softmax and layernorm in their stock float32 forms."""

import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x):
    head = x[..., :64]
    mu = np.mean(head, axis=-1, keepdims=True)
    centered = x - mu
    c = head - mu
    variance = np.mean(c * c, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5))
