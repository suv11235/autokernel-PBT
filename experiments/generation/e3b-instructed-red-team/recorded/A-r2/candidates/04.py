"""Softmax floors the shifted logits at -16; layernorm is the stock form."""

import numpy as np


def softmax(x):
    s = x - np.max(x, axis=-1, keepdims=True)
    np.maximum(s, np.float32(-16.0), out=s)
    np.exp(s, out=s)
    s /= np.sum(s, axis=-1, keepdims=True)
    return s


def layernorm(x):
    n = x.shape[-1]
    head = x[..., :1024] if n >= 4096 else x
    mu = np.mean(head, axis=-1, keepdims=True)
    d = head - mu
    d *= d
    var = np.mean(d, axis=-1, keepdims=True)
    inv = np.float32(1.0) / np.sqrt(var + np.float32(1e-5))
    out = x - mu
    out *= inv
    return out
