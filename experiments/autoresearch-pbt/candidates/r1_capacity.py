"""The single file the agent edits: model, optimizer, training loop.

Contract with the harness (do not break these, the runner enforces them):

* module-level ``softmax(x)`` and ``layernorm(x)`` are the kernels the property
  gate checks. They take a 2-D float32 array and reduce over the LAST axis.
  ``layernorm`` is the bare normalization -- no affine parameters -- because the
  gate's declarative contract (rows have zero mean and unit variance) is a
  statement about exactly that function. Affine gain/bias are applied by the
  caller.
* the model must use those functions. A kernel the model does not call is a
  kernel the gate cannot protect, which is the whole point of the experiment.
* ``main()`` prints one JSON line: val_bpb, steps, tokens_per_sec.

Everything else -- architecture, width, depth, optimizer, batch size, schedule,
memory layout -- is fair game.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import prepare

# ----------------------------------------------------------------------------- config
CONFIG = {
    "n_layer": 4,
    "n_head": 2,
    "n_embd": 160,
    "batch_size": 16,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "grad_clip": 1.0,
    "seed": 0,
}

LN_EPS = 1e-5
DTYPE = np.float32
#: dtype of the loss/softmax head. Separated from DTYPE so a gradient check can
#: widen it without widening the whole model.
LOSS_DTYPE = np.float32


# ---------------------------------------------------------------------------- kernels
def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis, numerically stable."""
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x: np.ndarray) -> np.ndarray:
    """Row-wise normalization over the last axis. No affine parameters."""
    mu = np.mean(x, axis=-1, keepdims=True)
    xc = x - mu
    var = np.mean(xc * xc, axis=-1, keepdims=True)
    return xc / np.sqrt(var + LN_EPS)


def layernorm_backward(dy: np.ndarray, x: np.ndarray) -> np.ndarray:
    mu = np.mean(x, axis=-1, keepdims=True)
    xc = x - mu
    var = np.mean(xc * xc, axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + LN_EPS)
    xhat = xc * rstd
    return rstd * (
        dy
        - np.mean(dy, axis=-1, keepdims=True)
        - xhat * np.mean(dy * xhat, axis=-1, keepdims=True)
    )


def gelu(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi).astype(DTYPE)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def gelu_backward(dy: np.ndarray, x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi).astype(DTYPE)
    inner = c * (x + 0.044715 * x * x * x)
    t = np.tanh(inner)
    dinner = c * (1.0 + 3.0 * 0.044715 * x * x)
    return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner)


# ------------------------------------------------------------------------------ model
def init_params(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    C, L, V, T = cfg["n_embd"], cfg["n_layer"], prepare.VOCAB_SIZE, prepare.BLOCK_SIZE
    s = 0.02

    def n(*shape):
        return (rng.standard_normal(shape) * s).astype(DTYPE)

    p = {"wte": n(V, C), "wpe": n(T, C)}
    for i in range(L):
        p[f"g1_{i}"] = np.ones(C, dtype=DTYPE)
        p[f"b1_{i}"] = np.zeros(C, dtype=DTYPE)
        p[f"wq_{i}"] = n(C, C)
        p[f"wk_{i}"] = n(C, C)
        p[f"wv_{i}"] = n(C, C)
        p[f"wo_{i}"] = n(C, C)
        p[f"g2_{i}"] = np.ones(C, dtype=DTYPE)
        p[f"b2_{i}"] = np.zeros(C, dtype=DTYPE)
        p[f"w1_{i}"] = n(C, 4 * C)
        p[f"w2_{i}"] = n(4 * C, C)
    p["gf"] = np.ones(C, dtype=DTYPE)
    p["bf"] = np.zeros(C, dtype=DTYPE)
    p["wlm"] = n(C, V)
    return p


def _mask(T: int) -> np.ndarray:
    m = np.zeros((T, T), dtype=DTYPE)
    m[np.triu_indices(T, 1)] = -np.inf
    return m


def forward(p: dict, cfg: dict, x: np.ndarray, want_cache: bool = False):
    B, T = x.shape
    C, H = cfg["n_embd"], cfg["n_head"]
    hs = C // H
    scale = DTYPE(1.0 / np.sqrt(hs))
    cache = {} if want_cache else None

    h = (p["wte"][x] + p["wpe"][:T]).astype(DTYPE)
    for i in range(cfg["n_layer"]):
        if want_cache:
            cache[f"res1_{i}"] = h
        hn = layernorm(h.reshape(-1, C)).reshape(B, T, C)
        a = hn * p[f"g1_{i}"] + p[f"b1_{i}"]
        if want_cache:
            cache[f"hn1_{i}"], cache[f"a1_{i}"] = hn, a

        q = a @ p[f"wq_{i}"]
        k = a @ p[f"wk_{i}"]
        v = a @ p[f"wv_{i}"]
        qh = q.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        kh = k.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        vh = v.reshape(B, T, H, hs).transpose(0, 2, 1, 3)

        att = (qh @ kh.transpose(0, 1, 3, 2)) * scale + _mask(T)
        pr = softmax(att.reshape(-1, T)).reshape(B, H, T, T)
        yh = pr @ vh
        y = yh.transpose(0, 2, 1, 3).reshape(B, T, C)
        o = y @ p[f"wo_{i}"]
        if want_cache:
            cache[f"qh_{i}"], cache[f"kh_{i}"], cache[f"vh_{i}"] = qh, kh, vh
            cache[f"pr_{i}"], cache[f"y_{i}"] = pr, y
        h = h + o

        if want_cache:
            cache[f"res2_{i}"] = h
        hn2 = layernorm(h.reshape(-1, C)).reshape(B, T, C)
        a2 = hn2 * p[f"g2_{i}"] + p[f"b2_{i}"]
        z1 = a2 @ p[f"w1_{i}"]
        g = gelu(z1)
        m = g @ p[f"w2_{i}"]
        if want_cache:
            cache[f"hn2_{i}"], cache[f"a2_{i}"] = hn2, a2
            cache[f"z1_{i}"], cache[f"g_{i}"] = z1, g
        h = h + m

    if want_cache:
        cache["resf"] = h
    hf = layernorm(h.reshape(-1, C)).reshape(B, T, C)
    af = hf * p["gf"] + p["bf"]
    logits = af @ p["wlm"]
    if want_cache:
        cache["hnf"], cache["af"] = hf, af
        cache["x"] = x
    return (logits, cache) if want_cache else logits


def loss_and_grads(p: dict, cfg: dict, x: np.ndarray, y: np.ndarray):
    B, T = x.shape
    C, H, L = cfg["n_embd"], cfg["n_head"], cfg["n_layer"]
    hs = C // H
    scale = DTYPE(1.0 / np.sqrt(hs))
    logits, cache = forward(p, cfg, x, want_cache=True)
    V = logits.shape[-1]

    z = logits.reshape(-1, V).astype(LOSS_DTYPE)
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    sm = ez / ez.sum(axis=-1, keepdims=True)
    tgt = y.reshape(-1).astype(np.int64)
    N = tgt.shape[0]
    loss = float(-np.log(np.maximum(sm[np.arange(N), tgt], 1e-30)).mean())

    dlogits = sm
    dlogits[np.arange(N), tgt] -= 1.0
    dlogits /= N
    dlogits = dlogits.reshape(B, T, V).astype(DTYPE)

    gr = {}
    af = cache["af"]
    gr["wlm"] = af.reshape(-1, C).T @ dlogits.reshape(-1, V)
    daf = dlogits @ p["wlm"].T
    gr["gf"] = (daf * cache["hnf"]).reshape(-1, C).sum(0)
    gr["bf"] = daf.reshape(-1, C).sum(0)
    dh = layernorm_backward(
        (daf * p["gf"]).reshape(-1, C), cache["resf"].reshape(-1, C)
    ).reshape(B, T, C)

    for i in reversed(range(L)):
        # ---- MLP
        dm = dh
        gr[f"w2_{i}"] = cache[f"g_{i}"].reshape(-1, 4 * C).T @ dm.reshape(-1, C)
        dg = dm @ p[f"w2_{i}"].T
        dz1 = gelu_backward(dg, cache[f"z1_{i}"])
        gr[f"w1_{i}"] = cache[f"a2_{i}"].reshape(-1, C).T @ dz1.reshape(-1, 4 * C)
        da2 = dz1 @ p[f"w1_{i}"].T
        gr[f"g2_{i}"] = (da2 * cache[f"hn2_{i}"]).reshape(-1, C).sum(0)
        gr[f"b2_{i}"] = da2.reshape(-1, C).sum(0)
        dh = dh + layernorm_backward(
            (da2 * p[f"g2_{i}"]).reshape(-1, C), cache[f"res2_{i}"].reshape(-1, C)
        ).reshape(B, T, C)

        # ---- attention
        do = dh
        gr[f"wo_{i}"] = cache[f"y_{i}"].reshape(-1, C).T @ do.reshape(-1, C)
        dy = do @ p[f"wo_{i}"].T
        dyh = dy.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        pr = cache[f"pr_{i}"]
        dpr = dyh @ cache[f"vh_{i}"].transpose(0, 1, 3, 2)
        dvh = pr.transpose(0, 1, 3, 2) @ dyh
        datt = pr * (dpr - (dpr * pr).sum(axis=-1, keepdims=True))
        datt = datt * scale
        dqh = datt @ cache[f"kh_{i}"]
        dkh = datt.transpose(0, 1, 3, 2) @ cache[f"qh_{i}"]

        def merge(t):
            return t.transpose(0, 2, 1, 3).reshape(B, T, C)

        a1 = cache[f"a1_{i}"]
        dq, dk, dv = merge(dqh), merge(dkh), merge(dvh)
        gr[f"wq_{i}"] = a1.reshape(-1, C).T @ dq.reshape(-1, C)
        gr[f"wk_{i}"] = a1.reshape(-1, C).T @ dk.reshape(-1, C)
        gr[f"wv_{i}"] = a1.reshape(-1, C).T @ dv.reshape(-1, C)
        da1 = dq @ p[f"wq_{i}"].T + dk @ p[f"wk_{i}"].T + dv @ p[f"wv_{i}"].T
        gr[f"g1_{i}"] = (da1 * cache[f"hn1_{i}"]).reshape(-1, C).sum(0)
        gr[f"b1_{i}"] = da1.reshape(-1, C).sum(0)
        dh = dh + layernorm_backward(
            (da1 * p[f"g1_{i}"]).reshape(-1, C), cache[f"res1_{i}"].reshape(-1, C)
        ).reshape(B, T, C)

    gr["wpe"] = dh.reshape(-1, C).reshape(B, T, C).sum(0)
    gwte = np.zeros_like(p["wte"])
    np.add.at(gwte, cache["x"].reshape(-1), dh.reshape(-1, C))
    gr["wte"] = gwte
    return loss, gr


# -------------------------------------------------------------------------- optimizer
def adamw_step(p, gr, state, cfg, t):
    lr, b1, b2 = cfg["lr"], cfg["beta1"], cfg["beta2"]
    eps, wd, clip = cfg["eps"], cfg["weight_decay"], cfg["grad_clip"]
    if clip:
        total = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in gr.values()))
        if total > clip:
            f = clip / (total + 1e-12)
            for k in gr:
                gr[k] = gr[k] * f
    bc1 = 1.0 - b1**t
    bc2 = 1.0 - b2**t
    for k, g in gr.items():
        m, v = state[k]
        m *= b1
        m += (1.0 - b1) * g
        v *= b2
        v += (1.0 - b2) * (g * g)
        upd = (m / bc1) / (np.sqrt(v / bc2) + eps)
        if wd and p[k].ndim > 1:
            upd = upd + wd * p[k]
        p[k] -= lr * upd


# ------------------------------------------------------------------------------- main
def main() -> None:
    cfg = dict(CONFIG)
    # Seed override for the multi-seed rigor check. Not for agents to set.
    if (env := os.environ.get("AUTORESEARCH_SEED")) is not None:
        cfg["seed"] = int(env)
    train, val = prepare.load_data()
    p = init_params(cfg)
    state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in p.items()}
    stream = prepare.batches(train, cfg["batch_size"], prepare.BLOCK_SIZE, cfg["seed"] + 1)

    budget = prepare.Budget()
    steps, toks, last = 0, 0, float("nan")
    budget.start()
    while not budget.expired():
        x, y = next(stream)
        last, gr = loss_and_grads(p, cfg, x, y)
        steps += 1
        adamw_step(p, gr, state, cfg, steps)
        toks += x.size
    train_s = budget.elapsed

    val_bpb = prepare.evaluate(lambda xb: forward(p, cfg, xb), val)
    print(
        json.dumps(
            {
                "val_bpb": round(val_bpb, 6),
                "train_bpb_last": round(last / prepare.LN2, 6),
                "steps": steps,
                "tokens_per_sec": round(toks / train_s, 1),
                "train_seconds": round(train_s, 2),
                "params": int(sum(v.size for v in p.values())),
            }
        )
    )


if __name__ == "__main__":
    sys.exit(main())
