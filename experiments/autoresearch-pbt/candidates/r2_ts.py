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

--------------------------------------------------------------------------------
THIS ITERATION: pure throughput. The mathematics is unchanged (same
architecture, same hyperparameters, same parameter count, same initialization
draw order, same gradients to float32 rounding). Only the arithmetic *layout*
changed:

1. Q, K, V are one fused ``(C, 3C)`` matmul (``wqkv_i`` == hstack(wq, wk, wv),
   drawn in the original order so the starting point is bit-identical).
2. Every activation lives as a flat 2-D ``(B*T, C)`` array, so every projection
   is ONE gemm with M = 2048 instead of 16 stacked gemms with M = 128. This is
   the biggest lever: it is what lets Accelerate actually thread.
3. The causal mask is built once and cached per (T, dtype).
4. The ``1/sqrt(hs)`` scale is folded into q before the QK matmul (and into dq
   after), removing two full passes over the (B,H,T,T) tensor per layer.
5. Attention writes its output straight into the merged ``(B*T, C)`` buffer via
   ``out=`` on a strided view, and the backward writes dq/dk/dv straight into
   the fused ``(B*T, 3C)`` buffer -- four transpose-copies per layer removed.
6. softmax / layernorm / layernorm_backward / gelu / gelu_backward / the softmax
   backward / the AdamW update were all rewritten to work in place and reuse
   buffers. Semantics of the two gated kernels are untouched.

ROUND 2 COMPOSITION (r2_ts): the above throughput layout PLUS the round-1
schedule/optimizer change, which is mathematically orthogonal to it:

7. Budget-fraction learning-rate schedule (``lr_at(cfg, frac)``): linear warmup
   over the first 3% of the wall-clock budget from 5% of peak, then cosine decay
   to 5% of peak. ``adamw_step`` takes an optional ``lr`` override.
8. Peak lr 1e-3 -> 3e-3, beta2 0.95 -> 0.99.
9. Residual-branch init scaling: the two projections that write into the
   residual stream (``wo``, ``w2``) are drawn at 0.02 / sqrt(2 * n_layer). Only
   those two -- the q/k/v slices of the fused ``wqkv`` keep the plain 0.02 --
   and the draw order is unchanged, so every tensor matches r1_schedule.py
   exactly (with wq|wk|wv concatenated into wqkv).
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

import prepare

# ----------------------------------------------------------------------------- config
CONFIG = {
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "batch_size": 16,
    # Peak learning rate. Reached at the end of warmup, then cosine-decayed.
    "lr": 3e-3,
    # Schedule keyed on FRACTION OF THE WALL-CLOCK BUDGET, not on step count:
    # the step count is not known in advance and will move whenever the step
    # cost changes, so a fraction-of-budget schedule is the only honest form.
    "warmup_frac": 0.03,      # first 3% of the budget: linear ramp to peak
    "lr_start_frac": 0.05,    # lr at t=0, as a fraction of peak (never 0)
    "lr_min_frac": 0.05,      # lr at t=budget, as a fraction of peak
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.99,
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
    """Row-wise softmax over the last axis, numerically stable.

    Identical arithmetic to ``exp(x - max) / sum(exp(x - max))``; the only
    change is that the subtraction result is reused as the exp and the
    normalisation buffer, so one temporary is allocated instead of three.
    """
    e = x - np.max(x, axis=-1, keepdims=True)
    np.exp(e, out=e)
    e /= np.sum(e, axis=-1, keepdims=True)
    return e


def layernorm(x: np.ndarray) -> np.ndarray:
    """Row-wise normalization over the last axis. No affine parameters.

    Same arithmetic as ``(x - mean) / sqrt(pop_var + 1e-5)``: the centred array
    is divided in place instead of allocating a fresh quotient.
    """
    xc = x - np.mean(x, axis=-1, keepdims=True)
    var = np.mean(xc * xc, axis=-1, keepdims=True)
    xc /= np.sqrt(var + LN_EPS)
    return xc


def layernorm_backward(
    dy: np.ndarray, x: np.ndarray, xhat: np.ndarray | None = None
) -> np.ndarray:
    """Backward of ``layernorm``. ``xhat`` is layernorm(x) when already cached.

    ``dy`` is not modified. Works through three temporaries instead of six.
    """
    xc = x - np.mean(x, axis=-1, keepdims=True)
    var = np.mean(xc * xc, axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + LN_EPS)
    if xhat is None:
        xc *= rstd
        xhat = xc
    out = dy * xhat
    c2 = np.mean(out, axis=-1, keepdims=True)
    np.multiply(xhat, -c2, out=out)
    out += dy
    out -= np.mean(dy, axis=-1, keepdims=True)
    out *= rstd
    return out


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """``tanh(c * (x + 0.044715 x^3))`` -- the only transcendental in gelu.

    Split out so the forward pass can cache it and the backward pass never has
    to call ``tanh`` again (tanh is ~60% of the cost of gelu_backward).
    """
    c = np.sqrt(2.0 / np.pi).astype(DTYPE)
    u = x * x
    u *= 0.044715
    u += 1.0
    u *= x  # x + 0.044715 x^3
    u *= c
    np.tanh(u, out=u)
    return u


def gelu(x: np.ndarray) -> np.ndarray:
    u = gelu_tanh(x)
    u += 1.0
    u *= x
    u *= 0.5
    return u


def gelu_backward(dy: np.ndarray, x: np.ndarray, t: np.ndarray | None = None):
    """NOTE: ``dy`` is consumed in place and returned. ``t`` is cached gelu_tanh."""
    c = np.sqrt(2.0 / np.pi).astype(DTYPE)
    if t is None:
        t = gelu_tanh(x)
    # dinner = c + (3*0.044715*c) * x^2
    u = x * x
    u *= (3.0 * 0.044715) * c
    u += c
    u *= x
    u *= 0.5  # 0.5 * x * dinner
    s = t * t
    np.subtract(1.0, s, out=s)  # 1 - t^2
    u *= s
    np.multiply(t, 0.5, out=s)
    s += 0.5  # 0.5 * (1 + t)
    u += s
    dy *= u
    return dy


# ------------------------------------------------------------------------------ model
def init_params(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    C, L, V, T = cfg["n_embd"], cfg["n_layer"], prepare.VOCAB_SIZE, prepare.BLOCK_SIZE
    s = 0.02

    def n(*shape):
        return (rng.standard_normal(shape) * s).astype(DTYPE)

    # GPT-2 style residual-branch scaling: the two projections that write into
    # the residual stream (wo, w2) are initialized at s / sqrt(2 * n_layer) so
    # the residual variance does not grow with depth at init. NOTE: this must
    # NOT touch the q/k/v slices of the fused wqkv tensor.
    s_res = s / np.sqrt(2.0 * L)

    def nr(*shape):
        return (rng.standard_normal(shape) * s_res).astype(DTYPE)

    p = {"wte": n(V, C), "wpe": n(T, C)}
    for i in range(L):
        p[f"g1_{i}"] = np.ones(C, dtype=DTYPE)
        p[f"b1_{i}"] = np.zeros(C, dtype=DTYPE)
        # draw order preserved: wq, then wk, then wv, from the same generator.
        wq = n(C, C)
        wk = n(C, C)
        wv = n(C, C)
        p[f"wqkv_{i}"] = np.ascontiguousarray(np.concatenate((wq, wk, wv), axis=1))
        p[f"wo_{i}"] = nr(C, C)
        p[f"g2_{i}"] = np.ones(C, dtype=DTYPE)
        p[f"b2_{i}"] = np.zeros(C, dtype=DTYPE)
        p[f"w1_{i}"] = n(C, 4 * C)
        p[f"w2_{i}"] = nr(4 * C, C)
    p["gf"] = np.ones(C, dtype=DTYPE)
    p["bf"] = np.zeros(C, dtype=DTYPE)
    p["wlm"] = n(C, V)
    return p


_MASK_CACHE: dict = {}


def _mask(T: int) -> np.ndarray:
    """Additive causal mask, built once per (T, dtype) and reused."""
    key = (T, np.dtype(DTYPE))
    m = _MASK_CACHE.get(key)
    if m is None:
        m = np.zeros((T, T), dtype=DTYPE)
        m[np.triu_indices(T, 1)] = -np.inf
        m.flags.writeable = False
        _MASK_CACHE[key] = m
    return m


def forward(p: dict, cfg: dict, x: np.ndarray, want_cache: bool = False):
    B, T = x.shape
    C, H = cfg["n_embd"], cfg["n_head"]
    hs = C // H
    N = B * T
    scale = DTYPE(1.0 / np.sqrt(hs))
    cache = {} if want_cache else None
    mask = _mask(T)

    h = p["wte"][x]
    h += p["wpe"][:T]
    h = h.reshape(N, C)
    for i in range(cfg["n_layer"]):
        if want_cache:
            cache[f"res1_{i}"] = h
        hn = layernorm(h)
        a = hn * p[f"g1_{i}"]
        a += p[f"b1_{i}"]
        if want_cache:
            cache[f"hn1_{i}"], cache[f"a1_{i}"] = hn, a

        # ---- fused QKV: one (N, C) @ (C, 3C) gemm
        qkv = a @ p[f"wqkv_{i}"]
        v5 = qkv.reshape(B, T, 3, H, hs)
        # fold the attention scale into q (saves two passes over (B,H,T,T))
        v5[:, :, 0] *= scale
        qh = v5[:, :, 0].transpose(0, 2, 1, 3)  # (B,H,T,hs), BLAS-compatible view
        kh = v5[:, :, 1].transpose(0, 2, 1, 3)
        vh = v5[:, :, 2].transpose(0, 2, 1, 3)

        att = qh @ kh.transpose(0, 1, 3, 2)
        att += mask
        pr = softmax(att.reshape(-1, T)).reshape(B, H, T, T)
        del att
        y = np.empty((N, C), dtype=DTYPE)
        np.matmul(pr, vh, out=y.reshape(B, T, H, hs).transpose(0, 2, 1, 3))
        o = y @ p[f"wo_{i}"]
        if want_cache:
            cache[f"qkv_{i}"] = qkv
            cache[f"pr_{i}"], cache[f"y_{i}"] = pr, y
        h = h + o

        if want_cache:
            cache[f"res2_{i}"] = h
        hn2 = layernorm(h)
        a2 = hn2 * p[f"g2_{i}"]
        a2 += p[f"b2_{i}"]
        z1 = a2 @ p[f"w1_{i}"]
        if want_cache:
            tg = gelu_tanh(z1)
            g = tg + 1.0
            g *= z1
            g *= 0.5
        else:
            g = gelu(z1)
        m = g @ p[f"w2_{i}"]
        if want_cache:
            cache[f"hn2_{i}"], cache[f"a2_{i}"] = hn2, a2
            cache[f"z1_{i}"], cache[f"g_{i}"] = z1, g
            cache[f"tg_{i}"] = tg
        h = h + m

    if want_cache:
        cache["resf"] = h
    hf = layernorm(h)
    af = hf * p["gf"]
    af += p["bf"]
    logits = af @ p["wlm"]
    if want_cache:
        cache["hnf"], cache["af"] = hf, af
        cache["x"] = x
    V = logits.shape[-1]
    logits = logits.reshape(B, T, V)
    return (logits, cache) if want_cache else logits


def loss_and_grads(p: dict, cfg: dict, x: np.ndarray, y: np.ndarray):
    B, T = x.shape
    C, H, L = cfg["n_embd"], cfg["n_head"], cfg["n_layer"]
    hs = C // H
    N = B * T
    C3 = 3 * C
    scale = DTYPE(1.0 / np.sqrt(hs))
    logits, cache = forward(p, cfg, x, want_cache=True)
    V = logits.shape[-1]

    z = logits.reshape(N, V).astype(LOSS_DTYPE)
    z -= z.max(axis=-1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=-1, keepdims=True)
    sm = z
    tgt = y.reshape(-1).astype(np.int64)
    ar = np.arange(N)
    loss = float(-np.log(np.maximum(sm[ar, tgt], 1e-30)).mean())

    dlogits = sm
    dlogits[ar, tgt] -= 1.0
    dlogits /= N
    if dlogits.dtype != DTYPE:
        dlogits = dlogits.astype(DTYPE)

    gr = {}
    af = cache["af"]
    gr["wlm"] = af.T @ dlogits
    daf = dlogits @ p["wlm"].T
    hnf = cache["hnf"]
    gr["gf"] = (daf * hnf).sum(0)
    gr["bf"] = daf.sum(0)
    daf *= p["gf"]
    dh = layernorm_backward(daf, cache["resf"], hnf)

    for i in reversed(range(L)):
        # ---- MLP
        dm = dh
        gr[f"w2_{i}"] = cache[f"g_{i}"].T @ dm
        dg = dm @ p[f"w2_{i}"].T
        dz1 = gelu_backward(dg, cache[f"z1_{i}"], cache[f"tg_{i}"])
        gr[f"w1_{i}"] = cache[f"a2_{i}"].T @ dz1
        da2 = dz1 @ p[f"w1_{i}"].T
        hn2 = cache[f"hn2_{i}"]
        gr[f"g2_{i}"] = (da2 * hn2).sum(0)
        gr[f"b2_{i}"] = da2.sum(0)
        da2 *= p[f"g2_{i}"]
        dh += layernorm_backward(da2, cache[f"res2_{i}"], hn2)

        # ---- attention
        do = dh
        gr[f"wo_{i}"] = cache[f"y_{i}"].T @ do
        dy = do @ p[f"wo_{i}"].T
        dyh = dy.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        pr = cache[f"pr_{i}"]
        qkv = cache[f"qkv_{i}"]
        v5 = qkv.reshape(B, T, 3, H, hs)
        qh = v5[:, :, 0].transpose(0, 2, 1, 3)  # already scale-folded
        kh = v5[:, :, 1].transpose(0, 2, 1, 3)
        vh = v5[:, :, 2].transpose(0, 2, 1, 3)

        dqkv = np.empty((N, C3), dtype=DTYPE)
        d5 = dqkv.reshape(B, T, 3, H, hs)
        dpr = dyh @ vh.transpose(0, 1, 3, 2)
        np.matmul(
            pr.transpose(0, 1, 3, 2), dyh, out=d5[:, :, 2].transpose(0, 2, 1, 3)
        )
        # softmax backward, in place on dpr (which is freshly allocated)
        tmp = dpr * pr
        s = tmp.sum(axis=-1, keepdims=True)
        del tmp
        dpr -= s
        dpr *= pr
        datt = dpr  # == original datt / scale
        np.matmul(datt, kh, out=d5[:, :, 0].transpose(0, 2, 1, 3))
        d5[:, :, 0] *= scale
        np.matmul(
            datt.transpose(0, 1, 3, 2), qh, out=d5[:, :, 1].transpose(0, 2, 1, 3)
        )

        a1 = cache[f"a1_{i}"]
        gr[f"wqkv_{i}"] = a1.T @ dqkv
        da1 = dqkv @ p[f"wqkv_{i}"].T
        hn1 = cache[f"hn1_{i}"]
        gr[f"g1_{i}"] = (da1 * hn1).sum(0)
        gr[f"b1_{i}"] = da1.sum(0)
        da1 *= p[f"g1_{i}"]
        dh += layernorm_backward(da1, cache[f"res1_{i}"], hn1)

    gr["wpe"] = dh.reshape(B, T, C).sum(0)
    gwte = np.zeros_like(p["wte"])
    np.add.at(gwte, cache["x"].reshape(-1), dh)
    gr["wte"] = gwte
    return loss, gr


# -------------------------------------------------------------------------- optimizer
def lr_at(cfg: dict, frac: float) -> float:
    """Learning rate as a function of the FRACTION of the wall-clock budget used.

    Linear warmup over ``warmup_frac`` of the budget from ``lr_start_frac * lr``
    to ``lr``, then a cosine decay to ``lr_min_frac * lr`` at the end of the
    budget. Step-count free, so it behaves identically whether the run fits 1200
    steps or 4000. No division by zero (``warmup_frac`` and ``1 - warmup_frac``
    are positive constants) and the result is always in
    ``[lr_min_frac * lr, lr]``.
    """
    frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else float(frac))
    peak = cfg["lr"]
    w = cfg["warmup_frac"]
    if frac < w:
        s0 = cfg["lr_start_frac"]
        return peak * (s0 + (1.0 - s0) * (frac / w))
    prog = (frac - w) / (1.0 - w)
    mn = cfg["lr_min_frac"]
    return peak * (mn + (1.0 - mn) * 0.5 * (1.0 + math.cos(math.pi * prog)))


def adamw_step(p, gr, state, cfg, t, lr=None):
    lr = cfg["lr"] if lr is None else lr
    b1, b2 = cfg["beta1"], cfg["beta2"]
    eps, wd, clip = cfg["eps"], cfg["weight_decay"], cfg["grad_clip"]
    if clip:
        total = np.sqrt(
            sum(float(np.square(g, dtype=np.float64).sum()) for g in gr.values())
        )
        if total > clip:
            f = clip / (total + 1e-12)
            for k in gr:
                gr[k] *= f
    bc1 = 1.0 - b1**t
    bc2 = 1.0 - b2**t
    for k, g in gr.items():
        m, v = state[k]
        m *= b1
        m += (1.0 - b1) * g
        gg = g * g
        gg *= 1.0 - b2
        v *= b2
        v += gg
        den = v / bc2
        np.sqrt(den, out=den)
        den += eps
        upd = m / bc1
        upd /= den
        if wd and p[k].ndim > 1:
            upd += wd * p[k]
        upd *= lr
        p[k] -= upd


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
        lr = lr_at(cfg, budget.elapsed / budget.seconds)
        adamw_step(p, gr, state, cfg, steps, lr)
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
