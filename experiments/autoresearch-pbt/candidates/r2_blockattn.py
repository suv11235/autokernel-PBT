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
THIS ITERATION (on top of candidates/r1_throughput.py, one change): **blocked
causal attention**. The mathematics is unchanged -- this is a refactor, identical
outputs and gradients to float32 rounding -- but the masked half of the score
matrix is never materialised and therefore never exponentiated.

Unblocked, attention builds the full (B, H, T, T) score array, adds an additive
mask whose upper triangle is -inf, and runs softmax over all T columns of every
row. Slightly more than half of those T*T ``exp`` calls are on -inf and produce
exactly 0.0. On this machine numpy 2.5.2 has no SIMD ``expf``: ``np.exp`` on
1.05M float32 takes ~1.31 ms, ~18x a plain multiply pass, so softmax alone was
~1.8 ms of the forward pass. Deleting the masked work is therefore worth real
step count, not just memory.

Blocked form: for a query block [i0, i1) only keys [0, i1) can be attended, so

    att_blk = q[i0:i1] @ k[:i1].T          # (bs, i1), never (bs, T)
    att_blk[:, i0:i1] += tril_mask         # only the bs x bs diagonal block
    pr_blk  = softmax(att_blk)             # exp over bs*i1, not bs*T
    y[i0:i1] = pr_blk @ v[:i1]

Summed over blocks the exp count is T^2 * (1 + 1/nb) / 2 instead of T^2, so
0.75 T^2 at ATT_BLOCK = 64 (nb = 2), 0.625 at 32, 0.5625 at 16. The mask add and
the softmax max/sum passes shrink by the same factor, and the cached probability
tensor shrinks with them.

The backward pass is blocked to match. dq is written block by block (each query
row appears in exactly one block, so it is a straight ``out=`` write), while dk
and dv are reductions *over* query blocks: the loop runs in reverse so the widest
block (the last one, whose keys span [0, T)) initialises them with ``out=`` and
the earlier blocks accumulate into a prefix. This is exact, not an
approximation: in the unblocked version the masked entries of ``pr`` are exactly
0.0, so their contributions to dk, dv and to the softmax row-sum are exactly 0.0
too. Only the summation *order* of the dk/dv reductions and of the softmax
denominators changes, which moves the last bit or two of some float32 results.

``softmax`` is still the module-level, gate-checked function -- the blocked code
calls it once per query block on a 2-D (B*H*bs, i1) view. No fast path, no
inlined exp.

ATT_BLOCK = 64 was chosen by measurement, not by taste. Blocking is a genuine
trade: the forward loses exp work proportional to 1/nb, but the backward gains
per-block allocation, temporary and batched-gemm-dispatch overhead proportional
to nb -- attention-backward alone measured 0.84 ms/layer unblocked, 0.89 at
nb = 2 and 1.05 at nb = 4. Interleaved A/B over 60 loss_and_grads calls at the
real config: 36.91 ms r1, 35.98 at block 64 (1.026x), 36.21 at 32 (1.019x),
39.18 at 16 (0.942x -- a regression). nb = 2 wins, and the headline number is
honest and small: ~2.6%, roughly +42 steps in the 60 s budget, not the ~7% that
counting exp calls alone would predict. T = 128 is a multiple of 64, but
``_blocks`` handles a ragged tail anyway.
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
    "n_head": 4,
    "n_embd": 128,
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

#: Query-block size for blocked causal attention. See the module docstring for
#: why 64 (it was measured against 16 and 32). T need not be a multiple of it.
ATT_BLOCK = 64


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

    p = {"wte": n(V, C), "wpe": n(T, C)}
    for i in range(L):
        p[f"g1_{i}"] = np.ones(C, dtype=DTYPE)
        p[f"b1_{i}"] = np.zeros(C, dtype=DTYPE)
        # draw order preserved: wq, then wk, then wv, from the same generator.
        wq = n(C, C)
        wk = n(C, C)
        wv = n(C, C)
        p[f"wqkv_{i}"] = np.ascontiguousarray(np.concatenate((wq, wk, wv), axis=1))
        p[f"wo_{i}"] = n(C, C)
        p[f"g2_{i}"] = np.ones(C, dtype=DTYPE)
        p[f"b2_{i}"] = np.zeros(C, dtype=DTYPE)
        p[f"w1_{i}"] = n(C, 4 * C)
        p[f"w2_{i}"] = n(4 * C, C)
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


_BLOCK_CACHE: dict = {}


def _blocks(T: int) -> tuple:
    """``((i0, i1), ...)`` query blocks of at most ATT_BLOCK rows, cached."""
    bl = _BLOCK_CACHE.get(T)
    if bl is None:
        bl = tuple(
            (i0, min(i0 + ATT_BLOCK, T)) for i0 in range(0, T, ATT_BLOCK)
        )
        _BLOCK_CACHE[T] = bl
    return bl


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

        # ---- blocked causal attention: keys for query block [i0,i1) stop at i1
        khT = kh.transpose(0, 1, 3, 2)  # (B,H,hs,T) view
        y = np.empty((N, C), dtype=DTYPE)
        yh = y.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        prs = []
        for i0, i1 in _blocks(T):
            att = qh[:, :, i0:i1] @ khT[:, :, :, :i1]  # (B,H,bs,i1)
            # only the bs x bs diagonal block needs masking; everything to its
            # left is fully visible and everything to its right is not built.
            att[:, :, :, i0:i1] += mask[i0:i1, i0:i1]
            pr = softmax(att.reshape(-1, i1)).reshape(B, H, i1 - i0, i1)
            np.matmul(pr, vh[:, :, :i1], out=yh[:, :, i0:i1])
            prs.append(pr)
        o = y @ p[f"wo_{i}"]
        if want_cache:
            cache[f"qkv_{i}"] = qkv
            cache[f"pr_{i}"], cache[f"y_{i}"] = prs, y
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
        pr_blocks = cache[f"pr_{i}"]
        qkv = cache[f"qkv_{i}"]
        v5 = qkv.reshape(B, T, 3, H, hs)
        qh = v5[:, :, 0].transpose(0, 2, 1, 3)  # already scale-folded
        kh = v5[:, :, 1].transpose(0, 2, 1, 3)
        vh = v5[:, :, 2].transpose(0, 2, 1, 3)

        dqkv = np.empty((N, C3), dtype=DTYPE)
        d5 = dqkv.reshape(B, T, 3, H, hs)
        dqh = d5[:, :, 0].transpose(0, 2, 1, 3)
        dkh = d5[:, :, 1].transpose(0, 2, 1, 3)
        dvh = d5[:, :, 2].transpose(0, 2, 1, 3)
        blocks = _blocks(T)
        for bi in range(len(blocks) - 1, -1, -1):
            i0, i1 = blocks[bi]
            pr = pr_blocks[bi]
            dyb = dyh[:, :, i0:i1]
            dpr = dyb @ vh[:, :, :i1].transpose(0, 1, 3, 2)
            prT = pr.transpose(0, 1, 3, 2)
            if bi == len(blocks) - 1:
                np.matmul(prT, dyb, out=dvh)
            else:
                dvh[:, :, :i1] += prT @ dyb
            tmp = dpr * pr
            sred = tmp.sum(axis=-1, keepdims=True)
            del tmp
            dpr -= sred
            dpr *= pr
            datt = dpr
            np.matmul(datt, kh[:, :, :i1], out=dqh[:, :, i0:i1])
            dattT = datt.transpose(0, 1, 3, 2)
            if bi == len(blocks) - 1:
                np.matmul(dattT, qh[:, :, i0:i1], out=dkh)
            else:
                dkh[:, :, :i1] += dattT @ qh[:, :, i0:i1]
        d5[:, :, 0] *= scale

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
def adamw_step(p, gr, state, cfg, t):
    lr, b1, b2 = cfg["lr"], cfg["beta1"], cfg["beta2"]
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
