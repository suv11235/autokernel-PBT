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
ROUND 3 (r3_all): the composition of every independently-measured round-2 win.
Nothing new is invented here; three changes that were measured separately and are
mutually orthogonal are put in one file.

Carried over unchanged from candidates/r2_tsc.py (val_bpb 2.456641, the incumbent
best):

1. Fused QKV: one ``(N, C) @ (C, 3C)`` gemm instead of three ``(C, C)`` gemms.
2. Every activation is a flat ``(B*T, C)`` array, so each projection is ONE gemm
   with M = 2048 rather than 16 stacked gemms with M = 128 -- this is what lets
   Accelerate thread at all.
3. The causal mask is built once and cached per ``(T, dtype)``.
4. ``1/sqrt(hs)`` folded into q before the QK matmul (and into dq after).
5. Attention writes straight into the merged output buffer via ``out=`` on
   strided views; dq/dk/dv are written straight into the fused ``(N, 3C)``
   buffer.
6. In-place / buffer-reusing softmax, layernorm, layernorm_backward, softmax
   backward and AdamW update.
7. Budget-fraction learning-rate schedule (``lr_at``): linear warmup over the
   first 3% of the *wall-clock budget* from 5% of peak, then cosine decay to 5%
   of peak. Peak lr 3e-3, beta2 0.99.
8. Residual-branch init scaling: ``wo`` and ``w2`` -- and only those two -- are
   drawn at 0.02 / sqrt(2 * n_layer).
9. CONFIG shape n_head = 2, n_embd = 160 (n_layer 4, batch_size 16). n_head is a
   pure cost knob at zero parameter cost, so the saving buys width. 160 % 2 == 0
   so head size is 80.

NEW IN THIS FILE -- ported from the two other round-2 candidates:

10. **Squared-ReLU MLP** (from candidates/r2_relu2.py, which measured 1.208x
    faster per step *and* slightly better per-step loss):

        act(x) = ALPHA * max(x, 0)**2,    act'(x) = 2 * ALPHA * max(x, 0)

    ``np.tanh`` disappears from both directions of the MLP. numpy 2.5.2 has no
    SIMD transcendental kernel on this machine, so tanh over the (2048, 640)
    hidden state was the single most expensive elementwise pass in the step.
    The forward caches ``r = max(x, 0)``, not ``x``, so the backward is two
    in-place multiplies and never re-clamps.

    ALPHA IS NOT A FREE KNOB AND IS NOT PORTABLE. ``max(x,0)**2`` is degree-2
    homogeneous, so its output scale goes as the *square* of its input scale.
    r2_relu2.py hard-coded ALPHA = 1.883410, a value derived for n_embd = 128;
    this file is n_embd = 160, where it would be ~11% too large. So ALPHA is
    computed from ``cfg["n_embd"]`` at import time by ``_alpha_for_embd`` and can
    never go stale again:

      * at init the MLP input is layernormed (unit variance) with gain 1 and
        bias 0 and ``w1 ~ N(0, 0.02**2)``, so ``z1 ~ N(0, sigma**2)`` with
        ``sigma = 0.02 * sqrt(n_embd)``;
      * ``RMS[max(z,0)**2] = sqrt(E[max(z,0)**4]) = sqrt(0.5 * 3 * sigma**4)
        = sqrt(1.5) * sigma**2`` exactly;
      * ``RMS[gelu_tanh_approx(z)]`` has no closed form, so it is integrated
        against the Gaussian density by a deterministic midpoint rule over
        +-10 sigma (converged to every printed digit by 2001 points, and equal
        to 200-point Gauss-Hermite to 1e-15);
      * ALPHA is the ratio, so the MLP branch leaves initialization at exactly
        the magnitude the GELU it replaces had.

    Values: n_embd = 128 -> sigma 0.22627417, RMS[gelu] 0.11812148,
    RMS[relu2] 0.06270694, ALPHA 1.883707 -- which reproduces r2_relu2.py's
    1.883410 to 1.6e-4 relative (that file quotes RMS[gelu] = 0.11810108, a
    Monte-Carlo estimate; the quadrature value 0.11812148 is the accurate one, so
    the small disagreement is sampling noise in the original, not a change of
    definition). n_embd = 160 -> sigma 0.25298221, RMS[gelu] 0.13328955,
    RMS[relu2] 0.07838367, **ALPHA = 1.700476**.

11. **Blocked causal attention** (from candidates/r2_blockattn.py, ATT_BLOCK =
    64). A pure refactor: for query block [i0, i1) only keys [0, i1) are
    reachable, so the masked half of the score matrix is never materialised and
    never exponentiated. Exp count falls from T**2 to 0.75 T**2 at nb = 2. The
    masked entries of ``pr`` are exactly 0.0 in the unblocked form, so their
    contribution to dk, dv and to the softmax row sums is exactly 0.0; only the
    summation *order* of the dk/dv reductions changes. ``softmax`` is still the
    module-level gate-checked function, called once per query block on a 2-D
    ``(B*H*bs, i1)`` view -- no inlined exp, no fast path. Measured at 1.026x on
    its own at n_embd = 128 / n_head = 4, and re-measured at only 1.005x at THIS
    config (interleaved A/B, 60 calls each: 31.34 ms blocked vs 31.51 ms
    unblocked) -- widening to n_embd = 160 and halving the head count shrank
    attention's share of the step, so blocking's already-small win shrank with
    it. It is kept because it is free and verified, not because it earns much.
    ``ATT_BLOCKED`` below keeps the unblocked path alive so "pure refactor"
    stays a testable claim: at the real config the forward logits and the loss
    are BIT-IDENTICAL between the two paths, and over all 1,334,080 gradient
    coordinates the worst absolute deviation is 1.49e-8, i.e. 7.5e-8 of the
    largest gradient magnitude -- one float32 ulp, from the reordered dk/dv
    reductions. (Elementwise relative error reaches 1.1e-1, but only on
    coordinates of magnitude ~7e-11, which is cancellation noise, not error:
    restricted to coordinates above 1e-3 of the gradient max the worst relative
    deviation is 1.6e-5, and above 1e-2 it is 1.9e-6.)
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
    "n_head": 2,
    "n_embd": 160,
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

#: Query-block height for the blocked causal attention. T = 128 -> nb = 2.
#: ``_blocks`` handles a ragged tail, so this need not divide T.
ATT_BLOCK = 64
#: Set False to run the mathematically-identical unblocked attention. Exists so
#: that "blocked attention is a pure refactor" stays a testable claim rather than
#: a comment. The branch is taken once per layer; it costs nothing measurable.
ATT_BLOCKED = True


# ------------------------------------------------------- squared-ReLU normalisation
def _gelu_tanh_ref(x: np.ndarray) -> np.ndarray:
    """float64 reference tanh-GELU, used only to calibrate ALPHA at import."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _alpha_for_embd(n_embd: int, npts: int = 2001, lim: float = 10.0) -> float:
    """ALPHA such that RMS[ALPHA * max(z,0)**2] == RMS[gelu(z)] at init scale.

    ``z ~ N(0, sigma**2)`` with ``sigma = 0.02 * sqrt(n_embd)``: the MLP input is
    layernormed (unit variance, gain 1, bias 0) and ``w1 ~ N(0, 0.02**2)``, so
    the pre-activation has that standard deviation at initialization.

    ``RMS[max(z,0)**2] = sqrt(1.5) * sigma**2`` in closed form (the fourth
    moment of a half-normal). ``RMS[gelu]`` has none, so it is integrated by a
    deterministic midpoint rule in units of sigma over ``+-lim``; the result is
    stable to 1e-15 from npts = 2001 upward and agrees with 200-point
    Gauss-Hermite. No RNG, so this is reproducible run to run.
    """
    sigma = 0.02 * math.sqrt(float(n_embd))
    u = (np.arange(npts, dtype=np.float64) + 0.5) / npts * (2.0 * lim) - lim
    w = np.exp(-0.5 * u * u)  # unnormalised Gaussian weight; normalised by w.sum()
    g = _gelu_tanh_ref(u * sigma)
    rms_gelu = math.sqrt(float((w * g * g).sum() / w.sum()))
    rms_relu2 = math.sqrt(1.5) * sigma * sigma
    return rms_gelu / rms_relu2


#: Scale on the squared-ReLU activation, derived from the configured width.
#: n_embd = 160 -> 1.700476 (n_embd = 128 would give 1.883707, matching
#: candidates/r2_relu2.py's hard-coded 1.883410 to 1.6e-4).
ALPHA = _alpha_for_embd(CONFIG["n_embd"])


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


def relu2_fwd(x: np.ndarray):
    """Forward of ``ALPHA * max(x, 0)**2`` plus the cache the backward needs.

    Returns ``(r, g)`` with ``r = max(x, 0)`` and ``g = ALPHA * r * r``. Caching
    ``r`` rather than ``x`` means the backward pass is two in-place multiplies:
    the clamp is never recomputed.
    """
    r = np.maximum(x, DTYPE(0.0))
    g = r * r
    g *= DTYPE(ALPHA)
    return r, g


def relu2_backward(dy: np.ndarray, r: np.ndarray) -> np.ndarray:
    """NOTE: ``dy`` is consumed in place and returned. ``r`` is ``max(x, 0)``."""
    dy *= r
    dy *= DTYPE(2.0 * ALPHA)
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


_BLOCK_CACHE: dict = {}


def _blocks(T: int) -> tuple:
    """``((i0, i1), ...)`` query blocks of at most ATT_BLOCK rows, cached."""
    bl = _BLOCK_CACHE.get(T)
    if bl is None:
        bl = tuple((i0, min(i0 + ATT_BLOCK, T)) for i0 in range(0, T, ATT_BLOCK))
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

        y = np.empty((N, C), dtype=DTYPE)
        yh = y.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        khT = kh.transpose(0, 1, 3, 2)  # (B,H,hs,T) view
        if ATT_BLOCKED:
            # ---- blocked causal attention: keys for query block [i0,i1) stop
            # at i1, so the masked half is never built and never exponentiated.
            prs = []
            for i0, i1 in _blocks(T):
                att = qh[:, :, i0:i1] @ khT[:, :, :, :i1]  # (B,H,bs,i1)
                # only the bs x bs diagonal block needs masking; everything to
                # its left is fully visible and everything right is not built.
                att[:, :, :, i0:i1] += mask[i0:i1, i0:i1]
                pr = softmax(att.reshape(-1, i1)).reshape(B, H, i1 - i0, i1)
                np.matmul(pr, vh[:, :, :i1], out=yh[:, :, i0:i1])
                if want_cache:
                    prs.append(pr)
        else:
            att = qh @ khT
            att += mask
            prs = [softmax(att.reshape(-1, T)).reshape(B, H, T, T)]
            del att
            np.matmul(prs[0], vh, out=yh)
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
            r, g = relu2_fwd(z1)
        else:
            # in place: z1 is a fresh gemm output, nothing else aliases it
            np.maximum(z1, DTYPE(0.0), out=z1)
            z1 *= z1
            z1 *= DTYPE(ALPHA)
            g = z1
        m = g @ p[f"w2_{i}"]
        if want_cache:
            cache[f"hn2_{i}"], cache[f"a2_{i}"] = hn2, a2
            cache[f"r_{i}"], cache[f"g_{i}"] = r, g
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
        dz1 = relu2_backward(dg, cache[f"r_{i}"])
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
        # dq is a straight per-block write (each query row is in one block);
        # dk and dv are reductions OVER query blocks, so the loop runs in
        # reverse: the widest block (last, keys span [0,T)) initialises them
        # with out= and earlier blocks accumulate into a prefix. Exact, because
        # the entries this skips are exactly 0.0 in the unblocked form.
        blocks = _blocks(T) if ATT_BLOCKED else ((0, T),)
        nb = len(blocks)
        for bi in range(nb - 1, -1, -1):
            i0, i1 = blocks[bi]
            pr = pr_blocks[bi]
            dyb = dyh[:, :, i0:i1]
            dpr = dyb @ vh[:, :, :i1].transpose(0, 1, 3, 2)
            prT = pr.transpose(0, 1, 3, 2)
            if bi == nb - 1:
                np.matmul(prT, dyb, out=dvh)
            else:
                dvh[:, :, :i1] += prT @ dyb
            # softmax backward, in place on dpr (which is freshly allocated)
            tmp = dpr * pr
            sred = tmp.sum(axis=-1, keepdims=True)
            del tmp
            dpr -= sred
            dpr *= pr
            datt = dpr  # == original datt / scale
            np.matmul(datt, kh[:, :, :i1], out=dqh[:, :, i0:i1])
            dattT = datt.transpose(0, 1, 3, 2)
            if bi == nb - 1:
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
