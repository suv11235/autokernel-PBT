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

ROUND 2 COMPOSITION (r2_tsc): the above throughput layout PLUS the round-1
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

10. CONFIG shape from the round-1 capacity change: n_head 4 -> 2, n_embd
    128 -> 160 (n_layer 4 and batch_size 16 unchanged). n_head is a pure cost
    knob at zero parameter cost -- attention is B*H separate small gemms -- so
    the saving is spent on width. 160 % 2 == 0, so head size is 80.

--------------------------------------------------------------------------------
ROUND 3 (r3_softmax_fastexp): identical to r3_softmax_exact except that softmax's
exponential is ``fast_exp`` -- a bit-trick + degree-4-polynomial APPROXIMATION of
exp with ~2.6e-06 max relative error, against np.exp's ~1 ulp. This is a real
approximation of the kernel's semantics, declared as such. It is here so the
property gate can adjudicate it against the exact arm; it is deliberately NOT
tuned to pass. ``layernorm`` is byte-for-byte r2_tsc's.

MEASURED OUTCOME, stated here so nobody has to run it to find out: the
approximation is 0.80x the speed of ``np.exp``, i.e. SLOWER, because thirteen
separate numpy ufunc passes cost more memory traffic than one fused SIMD call --
see ``fast_exp``'s docstring for the per-pass breakdown. So this arm trades
accuracy for a 15% SLOWER softmax and a ~1% slower step. It is not a candidate
worth accepting on speed; it is a probe of what the gate does and does not see.
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


# ---------------------------------------------------------------------------- kernels
#: log2(e). ``exp(x) == 2**(x * LOG2E)``, and base 2 is the base a binary float's
#: exponent field is written in, so the integer part of the reduced argument needs no
#: arithmetic at all -- it is a field of the result.
_LOG2E = np.float32(1.4426950408889634)

#: Degree-4 minimax-in-relative-error polynomial for ``2**f`` on ``f in [0, 1)``,
#: fitted by iteratively reweighted least squares (equioscillation) on a 200k-point
#: grid and then rounded to float32. MEASURED maximum relative error of the
#: polynomial alone, with float32 coefficients: 2.62e-06.
_P0 = np.float32(1.0000026114589433)
_P1 = np.float32(0.6930036396822946)
_P2 = np.float32(0.24144318799044306)
_P3 = np.float32(0.05201132145650294)
_P4 = np.float32(0.013534016344632784)

#: Floor for the reduced argument. -126 keeps ``k + 127 >= 1``, i.e. keeps the
#: constructed exponent field inside the normal range, so the bit assembly below
#: never has to reason about subnormals or about a negative field wrapping into the
#: sign bit. The cost is that ``fast_exp`` returns ~1.2e-38 rather than exactly 0.0
#: for arguments below -87.3 (including the ``-inf`` of the causal mask). Relative
#: to the row max of 1.0 that is 34 orders of magnitude down -- it cannot move a row
#: sum, and it leaves attention causal to within 1e-38 rather than exactly.
_ARG_FLOOR = np.float32(-126.0)


def fast_exp(x: np.ndarray) -> np.ndarray:
    """An APPROXIMATION of ``np.exp`` for float32. Returns a new array.

    ``exp(x) = 2**(x*log2e) = 2**k * 2**f`` with ``k = floor(x*log2e)`` an integer
    and ``f = x*log2e - k`` in [0, 1). ``2**k`` is built by writing ``k + 127`` into
    the float32 exponent field directly (an int add, a shift and a reinterpret --
    no arithmetic on the value), and ``2**f`` comes from the degree-4 polynomial
    above evaluated by Horner. That trades libm's correctly-rounded ``exp`` for
    thirteen cheap elementwise passes.

    MEASURED, AND IT LOSES: 1.59 ns/element against ``np.exp``'s 1.27 on this
    machine -- 0.80x, i.e. 25% SLOWER, at both (2048,128) and (512,4096). The
    per-element arithmetic really is much cheaper; the passes are not. Each numpy
    ufunc is its own full round trip through memory, and a plain float32 multiply
    pass costs 0.050 ns/element here, so thirteen passes is already 0.65 ns of pure
    traffic before the two expensive ones (``np.maximum`` against a scalar at 0.26
    and the float->int cast at 0.28, each ~5x a multiply). ``np.exp`` is itself a
    SIMD polynomial with the identical exponent trick, but it does all of it in
    registers in ONE pass. A degree-3 polynomial (7.5e-05 relative error) was also
    measured: 1.45 ns/element, 0.88x -- still slower. ``np.ldexp`` instead of the
    bit assembly is 2.68 ns/element, 0.47x. There is no NumPy-level arrangement of
    this algorithm that beats the fused call, so the accuracy is spent for nothing.
    Kept exactly as written, unturned, because the point of this file is to be
    adjudicated, not to win.

    THIS IS NOT ``np.exp``. Its relative error is ~2.6e-06 where ``np.exp``'s is
    ~6e-08 (one float32 ulp), roughly a factor of 40 worse, and its error is a
    non-smooth (piecewise polynomial, discontinuous in the derivative at each
    integer of ``x*log2e``) function of the argument. Everything the caller does
    with the result inherits that.

    ``x`` is not modified. NaN input is not handled: it reaches an int cast that is
    undefined and warns. The gate's domain and the model's attention scores never
    produce one.
    """
    t = np.multiply(x, _LOG2E, dtype=np.float32)
    # Clamps -inf (the causal mask) to the floor instead of letting it reach floor()
    # and then an int cast, which would be undefined.
    np.maximum(t, _ARG_FLOOR, out=t)
    k = np.floor(t)
    t -= k  # t is now the fractional part f in [0, 1)

    # Horner on f. One accumulator buffer, everything else in place.
    y = np.multiply(t, _P4)
    y += _P3
    y *= t
    y += _P2
    y *= t
    y += _P1
    y *= t
    y += _P0

    # 2**k by construction: exponent field = k + 127, mantissa zero.
    ki = k.astype(np.int32)
    ki += 127
    ki <<= 23
    y *= ki.view(np.float32)
    return y


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis, numerically stable -- with ``fast_exp``.

    Structurally identical to r2_tsc's softmax: same max-subtraction, same row-sum
    normalization, same shape and dtype handling. The ONLY difference is that the
    exponential is ``fast_exp`` (max relative error ~2.6e-06) instead of ``np.exp``
    (~1 ulp). Nothing here is dropped, reordered or specialized by shape.

    What that does and does not disturb, stated up front:

    * rows still sum to 1, because the normalizer is the sum of the very same
      approximate exponentials -- the approximation cancels in ``e_i / sum(e_j)``
      and cannot show up in the row sum at all;
    * values are still in [0, 1] and still finite, for the same reason;
    * the OUTPUT VALUES are wrong by ~5e-06 relative against a true softmax: what
      survives normalization is the *spread* of the per-element relative errors
      across a row, not their common part;
    * shift-invariance is only approximate. ``x + c`` rounds to a different float32,
      so the reduced arguments differ by ~1e-05 between base and shifted rows, and
      ``fast_exp``'s error is a non-smooth function of the argument, so the two
      outputs differ by more than they would with a correctly rounded exp.

    This is a genuine approximation and it is labelled as one. It exists to be
    adjudicated by the property gate, not to sneak past it.
    """
    e = np.subtract(x, np.max(x, axis=-1, keepdims=True))
    e = fast_exp(e)
    s = np.sum(e, axis=-1, keepdims=True)
    np.reciprocal(s, out=s)
    e *= s
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
