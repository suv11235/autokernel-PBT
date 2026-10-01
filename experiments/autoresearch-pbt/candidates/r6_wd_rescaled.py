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

--------------------------------------------------------------------------------
ROUND 4 (r4_muon): **Muon + AdamW hybrid optimizer**, the one thing upstream
(karpathy/autoresearch train.py) treats as baseline rather than as a
hyperparameter and that this transplant was still missing. Everything above --
kernels, architecture, schedule shape, ALPHA, blocked attention -- is unchanged;
the only edit is the parameter update.

12. **Muon on the 2-D hidden matrices.** For each of ``wqkv_i (160,480)``,
    ``wo_i (160,160)``, ``w1_i (160,640)``, ``w2_i (640,160)`` -- 16 tensors at
    n_layer = 4 -- keep a single momentum buffer and *orthogonalize the momentum*
    before applying it:

        buf   <- mu * buf + (1 - mu) * g                 (mu = MUON_MOMENTUM)
        geff  <- (1 - mu) * g + mu * buf                 (Nesterov, lerp form)
        O     <- newton_schulz5(geff)                    (see below)
        p     -= lr * MUON_LR_MULT * sqrt(max(m, n)) * O

    ``newton_schulz5`` is the published quintic iteration: transpose so that
    rows <= cols (the smaller dimension drives the m^3 term), divide by the
    Frobenius norm, then ``NS_STEPS`` times

        A <- X @ X.T
        B <- b * A + c * (A @ A)
        X <- a * X + B @ X

    with ``a = 3.4445, b = -4.7750, c = 2.0315`` -- algebraically
    ``a*X + b*(X X^T X) + c*(X X^T)^2 X``, i.e. the odd quintic p(X) = aX + bX^3
    + cX^5 in the singular values -- and transpose back. The coefficients are the
    ones tuned so that p maps (0, 1] into a neighbourhood of 1 as fast as
    possible; they do NOT converge to exactly 1 (p(1) = 0.701), which is why the
    resulting singular values sit in a band around ~0.7-1.4 rather than at
    exactly 1. That band, not exact orthogonality, is what Muon uses.

    AdamW keeps: ``wte``, ``wpe`` (vocabulary/position lookups, not
    feature-mixing matrices), ``wlm`` (the (160, 256) head: Newton-Schulz there
    is both expensive and conceptually wrong), and every 1-D tensor (all
    layernorm gains and biases). This is upstream's split.

13. **Scaling and the Muon learning rate.** The scale factor is
    ``sqrt(max(m, n))``, an RMS-matching equivalent of the more familiar
    ``sqrt(max(1, rows/cols))``. Because ``O`` has ~unit singular values, its
    Frobenius norm is ~sqrt(min(m, n)) and therefore
    ``RMS(O) ~ 1/sqrt(max(m, n))``; multiplying by ``sqrt(max(m, n))`` makes the
    Muon update's RMS exactly ``lr * MUON_LR_MULT`` on *every* matrix shape,
    independent of aspect ratio. AdamW's update RMS is ~``lr`` by construction
    (every coordinate is a bias-corrected m/sqrt(v) ~ +-1 times lr). So
    ``MUON_LR_MULT`` is, directly and shape-independently, **the ratio of the
    Muon update's RMS to the AdamW update's RMS at the same scheduled lr**.

    ``MUON_LR_MULT = 0.25``. Reasoned first, then confirmed on a 400-step run.
    The prior came from two independent places that agree: the Moonlight/Muon
    update-scaling convention targets an update RMS of 0.2 * lr when Muon shares
    AdamW's lr, and modded-nanogpt's own numbers imply the same ratio (Muon lr
    0.05 on 1024x1024 matrices under the sqrt(max(1, m/n)) scaling is an update
    RMS of 0.05/32 = 1.6e-3, against AdamW updates of ~8e-3 there). So the
    starting guess was 0.2-0.35, nudged up from the literature because this
    budget is only ~1700 steps and total displacement is scarce.

    Measured, at 400 steps with an identical seed, batch stream and lr schedule
    (train bpb averaged over steps 391-400): mult 0.15 -> 2.467, **0.25 ->
    2.421**, 0.35 -> 2.450, 0.5 -> 2.474, 0.7 -> 2.628, 1.0 -> 2.856, 1.5 ->
    2.898, against AdamW-only 2.777. The basin is flat between 0.25 and 0.35 and
    turns sharply bad above 0.5 -- full RMS parity with AdamW is much too large,
    which is the concrete refutation of "Muon just wants a bigger lr" in the naive
    sense. 0.25 is chosen over 0.35 because its advantage *grows* with horizon
    (they are tied at step 300 and 0.25 leads by 0.03 bpb at step 400), and the
    real run is 4x longer again. NOT validated at the full 1700 steps; 400 steps
    of a cosine schedule is not 1700, and this is the residual risk in the choice.

    The schedule is untouched: ``lr_at`` (3% budget-fraction warmup + cosine to
    5% of peak, peak 3e-3) drives both groups, and the Muon group simply
    multiplies it by MUON_LR_MULT.

    ``MUON_MOMENTUM = 0.95`` with Nesterov, upstream's default, rather than
    reusing beta1 = 0.9: there is no second moment to average against here, so
    the buffer is the only source of gradient smoothing. No bias correction is
    needed because Newton-Schulz normalizes by the Frobenius norm, making the
    update invariant to any scalar factor on ``geff`` -- the (1 - mu) warm-up
    transient cancels exactly.

    Gradient clipping is left byte-identical to r3_all: one global norm over ALL
    gradients. On the Muon group the clip is very nearly a no-op (a global
    rescale of g rescales buf and geff and is then divided out by the Frobenius
    normalization); it is kept so the AdamW group sees exactly the r3 dynamics.

    The honest risk is cost, not correctness: Newton-Schulz is 3 matmuls per
    iteration per tensor, 5 iterations, 16 tensors -- 4.6 GFLOP/step against a
    ~34 ms step. See the report for the measured ms/step; ``r4_muon_cheap.py`` is
    the same file with NS_STEPS = 3 in case that cost does not pay for itself.

--------------------------------------------------------------------------------
ROUND 5 (r5_muon_all): **the missing composition.** r4_muon (val_bpb 2.245087)
was branched from r3_all, not from r4_all, so it never carried r4_all's only
change: the per-parameter-group learning-rate multiplier. r4_all measured
2.348694 against r3_all's 2.385186 with that one line. This file is r4_muon plus
that one line. Nothing else moves: kernels, architecture, ALPHA, blocked
attention, the lr schedule, MUON_MOMENTUM, NS_STEPS and MUON_LR_MULT are all
byte-identical to r4_muon.

14. **``LR_MULT = {"wte": 0.25, "wlm": 0.25}``, AdamW branch only.**

    Mechanics. r4_muon's update path is ``hybrid_step``, which has two branches.
    ``is_muon(k, p)`` is ``p.ndim == 2 and k.startswith(("wqkv_","wo_","w1_",
    "w2_"))``, so ``wte``, ``wpe`` and ``wlm`` all fail it and all land on the
    AdamW branch, which is where the multiplier belongs and where it is applied.
    It is folded into the existing ``upd *= lr`` scalar multiply, so the step
    cost is unchanged (one extra Python dict lookup per tensor per step, ~40
    lookups). The Muon branch reaches ``continue`` before that line and scales
    only by ``lr_muon = lr * MUON_LR_MULT``, so LR_MULT is structurally unable to
    touch the 16 Muon tensors. The two multipliers are therefore independent and
    compose by construction rather than by luck.

    Placement relative to weight decay is deliberately r4_all's: the multiplier
    lands AFTER ``upd += wd * pk``, so it scales the Adam update and the
    decoupled decay together, which is standard AdamW per-group semantics (a
    group's lr multiplier multiplies that group's whole update). ``wd`` is 0.0 in
    CONFIG, so today this ordering is *unobservable* -- every arrangement gives
    bit-identical results. We match r4_all anyway so that if weight decay is ever
    turned on the composition does not silently change meaning. No disagreement
    with r4_all's author here; the only caveat worth recording is that "AdamW
    group semantics" is a convention, not a theorem -- an equally defensible
    reading is that a group lr multiplier should scale the gradient-driven update
    and leave decay on the global lr. That question is untested and untestable at
    wd = 0.0, so it is a note, not a change.

    Provenance of the VALUE, recorded so a later round does not re-litigate it.
    The direction is counter-intuitive: this is *damping* the vocabulary tensors,
    and boosting them is monotonically wrong. Measured by r4_all's author against
    a 2.462 baseline at their probe horizon:

        3x    -> 2.557      (worse)
        1.5x  -> 2.512      (worse)
        1.0x  -> 2.462      (baseline)
        0.5x  -> 2.409
        0.25x -> 2.405      <- chosen
        0.125x-> 2.422

    So 0.125-0.5 is a flat region and 0.25 is its MIDDLE, not a tuned point --
    do not re-tune inside that interval expecting a win, and do not read 0.25 as
    a fragile magic number. The obvious confound was controlled for: halving the
    GLOBAL peak lr to 1.5e-3 gives 2.518, i.e. much worse than damping wte/wlm
    alone, so this is not "the peak lr was simply too high". Most of the effect
    is ``wte`` alone; ``wlm`` is included because it was free and did not hurt.
    ``wpe`` is deliberately NOT in the dict.

    Residual risk. All of the above was measured under AdamW-on-everything. Under
    the hybrid the hidden matrices now move under Muon, so the *relative* speed
    of the vocabulary tensors versus the rest is already different from what
    r4_all measured, and the optimal damping under Muon need not be 0.25. The
    flatness of the 0.125-0.5 region is the reason to expect this to survive
    transplantation anyway. It is a composition of two independently measured
    wins, which the ledger shows is usually but not always additive -- r4_all_tied
    (2.397929) is the standing counterexample that compositions can fail.

--------------------------------------------------------------------------------
ROUND 5b (r5_muon_lr): **MUON_LR_MULT retuned at the real horizon, 0.25 -> 0.10.**
This file is r5_muon_all with that one constant changed. Nothing else differs.

15. r4_muon's author picked 0.25 and pre-registered exactly this doubt: they
    validated it at 400 steps, sanity-checked at ~900, never at the real ~1550-
    1760 step horizon with the cosine fully decayed, and observed that the
    optimum drifted DOWNWARD with every horizon increase they tried. They
    guessed the true optimum might sit near 0.15-0.20. **The direction of their
    suspicion is confirmed and its magnitude was understated: the optimum at the
    real horizon is ~0.10, below the range they guessed.**

    Method. The step cost of the Muon branch is independent of MUON_LR_MULT (it
    is one scalar on an already-computed Newton-Schulz output), which was
    verified rather than assumed: calibration measured 37.96-38.20 ms/step across
    0.15/0.20/0.25. So equal-wall-clock reduces to equal step count, and the
    count is PINNED at 1571 = 60 s / 38.2 ms for every setting rather than
    re-calibrated per batch -- a calibration taken under transient machine load
    silently hands one setting fewer steps and confounds everything (this
    actually happened on a discarded batch that gave 0.10 only 1251 steps).
    ``lr_at`` is driven with ``frac = i / 1571`` so the cosine decays to
    ``lr_min_frac`` exactly at the last step, and val_bpb is ``prepare.evaluate``
    on the fixed windows. Two seeds per point (seed drives init AND the batch
    stream), four seeds at the two decision points.

    val_bpb at 1571 steps, r5_muon_all's config, one column per seed:

        mult      seed0     seed1     seed2     seed3      mean
        0.05     2.264104  2.257995      -         -       2.2611
        0.075    2.229426  2.226524      -         -       2.2280
        0.10     2.218432  2.218368  2.223238  2.206823    2.2167   <- chosen
        0.125    2.246620  2.229341      -         -       2.2380
        0.15     2.229636  2.237463      -         -       2.2336
        0.20     2.254152  2.249210      -         -       2.2517
        0.25     2.241367  2.268385  2.246216  2.253002    2.2522   (r4_muon)

    Reading. The minimum is now INTERIOR and bracketed on both sides: 0.05 is
    clearly too small (the hidden matrices barely move) and everything from 0.125
    up is monotonically worse in the mean. 0.10 beats 0.25 on all four seeds
    (margins 0.023, 0.050, 0.023, 0.046; mean -0.0355), and it beats every other
    tested setting on both of the seeds where all seven were run. Seed-to-seed
    spread at a fixed mult reaches 0.027 (at 0.25), so the 0.10-vs-0.25 gap is
    comfortably outside noise while the 0.10-vs-0.125/0.15 gap (0.015-0.020) is
    NOT -- and indeed 0.125 and 0.15 come out inverted relative to the trend,
    which is exactly what noise at that scale looks like. So the honest claim is
    **0.075-0.15 is a shallow basin, 0.10 is its best measured point, and the
    whole basin beats 0.25**; 0.10 is not resolved as superior to 0.075 or 0.15
    individually. Do not re-tune inside 0.075-0.15 expecting a further win.

    Sanity check against the ledger: reproducing 0.25 here gives a 4-seed mean of
    2.2522 against the runner's measured r4_muon 2.245087 at 1556 steps, and the
    runner's single number sits inside this file's per-seed spread. The harness
    and this in-process driver agree, so the comparison above is measuring what
    it claims to.

    Why lower is right, mechanistically. MUON_LR_MULT is the ratio of the Muon
    update RMS to the AdamW update RMS at the same scheduled lr, so it is a
    *total displacement* knob for the 16 hidden matrices. At 400 steps under a
    cosine that never finished decaying, total displacement was scarce and a
    larger ratio bought needed movement; at 1571 steps with the cosine fully
    decayed it is not scarce, and the same ratio overshoots. That is precisely
    why the optimum drifted down with every horizon increase, and it predicts it
    would drift down again if the budget grew -- so this value is tied to the
    60-second budget and must be re-measured if the budget or the step cost
    changes materially.

    Residual risk, stated plainly. (a) 1571 steps is this machine's rate for
    r5_muon_all today; the runner measured r4_muon at 1556 and r4_all at 1756, so
    the real horizon has moved before and 0.10 was tuned at one point on that
    axis. (b) The sweep was run with LR_MULT already in place, i.e. at the
    composed configuration, which is correct -- but it means 0.10 is the optimum
    *given* LR_MULT and is not independently valid for r4_muon alone.
    (c) 0.075-0.15 is a basin, so if a future change shifts things slightly, the
    right response is to re-measure the basin, not to trust 0.10 to three digits.

ROUND 6b (r6_wd_rescaled): **Muon-branch decoupled weight decay, rescaled to
the new MUON_LR_MULT.**

    Single-variable change against r5_muon_lr.py: four executable lines, all in
    the Muon branch or its constants.

        MUON_WEIGHT_DECAY = 5.0                                   (new constant)
        wd_muon = cfg.get("muon_weight_decay", MUON_WEIGHT_DECAY) (new local)
        if wd_muon:                                               (new)
            pk *= DTYPE(1.0 - lr_muon * wd_muon)                  (new)

    LR_MULT, MUON_LR_MULT (0.10), the AdamW branch, the schedule, the
    architecture and both gated kernels are untouched. The decay expression is
    character-for-character r5_wd_muon's; only the constant differs.

16. **Why 5.0 and not 2.0: the invariant I chose, stated explicitly.**

    r5_wd_muon tuned ``pk *= (1 - lr_muon * wd)`` with wd = 2.0 at
    MUON_LR_MULT = 0.25, i.e. a per-step shrink of ``1 - 0.50 * lr``. At
    MUON_LR_MULT = 0.10 the same wd = 2.0 gives ``1 - 0.20 * lr`` -- 40% of the
    shrinkage per step. Carrying 2.0 forward is therefore not "the tuned decay,
    retested at a new operating point"; it is a 2.5x weaker decay that has
    never been measured anywhere.

    There are two defensible invariants and they disagree, so this is a choice,
    not an arithmetic fact:

      (A) **Match total shrinkage.** Hold ``lr_muon * wd`` fixed, i.e. hold the
          accumulated multiplicative decay ``prod_t (1 - lr_muon(t) * wd)`` ~
          ``exp(-wd * sum_t lr_muon(t))`` fixed. This is what 5.0 does.

      (B) **Match shrinkage relative to the update.** The Muon update has RMS
          exactly ``lr_muon`` by construction (that is what the
          ``sqrt(max(m, n))`` factor buys), and the decay step is
          ``lr_muon * wd * |p|``. Their ratio is ``wd * RMS(p)`` -- which does
          not contain lr_muon at all. So invariant (B) prescribes *no change*:
          wd stays 2.0. Equivalently, the fixed point of the weight norm, where
          decay pull balances gradient-driven growth, is set by wd alone and is
          untouched by MUON_LR_MULT.

    **I chose (A), wd = 5.0.** Two reasons.

      * (B) is degenerate here as a proposal: it says "change nothing", which
        would spend a runner row on wd = 2.0 at a new multiplier without any
        model of why that point should be good. (A) at least reproduces a
        quantity that was measured to help.
      * More substantively, r5_wd_muon's own evidence says the run is on the
        *approach* to the decay's equilibrium, not sitting at it: the effect
        existed only at the full horizon and decay hurt monotonically at 400
        steps. If the run reached the fixed point, the 400-step and full-length
        results would agree in sign. On the approach, what has been consumed is
        the accumulated shrinkage budget ``sum_t lr_muon(t) * wd``, and that is
        exactly what (A) holds fixed. (B)'s fixed-point argument is the right
        one only in the regime this run demonstrably is not in.

    Where 5.0 sits in the tuned range. r5_wd_muon found wd 2.0 and 4.0
    statistically tied at MUON_LR_MULT = 0.25, i.e. a flat plateau over
    ``lr_muon * wd`` in [0.50, 1.00] (in units of the scheduled lr). At
    MUON_LR_MULT = 0.10 that plateau is wd in [5.0, 10.0]. 5.0 is its low edge:
    the conservative end of a region that was flat in the old data, chosen so
    that if the rescaling logic is wrong the error is toward too little decay
    rather than toward over-regularizing a model that is still underfitting.

    Doubt, pre-registered. r5_wd_muon measured a 3/3-seed in-process win of
    -0.0245 and then a +0.0135 LOSS on the runner, on a machine running ~20%
    slow (1249 steps instead of ~1550). Two readings survive that: the decay
    genuinely does not help at the real horizon, or it does but was 2.5x
    mis-scaled and/or lost inside wall-clock schedule jitter (worth ~+/-0.01 on
    its own). This file tests the second reading at the correct scale and lets
    the runner, not an in-process pilot, decide. If it loses again, decoupled
    decay on the Muon branch should be closed for this program.
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

#: Momentum for the Muon group (Nesterov). Upstream default; deliberately not
#: beta1, because Muon has no second moment to smooth against.
MUON_MOMENTUM = 0.95
#: Newton-Schulz iterations. 5 is the published number.
NS_STEPS = 5
#: Ratio of the Muon update's RMS to the AdamW update's RMS at the same
#: scheduled lr (see the ROUND 4 note). Shape-independent by construction.
#: RETUNED IN THIS FILE from r4_muon's 0.25 to 0.10 at the real ~1571-step
#: horizon with the cosine fully decayed. See item 15 in the docstring for the
#: seven-point, multi-seed sweep this number comes from.
MUON_LR_MULT = 0.10
#: Quintic Newton-Schulz coefficients (Jordan et al.). Do not "clean these up":
#: they are a tuned polynomial, not an expansion of anything.
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315
#: Decoupled weight decay for the MUON group -- the 16 hidden matrices, 92.1%
#: of all parameters, which receive NO decay in r4_muon/r5_muon_lr (the Muon
#: branch ``continue``s before the decay line). Multiplies ``lr_muon``, so it
#: follows the cosine schedule. Deliberately NOT ``cfg["weight_decay"]``: that
#: one multiplies the raw lr, reaches only wte/wpe/wlm, and stays 0.0 so the
#: ledger's earlier rows remain comparable.
#: RESCALED from r5_wd_muon's 2.0 to 5.0 because MUON_LR_MULT moved 0.25 ->
#: 0.10 and the shrink is ``1 - lr * MUON_LR_MULT * wd``: 0.25 * 2.0 = 0.50 =
#: 0.10 * 5.0, so the per-step (and hence total) shrinkage is unchanged. See
#: item 16 in the module docstring for why total shrinkage is the invariant.
MUON_WEIGHT_DECAY = 5.0
#: Prefixes of the parameters that go to Muon: the 2-D hidden matrices only.
#: Embeddings (wte, wpe), the output head (wlm) and every 1-D tensor stay AdamW.
MUON_PREFIXES = ("wqkv_", "wo_", "w1_", "w2_")
#: Per-parameter-group learning-rate multipliers on the **AdamW branch only**
#: (ported unchanged from candidates/r4_all.py). Any key not listed steps at
#: 1.0x. Every key here is on the AdamW branch by construction: ``wte`` and
#: ``wlm`` do not match MUON_PREFIXES, so ``is_muon`` is False for both and the
#: Muon group can never see this dict.
LR_MULT = {"wte": 0.25, "wlm": 0.25}

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


def is_muon(key: str, param: np.ndarray) -> bool:
    """True for the 2-D hidden matrices Muon owns. 1-D tensors can never match."""
    return param.ndim == 2 and key.startswith(MUON_PREFIXES)


def newton_schulz5(g: np.ndarray, steps: int = NS_STEPS) -> np.ndarray:
    """Approximate orthogonalization of a 2-D matrix (the Muon inner routine).

    Returns ``U V^T``-ish: a matrix with the same singular vectors as ``g`` and
    singular values squashed into a band around 1 (roughly 0.7 to 1.4 after 5
    iterations from a Frobenius-normalized start -- the quintic's fixed point is
    not exactly 1 and is not meant to be).

    Transposes when ``rows > cols`` so the m x m intermediates are built on the
    *smaller* dimension, then transposes back; the result is the transpose of the
    result on the transpose, exactly, because the iteration is an odd polynomial
    in the singular values.
    """
    a, b, c = NS_A, NS_B, NS_C
    x = g if g.dtype == DTYPE else g.astype(DTYPE)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    # Frobenius-normalize so every singular value lands in (0, 1], which is the
    # basin the quintic coefficients were tuned for. 1e-7 guards an all-zero g.
    x = x / (DTYPE(np.linalg.norm(x)) + DTYPE(1e-7))
    for _ in range(steps):
        aa = x @ x.T
        bb = aa @ aa
        bb *= DTYPE(c)
        bb += DTYPE(b) * aa
        x2 = bb @ x
        x2 += DTYPE(a) * x
        x = x2
    return x.T if transposed else x


def hybrid_step(p, gr, state, cfg, t, lr=None):
    """Muon on the 2-D hidden matrices, AdamW on everything else.

    ``state[k]`` is a single momentum buffer for Muon params and the usual
    ``(m, v)`` pair for AdamW params -- see ``init_state``. The AdamW branch is
    byte-identical to ``adamw_step``'s body, and so is the global gradient clip.
    """
    lr = cfg["lr"] if lr is None else lr
    b1, b2 = cfg["beta1"], cfg["beta2"]
    eps, wd, clip = cfg["eps"], cfg["weight_decay"], cfg["grad_clip"]
    mu = MUON_MOMENTUM
    lr_muon = lr * MUON_LR_MULT
    #: overridable per-cfg for a sweep; defaults to the module constant.
    wd_muon = cfg.get("muon_weight_decay", MUON_WEIGHT_DECAY)
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
        pk = p[k]
        if is_muon(k, pk):
            buf = state[k]
            # buf <- mu*buf + (1-mu)*g ; geff <- (1-mu)*g + mu*buf  (Nesterov)
            buf *= DTYPE(mu)
            buf += DTYPE(1.0 - mu) * g
            geff = g * DTYPE(1.0 - mu)
            geff += DTYPE(mu) * buf
            o = newton_schulz5(geff)
            # RMS(o) ~ 1/sqrt(max(m,n)), so this makes RMS(update) == lr_muon
            o *= DTYPE(lr_muon * math.sqrt(max(pk.shape)))
            pk -= o
            # Decoupled weight decay, applied to the parameter and NOT folded
            # into geff (Newton-Schulz normalizes its input, so a gradient-side
            # L2 term would have its scale divided straight back out). Scaled by
            # the group's own scheduled lr, so it decays with the cosine.
            # Identical arithmetic to r5_wd_muon; only the constant is rescaled.
            if wd_muon:
                pk *= DTYPE(1.0 - lr_muon * wd_muon)
            continue
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
        if wd and pk.ndim > 1:
            upd += wd * pk
        # LR_MULT is applied HERE, inside the AdamW branch, folded into the
        # scalar multiply that was already present: no extra temporary and no
        # extra pass. Placement matches candidates/r4_all.py exactly -- AFTER
        # the decay term, so the multiplier scales the Adam update and the
        # decoupled decay together (standard AdamW per-group semantics). wd is
        # 0.0 today, so the ordering is currently unobservable; it is kept
        # identical so that turning wd on later reproduces r4_all's semantics
        # rather than silently diverging. The Muon branch above `continue`s
        # before reaching this line and uses lr_muon only, so LR_MULT cannot
        # leak into it.
        upd *= lr * LR_MULT.get(k, 1.0)
        pk -= upd


def init_state(p: dict) -> dict:
    """One momentum buffer per Muon param, an (m, v) pair per AdamW param."""
    return {
        k: (np.zeros_like(v) if is_muon(k, v) else (np.zeros_like(v), np.zeros_like(v)))
        for k, v in p.items()
    }


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
        # Same placement as in hybrid_step's AdamW branch. This function is not
        # on the live path (main() calls hybrid_step); it is kept in sync so it
        # stays a usable AdamW-only control.
        upd *= lr * LR_MULT.get(k, 1.0)
        p[k] -= upd


# ------------------------------------------------------------------------------- main
def main() -> None:
    cfg = dict(CONFIG)
    # Seed override for the multi-seed rigor check. Not for agents to set.
    if (env := os.environ.get("AUTORESEARCH_SEED")) is not None:
        cfg["seed"] = int(env)
    train, val = prepare.load_data()
    p = init_params(cfg)
    state = init_state(p)
    stream = prepare.batches(train, cfg["batch_size"], prepare.BLOCK_SIZE, cfg["seed"] + 1)

    budget = prepare.Budget()
    steps, toks, last = 0, 0, float("nan")
    budget.start()
    while not budget.expired():
        x, y = next(stream)
        last, gr = loss_and_grads(p, cfg, x, y)
        steps += 1
        lr = lr_at(cfg, budget.elapsed / budget.seconds)
        hybrid_step(p, gr, state, cfg, steps, lr)
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
