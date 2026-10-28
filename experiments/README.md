# Experiments

Every experiment this project has run or pre-registered, grouped by the two task segments the paper is being planned around
plus the verification layer they share. Each self-contained experiment lives in its own directory
with a README that says how to run it. Experiments that run through the library itself (the
oracle comparison, the hardware runs) have their code under `src/`, `kernels/` and `scripts/`, and
are indexed here by their records.

```
experiments/
├── generation/          the agent writes or optimizes a kernel
│   └── autoresearch-pbt/      a property-gated optimization loop, and the gate's saboteur matrix
├── translation/         a known kernel mapped to another setup (built; nothing has run on a device)
└── verification/        the correctness layer itself: PBT oracles and formal proofs
    ├── formal-rigidity/       sampled symmetry audit of a verifier's axioms
    └── oracle-research/       does a prover in the toolkit change what an agent concludes?
```

Formal proofs live in [`proofs/`](../proofs/), and are compiled by `tests/proofs/` when a Lean
toolchain is present. Results are recorded, dated and append-only, in
[`docs/measurements/`](../docs/measurements/). The ranked summary is
[`docs/findings.md`](../docs/findings.md).

**Status:** **[M]** measured and recorded · **[B]** built · **[ran]** ran, not yet scored ·
**[planned]** pre-registered · **[proposed]** designed, not pre-registered.

## Generation

| Experiment | Question | Status | Headline | Code | Record |
|---|---|---|---|---|---|
| **Property-gated autoresearch loop** | Can a PBT gate make an optimization loop safe, and what does the oracle choice cost? | [M] | −21.2% val_bpb, +27.7% throughput over 26 candidates; 0 in-loop rejections; 6 saboteurs caught; an `allclose` gate cannot accept the baseline | [`generation/autoresearch-pbt/`](./generation/autoresearch-pbt/) | [`2026-08-18-autoresearch-loop-pbt-gate`](../docs/measurements/2026-08-18-autoresearch-loop-pbt-gate.md) |
| **Temperature blind spot** | What wrong kernels does the gate's decisive arm admit? | [M] | The declarative arm admits softmax(βx) at every β, the 1/n fill (52–54× faster) and softmax(−x); the other arms reject them | [`generation/autoresearch-pbt/saboteurs.py`](./generation/autoresearch-pbt/saboteurs.py) | [`2026-10-01-softmax-temperature-blind-spot`](../docs/measurements/2026-10-01-softmax-temperature-blind-spot.md) |
| **Log-ratio headroom** (E2 design input) | Can one float threshold on the log-ratio law accept every correct softmax, device runs included, and reject every wrong family? | [M] | Yes at 30 eps on everything recorded: correct ≤ 12.3 (A10), wrong ≥ 2.67e7; but device spread grows with row range (r = 0.92), so E2 should test a range-aware bound | [`generation/log-ratio-headroom/`](./generation/log-ratio-headroom/) | [`2026-10-14-log-ratio-headroom-prereg`](../docs/measurements/2026-10-14-log-ratio-headroom-prereg.md) |
| **E2: complete the bundle** — softmax half (the log-ratio law, feature 0014) | Do the log-ratio and affine-tie laws close every admitted family with no new false positive? | [M] softmax; layernorm [planned] | Softmax: the completed gate rejects softmax(βx) at β = 2, 0, −1, the 1/n fill, rows rolled and columns reversed. 0 new false positives: 26 loop candidates, 21 A10 groups, 464 wide or masked CPU cases (worst 1.38 against 30). Fast-exp is admitted on the gate's domain (17.2) but rejected on rows that underflow or carry −∞ masks | [`generation/e2-log-ratio-law/`](./generation/e2-log-ratio-law/) | [`2026-10-14-e2-log-ratio-law-prereg`](../docs/measurements/2026-10-14-e2-log-ratio-law-prereg.md) |
| E3: red-team the gate | With a speed objective, no forbidden list and a hidden checker, is the gate hacked? | [planned] | — | — | paper §5 |
| E1: agent-written adherence proofs | Can an agent certify kernel–spec adherence, in which of 7 languages, and how often falsely? | [planned] | 147 sessions designed | [`src/autokernel_pbt/formal/`](../src/autokernel_pbt/formal/) (core, [B]) | [design](../docs/superpowers/specs/2026-10-01-agent-formal-adherence-design.md) |
| E4: formal feedback | Does returning failed checks and unsolved goals improve the agent beyond a better gate? | [planned] | — | — | paper §5 |

## Translation

Known kernels mapped from one setup to another, with PBT and formal proofs as the gate. The first
pair (Triton → CUDA) has run on a device once, in TE1 (2026-10-20, one RTX A6000).
The pipelines (T0–T4) and experiments (TE1–TE4) are in
[`docs/pipelines/translation.md`](../docs/pipelines/translation.md); see
[`translation/README.md`](./translation/README.md).

| Experiment | Question | Status | Headline | Code | Record |
|---|---|---|---|---|---|
| Cross-setup consistency, proved | Do the known Triton ladder kernels and their CUDA ports compute the same function? | [M] | Yes in exact arithmetic, for every row width and power-of-two block width; the Triton tile needs `B ≥ n`, and drops the tail otherwise. Six theorems each in Lean, Rocq and Isabelle/HOL | [`proofs/cross-setup-reductions/`](../proofs/cross-setup-reductions/) | the directory README |
| **TE1**: Triton → CUDA | Do the Tier-1 laws and the cross-setup ratio accept the correct CUDA port, where `allclose` does not? | [M] | All 8 predictions hold (P2 after a telemetry fix made before recording). On a correct port, T1 and the laws reject **0** groups, T1's largest ratio is **0.986**, and T0 rejects **2/9** layernorm groups. Outputs are bitwise equal on 25–45% of reduction elements | `kernels/cuda/`, features 0012–0013, [`translation/te1/`](./translation/te1/) | [`2026-10-12-te1-triton-to-cuda-prereg`](../docs/measurements/2026-10-12-te1-triton-to-cuda-prereg.md) |
| TE2–TE4 | Cross-setup calibration; a translation mutation corpus; the translation loop | [proposed] | — | — | [pipeline page](../docs/pipelines/translation.md) |

The evidence before this segment had code of its own is the cross-backend half of the hardware
runs below, which show Tier-1 laws moving NumPy → Triton unchanged.

## Verification layer (shared by both segments)

### Property-based testing: the oracle comparison

Four arms (`allclose`, normalized reference ratio, declarative law bundle, hybrid), all scored
over byte-identical recorded executions. Code: `src/autokernel_pbt/props/`, `kernels/`,
`scripts/gpu_record.py`.

| Experiment | Status | Headline | Record |
|---|---|---|---|
| False-positive rate on correct kernels | [M] | `allclose` 10/54; the other arms 0/54; a "correct" variant found to be a cancellation bug | [`2026-08-19-false-positive-rate`](../docs/measurements/2026-08-19-false-positive-rate.md), [`2026-08-16-allclose-layernorm-false-positives`](../docs/measurements/2026-08-16-allclose-layernorm-false-positives.md) |
| Detection over mutation corpora | [M] | 20 admitted mutants from 3 corpora, **0** arm disagreements | [`2026-08-18-arm-differentiation-null-result`](../docs/measurements/2026-08-18-arm-differentiation-null-result.md), [`…-blinded-corpus-pilot`](../docs/measurements/2026-08-17-blinded-corpus-pilot.md), [`…-provisional-corpus-pilot`](../docs/measurements/2026-08-17-provisional-corpus-pilot.md) |
| Reference-arm normalization | [M] | `log2 n` over linear `n`; tile-aware bound 3.5× vs 5.9× drift | [`2026-08-17-hybrid-reduction-tree`](../docs/measurements/2026-08-17-hybrid-reduction-tree.md), [`…-normalization-discrepancy-resolved`](../docs/measurements/2026-08-17-normalization-discrepancy-resolved.md) |
| Hardware runs: transfer and scale (NumPy → Triton, A10) | [M] | Laws transfer unchanged up to 29.4 M elements; a tile-tail launch defect found while deriving the tile width, now refused by the launcher | [`2026-08-17-first-hardware-run`](../docs/measurements/2026-08-17-first-hardware-run.md), [`…-at-scale-hardware-run`](../docs/measurements/2026-08-17-at-scale-hardware-run.md) |
| compute-sanitizer viability | [M] | Works only with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | [`2026-08-17-sanitizer-viability-probe`](../docs/measurements/2026-08-17-sanitizer-viability-probe.md) |
| Authoring cost (layernorm) | [M], `n = 1` | Declarative arm 176 lines and 1 serious defect, against 77 lines for the reference arm | [`2026-08-16-layernorm-authoring-cost`](../docs/measurements/2026-08-16-layernorm-authoring-cost.md) |

### Formal verification

| Experiment | Status | Headline | Code | Record |
|---|---|---|---|---|
| Proposition 1, mechanized | [M] | The bundle admits softmax(βx) for all β; the log-ratio law pins softmax. Part (b), and (a) for the temperature family, in Lean 4 without Mathlib, 6 theorems | [`proofs/softmax-bundle/`](../proofs/softmax-bundle/) | [`2026-10-01-softmax-bundle-mechanized`](../docs/measurements/2026-10-01-softmax-bundle-mechanized.md) |
| Why3 real theories, audited | [M] | All 10 `ExpLog` axioms hold for exp(βx); `Square` categorical; trig ×5 alias | [`proofs/why3-real-theories/`](../proofs/why3-real-theories/), [`verification/formal-rigidity/`](./verification/formal-rigidity/) | [`2026-10-02-why3-real-theories-symmetry`](../docs/measurements/2026-10-02-why3-real-theories-symmetry.md) |
| Prover in the toolkit | [ran] | 11 of 24 agent self-reports survive, unscored; not quotable | [`verification/oracle-research/`](./verification/oracle-research/) | [`2026-10-02-prover-in-the-toolkit-prereg`](../docs/measurements/2026-10-02-prover-in-the-toolkit-prereg.md) |
| Formal adherence core (feature 0009) | [B] | Six checks, each the sole catcher of one cheat | [`src/autokernel_pbt/formal/`](../src/autokernel_pbt/formal/) | [design](../docs/superpowers/specs/2026-10-01-agent-formal-adherence-design.md), [ADR 0002](../docs/adr/0002-formal-track.md) |

## Running the experiments' own tests

The experiments' tests sit outside the main suite's `testpaths`, and CI runs them as a separate
step:

```bash
pytest experiments -q
```

## Adding an experiment

1. Pick the segment (`generation/`, `translation/`, `verification/`) and create
   `experiments/<segment>/<name>/` with a README covering the question, how to run it, its
   environment, and its status.
2. Pre-register: write `docs/measurements/YYYY-MM-DD-<name>-prereg.md` with the construction and
   predictions *before* the first run, and append results below its line afterwards. Never edit
   above that line.
3. Copy agent artifacts into the repository the moment each agent finishes. Scratch directories
   under `/private/tmp` are wiped within days, and one experiment lost all 24 sandboxes that way.
4. Add a row to this index and, once results are recorded, to
   [`docs/findings.md`](../docs/findings.md).
5. Keep run directories off synced folders such as OneDrive (see `CLAUDE.md`).
