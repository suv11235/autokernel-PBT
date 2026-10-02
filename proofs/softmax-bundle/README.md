# Proofs about the oracle: the softmax property bundle

`SoftmaxBundle.lean` mechanizes Proposition 1 (a) and (b) of the paper — why the gate's
softmax bundle admits softmax(βx) at every β, and that adding the log-ratio law closes the
gap — in **core Lean 4, with no Mathlib**. It then proves two things the paper did not state:

| theorem | says | assumes about `exp` |
|---|---|---|
| `softmax_unique` | row sums + the log-ratio law ⇒ the row is `softmaxRow xs` | **nothing** — pure field algebra |
| `softmaxBeta_satisfies_bundle` | softmax(βx) satisfies ROW_SUMS, UNIT_INTERVAL and SHIFT_INVARIANCE for every β | `exp_pos`, `exp_add` |
| `bundle_incomplete` | for β ≠ 1, softmax(βx) ≠ softmax(x) on the row (0, 1) | + `exp_inj` |
| `expAxioms_scaled` | those three axioms also hold for `x ↦ exp (β x)`, β ≠ 0 | — |
| `softmaxRowWith_scaled` | softmax w.r.t. the rescaled `exp` *is* softmax(βx) | — |
| `tangent_rigid` | if `exp` and `x ↦ exp (β x)` both satisfy `1 + x ≤ exp x`, then β = 1 | `exp_pos`, `exp_add` |

Read together, the last three say: a proof that a kernel is "softmax with respect to `exp`"
carried out from the natural first-order axioms of `exp` is **temperature-blind** — it is equally
a proof about softmax(βx) — exactly as the test bundle is. Two things break the symmetry: the
executable reading, which instantiates `exp` at the real one (the certificate's checks C2–C3),
or one more first-order axiom, the tangent line, which C6 can sample.

See `docs/measurements/2026-10-01-softmax-bundle-mechanized.md` for the record and what is
*not* claimed (floats, layernorm, permutations).

## Check it

No `lake`, no project, no downloads. The toolchain is pinned by `lean-toolchain`; call that
toolchain's binary directly (a bare `lean` may trigger elan to download a newer release):

```bash
~/.elan/toolchains/leanprover--lean4---v4.33.0/bin/lean proofs/softmax-bundle/SoftmaxBundle.lean
```

Exit code 0, no `sorry`, and every `#print axioms` line lists only `propext`,
`Classical.choice` and `Quot.sound`. `tests/proofs/test_softmax_bundle_lean.py` runs exactly
this and is skipped when the toolchain is absent (marker `formal`, as `gpu` tests are).

Compile time: about one second.
