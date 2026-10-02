# Feature 0009: The formal adherence core

## Problem

An agent's proof that a kernel adheres to its spec certifies nothing by itself when the agent also
wrote the spec, the model of the kernel and the theorem: each can be fitted to make the proof go
through. Before any formal language or agent is involved, the checks that decide whether a proof
counts have to exist, and have to be shown to catch each way of cheating them.

## Scope

1. **The attempt** — what a language adapter hands the harness: the executable readings of the
   agent's spec, model and hypotheses; its axioms; what the language's checker found.
2. **The ground** — what an attempt is judged against: the whole ladder of generated cases, the
   reference, the kernel's own executions, and a hidden set of wrong kernels — refused if it
   cannot judge.
3. **Harness-owned comparisons** — closeness budgeted at the kernel output's dtype and the input's
   accumulation length, never by the spec.
4. **Six checks and the certificate record** — proof, spec soundness, spec completeness, model
   fidelity, non-vacuity, axioms. A certificate is issued only if all six ran and passed.

## Non-goals

- Any formal language or toolchain (feature 0010)
- Any agent run, and persisting certificates (feature 0011)
- Establishing that an adapter's readings come from the proved definitions — the adapter's
  obligation, specified with it

## Acceptance

See [acceptance.yaml](./acceptance.yaml).
