# Kernels

Kernels under test, per-task contracts, and the mutation corpus. NumPy references live in
`src/autokernel_pbt/props/tasks.py`.

## Layout

```
kernels/
├── triton/ladder.py # stock Triton kernels for the ladder tasks
├── mutants/         # blinded/grown mutation corpus + correct-but-different variants
├── tasks/<task>/acceptance.yaml   # the declarative arm's contract per task
└── cuda/            # CUDA extensions (future; empty)
```

Mutant bodies are recorded verbatim from blinded authoring agents. Do not edit them — see
`docs/protocol/mutant-authoring.md`.
