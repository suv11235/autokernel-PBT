# Kernels

Candidate and reference implementations. Each task directory should contain:

- `reference.py` — PyTorch (or spec) baseline
- `candidate.py` — optimized kernel under test
- `README.md` — shapes, dtypes, operator name

## Layout

```
kernels/
├── triton/          # Triton @triton.jit kernels
├── cuda/            # CUDA extensions (future)
└── tasks/           # per-operator folders (KernelBench-style, future)
```

## Skeleton task

See `triton/reference_relu.py` and `triton/candidate.py`.
