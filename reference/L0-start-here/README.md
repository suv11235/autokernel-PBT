# L0 — Start Here

**Goal:** Understand the problem space before diving into papers.

## Read first

| Order | Paper | Why |
|-------|-------|-----|
| 1 | [2601.15727 — Survey: Automated Kernel Generation in the Era of LLMs](./papers/2601.15727.pdf) | Single map of the whole field: compilers → LLMs → agents → benchmarks |

## What you should take away

- GPU **kernels** are the hot loops (matmul, attention, norm, MoE dispatch) that bound LLM train/infer cost.
- **Manual** expert kernels (FlashAttention, CUTLASS) set the performance bar.
- **Compiler autotuning** (TVM/Ansor) searches schedules without writing CUDA by hand.
- **LLM generation** writes kernels from PyTorch specs; still hard to beat `torch.compile`.
- **Agent loops** iterate with compile/run/profile feedback — the line autokernel-PBT extends.

## Next

→ [L1 — Foundations](../L1-foundations/README.md): how kernels are built and optimized *before* LLMs.
