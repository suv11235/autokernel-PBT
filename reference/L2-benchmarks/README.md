# L2 — Benchmarks & Evaluation

**Goal:** Understand how kernel-generation systems are scored before reading method papers.

## Core idea

Most LLM kernel benchmarks report:

1. **Correctness** — output matches PyTorch reference within tolerance  
2. **Speedup** — wall time vs eager or `torch.compile`  
3. **`fast_p`** — fraction correct *and* faster than p× baseline (KernelBench)

## Papers (read in this order)

| Order | Paper | File | Notes |
|-------|-------|------|-------|
| 1 | ParEval | [papers/2401.12554.pdf](./papers/2401.12554.pdf) | Early parallel/CUDA codegen eval (420 tasks) |
| 2 | **KernelBench** | [papers/2502.10517.pdf](./papers/2502.10517.pdf) | **Primary benchmark** — 250 PyTorch→CUDA tasks |
| 3 | TritonBench | [papers/2502.14752.pdf](./papers/2502.14752.pdf) | Triton operator generation |
| 4 | MultiKernelBench | [papers/2507.17773.pdf](./papers/2507.17773.pdf) | Multi-platform (GPU/NPU/TPU) |
| 5 | FlashInfer-Bench | [papers/2601.00227.pdf](./papers/2601.00227.pdf) | Production inference kernels |
| 6 | SOL-ExecBench | [papers/2603.19173.pdf](./papers/2603.19173.pdf) | Speed-of-light hardware limits |
| 7 | Robust agentic CUDA bench | [papers/2509.14279.pdf](./papers/2509.14279.pdf) | Robustness + baseline comparisons |

## Next

→ [L3 — LLM kernel models](../L3-llm-kernel-models/README.md): training models to emit kernels in one shot.
