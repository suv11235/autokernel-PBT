# L1 — Foundations

**Goal:** Understand how high-performance GPU kernels are traditionally designed — the baselines automated methods must beat.

Read subfolders in order: **compiler autotuning → DSLs → LLM workload kernels**.

## 1. Compiler autotuning

Search over schedules / tensor programs instead of hand-writing every CUDA variant.

| Paper | File |
|-------|------|
| Halide (algorithm vs schedule separation) | [compiler-autotuning/halide-pldi2013.pdf](./compiler-autotuning/halide-pldi2013.pdf) |
| TVM (end-to-end DL compiler) | [compiler-autotuning/1802.04799.pdf](./compiler-autotuning/1802.04799.pdf) |
| AutoTVM (learned cost model) | [compiler-autotuning/1805.08166.pdf](./compiler-autotuning/1805.08166.pdf) |
| Ansor (template-free auto-scheduler) | [compiler-autotuning/2006.06762.pdf](./compiler-autotuning/2006.06762.pdf) |
| Meta Schedule (probabilistic search) | [compiler-autotuning/2205.13603.pdf](./compiler-autotuning/2205.13603.pdf) |
| Mirage (multi-level superoptimizer) | [compiler-autotuning/2405.05751.pdf](./compiler-autotuning/2405.05751.pdf) |
| AlphaTensor (RL discovers matmul algorithms) | [compiler-autotuning/alphatensor-nature2022.pdf](./compiler-autotuning/alphatensor-nature2022.pdf) |

## 2. Domain-specific languages

Higher-level languages that compile to GPU code — common **targets** for LLM-generated kernels.

| Paper | File |
|-------|------|
| Triton (tile-centric GPU DSL) | [dsls/triton-mapl2019.pdf](./dsls/triton-mapl2019.pdf) |
| TileLang (composable tiled model) | [dsls/2504.17577.pdf](./dsls/2504.17577.pdf) |

## 3. LLM workload kernels (expert-written)

Canonical hand-tuned kernels for transformer **training** and **serving** — the quality bar.

| Paper | File | Focus |
|-------|------|-------|
| FlashAttention | [llm-kernels/2205.14135.pdf](./llm-kernels/2205.14135.pdf) | Training attention |
| FlashAttention-2 | [llm-kernels/2307.08691.pdf](./llm-kernels/2307.08691.pdf) | Better GPU parallelism |
| FlashAttention-3 | [llm-kernels/2407.08608.pdf](./llm-kernels/2407.08608.pdf) | Hopper / FP8 |
| FlashInfer | [llm-kernels/2501.01005.pdf](./llm-kernels/2501.01005.pdf) | Serving attention engine |
| PagedAttention (vLLM) | [llm-kernels/2309.06180.pdf](./llm-kernels/2309.06180.pdf) | KV-cache + batching |
| MPK (megakernel inference) | [llm-kernels/2512.22219.pdf](./llm-kernels/2512.22219.pdf) | Fused inference megakernel |

## Next

→ [L2 — Benchmarks](../L2-benchmarks/README.md): how automated kernel work is measured.
