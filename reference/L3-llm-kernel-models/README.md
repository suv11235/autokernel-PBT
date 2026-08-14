# L3 — LLM Kernel Models (SFT & RL)

**Goal:** Models specialized to **generate** CUDA/Triton kernels — typically one-shot or fixed multi-turn RL, not open-ended agent search.

## 3a — Supervised fine-tuning

| Paper | File |
|-------|------|
| ConCuR → KernelCoder | [papers/2510.07356.pdf](./papers/2510.07356.pdf) |
| InCoder-32B | [papers/2603.16790.pdf](./papers/2603.16790.pdf) |

## 3b — Reinforcement learning

| Paper | File | Highlight |
|-------|------|-----------|
| Kevin (multi-turn RL) | [papers/2507.11948.pdf](./papers/2507.11948.pdf) | Cross-turn credit assignment |
| AutoTriton | [papers/2507.05687.pdf](./papers/2507.05687.pdf) | Triton + structural/runtime rewards |
| TritonRL | [papers/2510.17891.pdf](./papers/2510.17891.pdf) | Hierarchical rewards |
| QiMeng-Kernel | [papers/2511.20100.pdf](./papers/2511.20100.pdf) | Macro-thinking / micro-coding |
| CUDA-L2 | [papers/2512.02551.pdf](./papers/2512.02551.pdf) | Contrastive RL; GEMM vs cuBLAS |
| Dr. Kernel | [papers/2602.05885.pdf](./papers/2602.05885.pdf) | Distributed Triton RL |
| **CUDA Agent** | [papers/2602.24286.pdf](./papers/2602.24286.pdf) | SOTA KernelBench via agentic RL |
| AscendKernelGen | [papers/2601.07160.pdf](./papers/2601.07160.pdf) | NPU / AscendC |
| QiMeng-TensorOp | [papers/2505.06302.pdf](./papers/2505.06302.pdf) | Hardware-primitive prompts |

## 3c — Hybrid (RL + evolution training data)

| Paper | File |
|-------|------|
| Kernel-Smith | [papers/2603.28342.pdf](./papers/2603.28342.pdf) *(also in [L4/evolutionary-pbt](../L4-agentic-search/evolutionary-pbt/))* |

## Next

→ [L4 — Agentic search](../L4-agentic-search/README.md): closed-loop optimize with execution feedback (autokernel-PBT core).
