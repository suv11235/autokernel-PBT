# L4 — Agentic Kernel Search

**Goal:** Systems that **iterate** — generate, compile, test, profile, keep/revert — rather than one-shot codegen. This is the direct context for **autokernel-PBT**.

Read in order: **iterative agents → evolutionary / PBT**.

## 4a — Iterative agent loops

| Paper | File | Highlight |
|-------|------|-----------|
| CUDA-LLM | [iterative-agents/2506.09092.pdf](./iterative-agents/2506.09092.pdf) | GPU specs + runtime feedback in loop |
| TritonForge | [iterative-agents/2512.09196.pdf](./iterative-agents/2512.09196.pdf) | Profiling-guided Triton refinement |
| **AutoKernel** | [iterative-agents/2603.21331.pdf](./iterative-agents/2603.21331.pdf) | **Project baseline** — Amdahl-ranked bottlenecks, 300+ experiments |
| Astra | [iterative-agents/2509.07506.pdf](./iterative-agents/2509.07506.pdf) | Multi-agent; production SGLang kernels |
| PRAGMA | [iterative-agents/2511.06345.pdf](./iterative-agents/2511.06345.pdf) | NCU metrics → NL suggestions |
| KernelBand | [iterative-agents/2511.18868.pdf](./iterative-agents/2511.18868.pdf) | Bandit over optimization strategies |
| MaxCode | [iterative-agents/2601.05475.pdf](./iterative-agents/2601.05475.pdf) | Max-reward RL unification |
| K-Search | [iterative-agents/2602.19128.pdf](./iterative-agents/2602.19128.pdf) | LLM world model guides search |

## 4b — Evolutionary & population-based search

| Paper | File | Highlight |
|-------|------|-----------|
| **Population Based Training** | [evolutionary-pbt/1711.09846.pdf](./evolutionary-pbt/1711.09846.pdf) | **PBT theory** — exploit/explore populations |
| EvoEngineer | [evolutionary-pbt/2510.03760.pdf](./evolutionary-pbt/2510.03760.pdf) | Formalizes traverse × population management |
| **cuPilot** | [evolutionary-pbt/2512.16465.pdf](./evolutionary-pbt/2512.16465.pdf) | Strategy-level crossover + roofline prompts |
| FM Agent | [evolutionary-pbt/2510.26144.pdf](./evolutionary-pbt/2510.26144.pdf) | Multi-population evolution |
| Kernel-Smith | [evolutionary-pbt/2603.28342.pdf](./evolutionary-pbt/2603.28342.pdf) | Population archive + RL from trajectories |

## Next

→ [L5 — Advanced topics](../L5-advanced-topics/README.md): multi-agent orchestration, memory, hardware-specific depth.
