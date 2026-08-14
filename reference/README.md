# Reference Library — Automated GPU Kernel Development

A **tiered reading path** from high-level overview to specialized research. Each level has a `README.md` with ordered paper lists and local PDF links.

```
reference/
├── README.md                 ← you are here
├── manifest.csv              ← flat index of all papers
│
├── L0-start-here/            Survey & big picture
├── L1-foundations/           Compilers, DSLs, expert LLM kernels
├── L2-benchmarks/            How work is evaluated
├── L3-llm-kernel-models/     SFT & RL one-shot generation
├── L4-agentic-search/        Iterative agents & PBT/evolution  ★ autokernel-PBT
└── L5-advanced-topics/       Multi-agent, memory, hardware depth
```

## Recommended path

| Step | Level | Time | Outcome |
|------|-------|------|---------|
| 1 | [L0 — Start Here](./L0-start-here/README.md) | ~1 hr | Map of the field |
| 2 | [L1 — Foundations](./L1-foundations/README.md) | ~4–6 hr | Understand TVM, Triton, FlashAttention baselines |
| 3 | [L2 — Benchmarks](./L2-benchmarks/README.md) | ~2 hr | Know KernelBench, `fast_p`, evaluation pitfalls |
| 4 | [L3 — LLM Models](./L3-llm-kernel-models/README.md) | ~3 hr | How models are trained to emit kernels |
| 5 | [L4 — Agentic Search](./L4-agentic-search/README.md) | ~4 hr | AutoKernel, cuPilot, PBT — **project focus** |
| 6 | [L5 — Advanced](./L5-advanced-topics/README.md) | optional | Deep dives by topic |

## Quick links (autokernel-PBT)

| Paper | Path |
|-------|------|
| Field survey | [L0/papers/2601.15727.pdf](./L0-start-here/papers/2601.15727.pdf) |
| KernelBench | [L2/papers/2502.10517.pdf](./L2-benchmarks/papers/2502.10517.pdf) |
| AutoKernel | [L4/iterative-agents/2603.21331.pdf](./L4-agentic-search/iterative-agents/2603.21331.pdf) |
| Population Based Training | [L4/evolutionary-pbt/1711.09846.pdf](./L4-agentic-search/evolutionary-pbt/1711.09846.pdf) |
| cuPilot | [L4/evolutionary-pbt/2512.16465.pdf](./L4-agentic-search/evolutionary-pbt/2512.16465.pdf) |

## Conventions

- PDFs named `{arxiv_id}.pdf` or descriptive name for non-arXiv (e.g. `triton-mapl2019.pdf`)
- arXiv canonical URL: `https://arxiv.org/abs/{id}`

## Refresh a paper

```bash
curl -fsSL -o path/to/ID.pdf "https://arxiv.org/pdf/ID.pdf"
```
