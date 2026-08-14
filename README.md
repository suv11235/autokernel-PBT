# autokernel-PBT

**Property-based testing (PBT) for automated kernel development** — spec-driven and test-driven from day one.

Kernel correctness is checked by generating inputs and evaluating **properties** — algebraic laws, metamorphic relations, and reference comparisons — rather than by a fixed set of hand-written cases. The scope is framework-agnostic: CUDA and Triton today, AWS Trainium (NKI) and other accelerator stacks by design.

Two workstreams share the property layer:

- **New kernel development** — generate and validate kernels against portable properties.
- **Kernel translation** — port a kernel between backends, using shared properties as the equivalence contract.

## Development model

```
specs/features/<id>/spec.md          → human-readable requirement
specs/features/<id>/acceptance.yaml  → machine-checkable criteria
tests/**                             → written against acceptance (TDD)
src/autokernel_pbt/**                → implementation
harness/bench.py                     → fixed evaluation entrypoint
```

See [specs/README.md](./specs/README.md) for the full workflow.

## Layout

| Path | Purpose |
|------|---------|
| `specs/` | Feature specs, schemas, architecture notes |
| `contracts/` | Stable interfaces between harness ↔ kernels |
| `src/autokernel_pbt/` | Library code |
| `harness/` | Correctness + benchmark harness (agent-evaluable) |
| `kernels/` | Candidate kernel implementations (Triton / CUDA) |
| `tests/` | Unit, integration, spec-derived tests |
| `reference/` | Tiered paper library ([README](./reference/README.md)) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                    # unit + spec tests (no GPU)
pytest -m gpu             # requires CUDA + torch
python harness/bench.py --help
```

## Agent commits

Cursor agents must **not** use plain `git commit` (adds `Co-authored-by: Cursor` to history). Use the project skill [`.cursor/skills/clean-git-commits/`](./.cursor/skills/clean-git-commits/SKILL.md) or `scripts/git_commit_clean.sh` after staging.

## Reference library

Local PDFs live under `reference/L0-start-here/` … `L5-advanced-topics/`. PDFs are gitignored; clone then refresh via `reference/manifest.csv` if needed.

## License

MIT
