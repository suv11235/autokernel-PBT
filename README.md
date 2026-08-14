# autokernel-PBT

Population-based training (PBT) for automated GPU kernel search — spec-driven and test-driven from day one.

Extends the [AutoKernel](https://arxiv.org/abs/2603.21331) agent loop with **populations** of kernel candidates (exploit best, explore mutations) instead of single-trajectory refinement.

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
| `contracts/` | Stable interfaces between harness ↔ PBT ↔ kernels |
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

Local PDFs live under `reference/L0-start-here/` … `L5-advanced-topics/`. PDFs are gitignored (~160 MB); clone then refresh via `reference/manifest.csv` if needed.

## License

MIT
