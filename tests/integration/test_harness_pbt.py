"""Integration tests (skeleton)."""

import pytest

from autokernel_pbt.pbt.fitness import fitness_from_harness_result
from autokernel_pbt.pbt.population import KernelCandidate
from autokernel_pbt.pbt.scheduler import PBTConfig, PBTScheduler
from autokernel_pbt.harness.runner import run_harness


@pytest.mark.integration
def test_harness_to_pbt_fitness_chain(repo_root):
    result = run_harness(
        str(repo_root / "kernels/triton/candidate.py"),
        str(repo_root / "kernels/triton/reference_relu.py"),
        dry_run=True,
    )
    fitness = fitness_from_harness_result(result)
    member = KernelCandidate(
        id="m1",
        source_path="kernels/triton/candidate.py",
        backend="triton",
        generation=0,
        fitness=fitness,
    )
    scheduler = PBTScheduler([member], config=PBTConfig(population_size=3))
    next_gen = scheduler.step()
    assert len(next_gen) == 3
