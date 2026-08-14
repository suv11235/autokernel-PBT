"""Population-based training scheduler (skeleton)."""

from __future__ import annotations

from dataclasses import dataclass

from autokernel_pbt.pbt.mutation import mutate
from autokernel_pbt.pbt.population import KernelCandidate


@dataclass
class PBTConfig:
    population_size: int = 8
    exploit_fraction: float = 0.25
    generation: int = 0


class PBTScheduler:
    def __init__(self, members: list[KernelCandidate], config: PBTConfig | None = None):
        self.config = config or PBTConfig()
        self.members = members[: self.config.population_size]
        while len(self.members) < self.config.population_size:
            seed = self.members[0] if self.members else _default_seed()
            self.members.append(mutate(seed, generation=0))

    def step(self) -> list[KernelCandidate]:
        self.config.generation += 1
        ranked = sorted(self.members, key=lambda m: m.fitness, reverse=True)
        n_exploit = max(1, int(self.config.population_size * self.config.exploit_fraction))
        exploiters = ranked[:n_exploit]

        next_gen: list[KernelCandidate] = []
        for parent in exploiters:
            child = mutate(parent, generation=self.config.generation)
            next_gen.append(child)

        while len(next_gen) < self.config.population_size:
            parent = exploiters[len(next_gen) % len(exploiters)]
            next_gen.append(mutate(parent, generation=self.config.generation))

        self.members = next_gen[: self.config.population_size]
        return self.members


def _default_seed() -> KernelCandidate:
    return KernelCandidate(
        id="seed",
        source_path="kernels/triton/reference_relu.py",
        backend="triton",
        generation=0,
    )
