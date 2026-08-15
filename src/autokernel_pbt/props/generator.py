"""Phase A: deterministic, seeded case-group generation."""

from __future__ import annotations

import numpy as np

from autokernel_pbt.props.case import BASE_RELATION, Case, CaseGroup
from autokernel_pbt.props.domain import InputDomain, TensorSpec
from autokernel_pbt.props.relations import RELATIONS, Relation


def _sample(spec: TensorSpec, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    dtype = spec.numpy_dtype()
    if spec.distribution == "normal":
        values = rng.normal(0.0, 1.0, size=shape)
    elif spec.distribution == "uniform":
        values = rng.uniform(spec.low, spec.high, size=shape)
    elif spec.distribution == "zeros":
        values = np.zeros(shape)
    elif spec.distribution == "ones":
        values = np.ones(shape)
    else:  # pragma: no cover - guarded by TensorSpec.__post_init__
        raise ValueError(f"unsupported distribution {spec.distribution!r}")
    return values.astype(dtype)


class Generator:
    """Produces case groups deterministically from a domain and a seed."""

    def __init__(self, domain: InputDomain, seed: int) -> None:
        self.domain = domain
        self.seed = seed

    def _relation(self, name: str) -> Relation:
        """Look up a relation, failing loudly on a typo rather than with a bare KeyError."""
        factory = RELATIONS.get(name)
        if factory is None:
            msg = (
                f"unknown relation {name!r} in domain {self.domain.task_id!r}; "
                f"available relations: {sorted(RELATIONS)}"
            )
            raise ValueError(msg)
        return factory()

    def generate(self, n_groups: int) -> list[CaseGroup]:
        rng = np.random.default_rng(self.seed)
        groups: list[CaseGroup] = []
        for index in range(n_groups):
            # Shape-first: cycle through every shape before repeating any.
            shape = self.domain.shapes[index % len(self.domain.shapes)]
            group_id = f"{self.domain.task_id}-g{index:05d}"
            base = Case(
                case_id=f"{group_id}-base",
                group_id=group_id,
                relation=BASE_RELATION,
                task_id=self.domain.task_id,
                dtype=self.domain.tensors[0].dtype,
                shape=shape,
                tensors={t.name: _sample(t, shape, rng) for t in self.domain.tensors},
            )
            cases = [base]
            for relation_name in self.domain.relations:
                cases.append(self._relation(relation_name).derive(base, rng))
            groups.append(CaseGroup(group_id=group_id, cases=tuple(cases)))
        return groups
