"""Phase A: deterministic, seeded case-group generation."""

from __future__ import annotations

import warnings

import numpy as np

from autokernel_pbt.props.case import BASE_RELATION, Case, CaseGroup
from autokernel_pbt.props.domain import InputDomain, TensorSpec
from autokernel_pbt.props.relations import RELATIONS, Relation
from autokernel_pbt.props.spec import CaseSpec


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

    def _unexercised_shapes_warning(self, n_groups: int) -> str | None:
        """Boundary shape coverage is the design's main recall mechanism.

        Coverage is "was this shape visited at all", not frequency-weighted, so an
        uneven split across shapes is fine -- only a never-visited shape loses recall.
        ``n_groups == 0`` means "produce nothing" and is explicitly supported, so it
        is silent: there is no coverage to lose when no cases were asked for.

        Returns the warning message, or None if coverage is fine. The caller emits it
        so that ``stacklevel=2`` points at user code rather than at this module.
        """
        if n_groups == 0 or n_groups >= len(self.domain.shapes):
            return None
        unexercised = self.domain.shapes[n_groups:]
        return (
            f"n_groups={n_groups} is fewer than the {len(self.domain.shapes)} shapes in "
            f"domain {self.domain.task_id!r}; these shapes will never be exercised: "
            f"{list(unexercised)}"
        )

    def generate(self, n_groups: int) -> list[CaseGroup]:
        """Generate ``n_groups`` case groups.

        Stability boundary: group *i*'s bytes are a pure function of ``seed``,
        ``i``, and the specs that group actually reads. They do not change when
        ``n_groups`` changes, when an unrelated tensor is added to the domain, or
        when ``relations`` is reordered -- so an expensive recorded corpus stays
        reusable across those edits. They do change when ``seed``, ``i``, that
        group's own tensor/relation specs, or ``shapes`` change. Editing
        ``shapes`` remaps index -> shape, which is a visible semantic change to
        what the domain means rather than an invisible value shift.

        Every group is built by ``group_from_spec``, never alongside it. Two code
        paths producing "the same" group is the drift that would make a regenerated
        case differ from the recorded one by a bit -- and nothing would catch it
        until a shrink reported a minimal case the run had never actually executed.
        """
        if n_groups < 0:
            raise ValueError(f"n_groups must be non-negative, got {n_groups}")
        coverage_warning = self._unexercised_shapes_warning(n_groups)
        if coverage_warning is not None:
            warnings.warn(coverage_warning, stacklevel=2)
        return [self.group_from_spec(self._spec_for(index)) for index in range(n_groups)]

    def _spec_for(self, index: int) -> CaseSpec:
        """The recipe for group ``index`` under this generator's domain and seed."""
        return CaseSpec(
            seed=self.seed,
            task_id=self.domain.task_id,
            group_index=index,
            # Shape-first: cycle through every shape before repeating any.
            shape=self.domain.shapes[index % len(self.domain.shapes)],
            transforms=tuple(self.domain.relations),
        )

    def group_from_spec(self, spec: CaseSpec) -> CaseGroup:
        """Rebuild one case group from its recipe.

        Byte-identical to the original, because the rng is a pure function of
        ``(seed, group_index)`` and the transforms are applied in recorded order.

        A spec with a *reduced* transform list rebuilds the same base case with fewer
        partners, which is the unit move a future shrinker makes. Note the base case
        is unaffected by that reduction: each relation draws from the stream *after*
        the base has been sampled, so dropping a trailing transform cannot perturb it.
        Dropping a non-final one does change what the later relations draw, which is
        why a shrinker must re-execute rather than assume.
        """
        if spec.task_id != self.domain.task_id:
            msg = (
                f"spec is for task {spec.task_id!r} but this generator carries a domain "
                f"for {self.domain.task_id!r}; the rebuilt group would claim an id it "
                f"cannot join back to"
            )
            raise ValueError(msg)
        # One independent stream per group: group i's bytes depend only on
        # (seed, i) and the specs it actually reads -- never on how many groups
        # were requested, nor on unrelated tensors or relations. The list-key
        # form is a pure function of (seed, index), so a single group can be
        # regenerated standalone; rng.spawn() would force a walk of 0..i-1.
        rng = np.random.default_rng([spec.seed, spec.group_index])
        group_id = f"{spec.task_id}-g{spec.group_index:05d}"
        base = Case(
            case_id=f"{group_id}-base",
            group_id=group_id,
            relation=BASE_RELATION,
            task_id=spec.task_id,
            dtype=self.domain.tensors[0].dtype,
            shape=spec.shape,
            tensors={t.name: _sample(t, spec.shape, rng) for t in self.domain.tensors},
        )
        cases = [base]
        for relation_name in spec.transforms:
            cases.append(self._relation(relation_name).derive(base, rng))
        return CaseGroup(group_id=group_id, cases=tuple(cases), spec=spec)
