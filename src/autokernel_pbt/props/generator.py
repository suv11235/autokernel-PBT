"""Phase A: deterministic, seeded case-group generation."""

from __future__ import annotations

import operator
import warnings

import numpy as np

from autokernel_pbt.props.case import BASE_RELATION, Case, CaseGroup
from autokernel_pbt.props.domain import InputDomain, TensorSpec
from autokernel_pbt.props.relations import RELATIONS, Relation
from autokernel_pbt.props.spec import CaseSpec

#: Seeds must fit one 32-bit word. The per-group stream is keyed by the list
#: ``[seed, group_index]``, and numpy splits a wider int into several 32-bit words,
#: so ``[2**32 + 5, 0]`` is the word list ``[5, 1, 0]`` -- and trailing zero words do
#: not change a seed sequence, making it the same stream as ``[5, 1]``. Bounding the
#: seed to one word makes (seed, index) -> stream injective again WITHOUT changing
#: the key for any seed already in use, so every recorded run stays regenerable.
SEED_BOUND = 2**32


def _checked_seed(seed: object) -> int:
    """The seed as a Python int in ``[0, SEED_BOUND)``, or a loud refusal.

    ``operator.index`` accepts numpy integers (``rng.integers`` returns ``np.int64``,
    which ``json`` cannot serialize into a ``CaseSpec``) and rejects floats. ``bool``
    is refused separately because ``operator.index(True) == 1``: a flag in the seed
    slot is a call-site mistake, and accepting it would record a run under a seed
    nobody chose. The value, and therefore the stream, is unchanged by coercion.
    """
    if isinstance(seed, (bool, np.bool_)):
        msg = f"seed must be an integer, not bool ({seed})"
        raise TypeError(msg)
    try:
        value = operator.index(seed)
    except TypeError:
        msg = f"seed must be an integer, got {seed!r} ({type(seed).__name__})"
        raise TypeError(msg) from None
    if not 0 <= value < SEED_BOUND:
        msg = (
            f"seed must be in [0, 2**32), got {value}; the per-group stream is keyed by "
            f"[seed, group_index], and a wider seed spills into the index's 32-bit word, "
            f"so two different (seed, index) pairs would draw byte-identical groups"
        )
        raise ValueError(msg)
    return value


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
        self.seed = _checked_seed(seed)

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

        Stability boundary. Group *i* draws from its own stream,
        ``default_rng([seed, i])``, and consumes it in one fixed order: every tensor
        of ``domain.tensors`` in declaration order, then every relation of
        ``relations`` in list order. That order decides what survives an edit:

        * **The group count and other groups.** Group *i* is a pure function of
          ``(seed, i, domain)``; ``generate(10)[:4]`` equals ``generate(4)``, and one
          group can be rebuilt alone (``group_from_spec``).
        * **Base tensors.** A base tensor depends on its own spec, its shape, and the
          stream position the tensors declared *before* it left behind. Appending a
          tensor, or any change to ``relations``, leaves every existing base tensor
          byte-identical. Inserting a tensor ahead of it can shift it, and so can
          changing the distribution of one declared before it: ``zeros``/``ones``
          draw nothing, ``uniform`` one word per element, ``normal`` a
          data-dependent number.
        * **Relation partners are NOT stable under domain edits.** A partner is
          derived from the base's values and draws from the stream after every base
          tensor and after the relations listed before it. So it changes when a
          drawing tensor (anything but ``zeros``/``ones``) is added *anywhere* --
          appended included, even though the base it derives from is untouched --
          when ``relations`` is reordered, or when a non-final relation is inserted
          or dropped; and it can change when any tensor's distribution does. Only
          appending or dropping a *trailing* relation leaves the earlier partners
          intact.
        * **Shapes.** Editing ``shapes`` remaps index -> shape, a visible change to
          what the domain means rather than an invisible value shift.

        An earlier version of this docstring also promised the partners were stable
        under an added tensor or reordered relations. They are not, and the stream
        was deliberately NOT re-derived to make them so: the GPU runs recorded at
        seed 42 must stay byte-identically regenerable, and any change to how a
        group's stream is derived or consumed would silently orphan all of them.
        ``test_generator.py::test_the_recorded_corpus_is_regenerated_byte_for_byte``
        pins that.

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

        Byte-identical to the original under the same domain, because the rng is a
        pure function of ``(seed, group_index)`` and the transforms are applied in
        recorded order. The stream is keyed by ``spec.seed``, not ``self.seed``, so
        the spec's seed is range-checked here too: a check in the constructor alone
        would leave the wide-seed collision reachable through a spec.

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
        _checked_seed(spec.seed)
        # One independent stream per group: group i's bytes depend only on
        # (seed, i) and the domain -- never on how many groups were requested or on
        # any other group. Which domain edits it survives is spelled out in
        # `generate`. The list-key form is a pure function of (seed, index), so a
        # single group can be regenerated standalone; rng.spawn() would force a walk
        # of 0..i-1. DO NOT CHANGE THIS KEY OR THE DRAW ORDER BELOW: recorded runs
        # are regenerated from exactly this stream.
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
