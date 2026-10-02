"""What an attempt is judged against: the cases, the reference, the kernel's own executions, and
wrong kernels the agent never sees.

Realized once, before any check runs, and refused if it cannot judge. A ground with no wrong
kernels would let spec completeness pass vacuously, and a wrong kernel never wrong by more than
the rounding budget would ask a spec to reject what the harness itself cannot tell from the
reference. Both are bad *calls* — cheap to fix and silent if accepted — so both raise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from autokernel_pbt.formal.comparisons import Comparisons
from autokernel_pbt.props.backends.base import OUTPUT_NAME, Status, kernel_inputs
from autokernel_pbt.props.backends.numpy_backend import NumpyBackend
from autokernel_pbt.props.case import Case
from autokernel_pbt.props.generator import Generator
from autokernel_pbt.props.tasks import Task

Kernel = Callable[..., np.ndarray]


@dataclass(frozen=True)
class Ground:
    """The judge's side of one (task, kernel) pair. Never shown to the agent.

    ``primitives`` are the functions an axiom may be about: those the informal spec itself uses,
    such as ``exp`` for softmax.
    """

    task: Task
    reference: Kernel
    kernel: Kernel
    hidden: Mapping[str, Kernel]
    primitives: frozenset[str]
    seed: int = 42


@dataclass(frozen=True)
class CaseRecord:
    """One generated case, with every output a check may compare against.

    Every array is a read-only copy, and ``inputs`` sit in a read-only mapping, so a reading that
    writes into an array in the ordinary way, or rebinds a name, raises instead of silently
    changing what every later check sees. Outputs matter as much as inputs here: the spec is
    handed the reference and every hidden output, and the reference anchors every comparison
    after it. Copies rather than flags on the kernels' own arrays, so a kernel that reuses an
    output buffer is not broken on its next call. ``kernel`` is ``None`` where the kernel under
    proof failed to run: there is nothing there for a model to be faithful to.
    """

    case_id: str
    inputs: Mapping[str, np.ndarray]
    reference: np.ndarray
    kernel: np.ndarray | None
    hidden: Mapping[str, np.ndarray]


def realize(ground: Ground) -> tuple[CaseRecord, ...]:
    """Run the reference, the kernel and every hidden kernel over the whole ladder.

    The whole ladder, always: fewer groups than shapes leaves rungs unexercised, which the
    generator warns about, and the degenerate rungs are exactly where wrong kernels look right.
    """
    if not ground.hidden:
        msg = (
            "no hidden wrong kernels: with nothing to reject, spec completeness would pass "
            "vacuously"
        )
        raise ValueError(msg)
    backend = NumpyBackend()
    groups = Generator(ground.task.domain, ground.seed).generate(len(ground.task.domain.shapes))
    never_wrong = set(ground.hidden)
    records = []
    for group in groups:
        for case in group.cases:
            inputs = MappingProxyType(_read_only_copies(kernel_inputs(case)))
            reference = _reference(ground.reference, inputs, case)
            hidden = {}
            for name, wrong in ground.hidden.items():
                output = _run(backend, wrong, case)
                if output is None:
                    msg = (
                        f"hidden kernel {name!r} failed on case {case.case_id!r}; the ground "
                        f"cannot use it"
                    )
                    raise ValueError(msg)
                # At the reference's dtype — the task's precision — and never at whatever the
                # hidden kernel returned: a correct float64 twin is "wrong" against float64's far
                # tighter budget, and would then fail every honest spec for accepting it.
                if not Comparisons.for_output(inputs, reference.dtype).close(output, reference):
                    never_wrong.discard(name)
                hidden[name] = output
            kernel = _run(backend, ground.kernel, case)
            records.append(CaseRecord(case.case_id, inputs, reference, kernel, hidden))
    if never_wrong:
        msg = (
            f"hidden kernel(s) {sorted(never_wrong)} are never wrong by more than the rounding "
            f"budget; no spec can be asked to reject what the harness cannot tell from the "
            f"reference"
        )
        raise ValueError(msg)
    return tuple(records)


def _read_only_copy(array: np.ndarray) -> np.ndarray:
    copy = np.array(array, copy=True)
    copy.flags.writeable = False
    return copy


def _read_only_copies(inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: _read_only_copy(array) for name, array in inputs.items()}


def _reference(reference: Kernel, inputs: Mapping[str, np.ndarray], case: Case) -> np.ndarray:
    # Suppressed, then checked, as the reference arm does: under filterwarnings=error an overflow
    # would abort the run, and in production it would pass silently as inf.
    with np.errstate(all="ignore"):
        expected = np.asarray(reference(**inputs))
    if expected.dtype.kind in "fc" and not np.all(np.isfinite(expected)):
        msg = f"the reference is non-finite on case {case.case_id!r}; it cannot anchor any check"
        raise ValueError(msg)
    return _read_only_copy(expected)


def _run(backend: NumpyBackend, kernel: Kernel, case: Case) -> np.ndarray | None:
    result = backend.run(kernel, case)
    if result.status != Status.OK or OUTPUT_NAME not in result.outputs:
        return None
    return _read_only_copy(result.outputs[OUTPUT_NAME])
