"""The callable adapter that lets a Triton kernel satisfy the `Backend` protocol.

`Backend.run` types its kernel as `Callable[..., np.ndarray]` and calls it with the
case's tensors as keyword arguments. A Triton kernel is not that: it needs a launch
grid, constexpr block sizes, and device tensors. Rather than widen the protocol --
which would make the NumPy and Triton backends stop being substitutable, and
substitutability is what makes cross-backend comparison possible -- the *kernel* is
wrapped in an object that is callable in exactly the way the protocol expects.

The adapter additionally owns two things the backend needs and the protocol has no
place for: the launch configuration actually used, and the compiled artifact telemetry
is read from. Triton compiles lazily, so the artifact does not exist until after the
first call.

`launcher` is injected rather than built here. On CPU there is no Triton to build one
with, and injecting it is what makes every structural behaviour in this module
testable without a GPU.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from autokernel_pbt.props.backends.base import single_output

HASH_CHARS = 16


class InputMutatedError(RuntimeError):
    """A kernel wrote to one of its input buffers.

    WHY THIS IS NOT CHECKED IN THE BACKEND, which is where the CPU equivalent lives.
    `readonly_inputs` protects the *host* array, and a Triton kernel never touches it
    -- it writes to the device copy. The backend holds only host arrays, so a
    before/after hash there is structurally incapable of firing: it would compare two
    buffers the kernel could not have reached, pass every time, and read as a
    guarantee. That is strictly worse than no check at all.

    The device tensors exist only inside the launcher, so the check must live there.
    That makes it the launcher author's responsibility, which is weaker than an
    enforced invariant -- `device_digest` exists to make doing it a one-liner, and the
    ladder launchers all do. A launcher that omits it loses the guarantee for its own
    kernel rather than silently weakening it for everyone.

    Left unchecked, the failure is quiet and severe: an oracle recomputing a reference
    from a corrupted input can agree with an output computed from that same
    corruption, and both arms record a clean pass.
    """


def device_digest(tensor: Any) -> str:
    """Content digest of a device tensor, for the input-mutation check.

    Computed on device -- copying the buffer back to host twice per launch would cost
    more than the launch. A plain sum would miss a permutation, so it is paired with a
    position-weighted sum, which makes reordering visible for one extra reduction.
    """
    import torch

    with torch.no_grad():
        flat = tensor.reshape(-1).to(torch.float64)
        weights = torch.arange(1, flat.numel() + 1, device=flat.device, dtype=torch.float64)
        return f"{float(flat.sum()):.17g}:{float((flat * weights).sum()):.17g}"


class TritonKernel:
    """One Triton kernel, callable as `kernel(**numpy_inputs) -> np.ndarray`."""

    def __init__(
        self,
        kernel_id: str,
        jit_fn: Callable[..., Any],
        grid: Callable[[tuple[int, ...], dict[str, Any]], tuple[int, ...]],
        constexprs: dict[str, Any],
        launcher: Callable[..., Any],
    ) -> None:
        self.kernel_id = kernel_id
        self.jit_fn = jit_fn
        self.grid = grid
        self.constexprs = dict(constexprs)
        self.launcher = launcher
        self.compiled: Any = None
        self._grid_used: tuple[int, ...] | None = None

    @property
    def source_hash(self) -> str:
        """Content identity of what will actually run.

        `kernel_id` is a label. Two runs must not be able to both call something
        `relu_triton` and mean different code, which would silently merge two
        variants' results into one number. Both the jit function and the launcher are
        hashed, because the launcher carries the block and stride arithmetic and a
        change there changes the kernel as surely as editing its body.
        """
        material = []
        for fn in (self.jit_fn, self.launcher):
            try:
                material.append(inspect.getsource(fn))
            except (OSError, TypeError):
                module = getattr(fn, "__module__", "?")
                qualname = getattr(fn, "__qualname__", repr(fn))
                material.append(f"{module}.{qualname}")
        digest = hashlib.sha256("\n".join(material).encode("utf-8"))
        return digest.hexdigest()[:HASH_CHARS]

    def _record_compiled(self, compiled: Any) -> None:
        self.compiled = compiled

    def launch_telemetry(self) -> dict[str, Any]:
        """The launch group of the telemetry schema, as actually used."""
        return {
            "grid": list(self._grid_used) if self._grid_used is not None else None,
            "constexprs": dict(self.constexprs),
        }

    def __call__(self, **inputs: np.ndarray) -> np.ndarray:
        # Copy, do not alias. `readonly_inputs` flips the host arrays non-writeable
        # for the duration of execution, and `torch.from_numpy` warns on a
        # non-writeable array -- which this project turns into an error. The copy is
        # negligible next to a host-to-device transfer, and `base.readonly_inputs`
        # names this as Phase 3's hazard to solve.
        writable = {name: np.array(value, copy=True) for name, value in inputs.items()}
        primary = next(iter(writable.values()))
        grid = tuple(self.grid(primary.shape, self.constexprs))
        self._grid_used = grid
        # The launcher is HANDED the grid and constexprs rather than recomputing them.
        # If it computed its own, the recorded launch telemetry would be a label next
        # to the behaviour rather than a description of it, and the two could drift
        # apart silently -- with the artifacts reporting a grid that never launched.
        #
        # `record_compiled` is a callback rather than a return value because Triton's
        # launch returns the CompiledKernel *alongside* nothing else useful, and
        # threading it back through the output would make every launcher return a
        # tuple for the benefit of telemetry. Without this the artifact is never
        # captured and every compiled field reads MISSING -- which is what the first
        # hardware run actually found.
        return single_output(
            self.launcher(
                grid=grid,
                constexprs=self.constexprs,
                record_compiled=self._record_compiled,
                **writable,
            )
        )
