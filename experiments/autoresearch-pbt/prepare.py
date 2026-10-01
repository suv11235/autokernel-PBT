"""Fixed constants, data prep and evaluation. NOT modified by agents.

The autoresearch contract, transplanted to CPU/NumPy:

* training runs for a **fixed wall-clock budget**, not a fixed step count, so any
  change that makes a step cheaper converts directly into more steps;
* the metric is **val_bpb** -- validation bits per byte. The tokenizer is the
  identity on bytes (vocab 256, one token == one byte), so bpb is
  ``cross_entropy_nats / ln(2)`` with no vocabulary correction to argue about,
  and it stays comparable across every architectural change an agent makes;
* evaluation is over a **fixed** set of validation windows at fixed offsets, so
  two runs are compared on identical bytes.

The one thing this file fixes that karpathy's does not is ``BLOCK_SIZE``: it is the
context length the evaluation grants the model, so letting an agent change it would
change what val_bpb measures. Everything else about the model is fair game.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "input.txt"

VOCAB_SIZE = 256          # raw bytes; one token is one byte, so loss/ln2 IS bpb
BLOCK_SIZE = 128          # eval context length -- FIXED, see docstring
VAL_FRACTION = 0.1
N_EVAL_WINDOWS = 40       # fixed windows, fixed offsets -> deterministic val_bpb
EVAL_SEED = 1337
TRAIN_SECONDS = 60.0      # the budget; excludes setup, compile-equivalent, and eval

LN2 = math.log(2.0)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Byte arrays for train and val, split at a fixed point."""
    raw = np.frombuffer(DATA.read_bytes(), dtype=np.uint8)
    split = int(len(raw) * (1.0 - VAL_FRACTION))
    return raw[:split].copy(), raw[split:].copy()


def eval_windows(val: np.ndarray) -> np.ndarray:
    """The fixed evaluation offsets. Deterministic in EVAL_SEED alone.

    Returned as an int array of start indices; every window is ``BLOCK_SIZE + 1``
    bytes so the targets are the inputs shifted by one.
    """
    rng = np.random.default_rng(EVAL_SEED)
    high = len(val) - BLOCK_SIZE - 1
    return rng.integers(0, high, size=N_EVAL_WINDOWS, dtype=np.int64)


def batches(data: np.ndarray, batch_size: int, block_size: int, seed: int):
    """Infinite stream of (x, y) uint8 batches. Seeded, so the order is fixed."""
    rng = np.random.default_rng(seed)
    high = len(data) - block_size - 1
    while True:
        starts = rng.integers(0, high, size=batch_size, dtype=np.int64)
        idx = starts[:, None] + np.arange(block_size + 1, dtype=np.int64)[None, :]
        win = data[idx]
        yield win[:, :-1], win[:, 1:]


def cross_entropy_bpb(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy in **bits per byte**, computed in float64.

    Deliberately not reusing the model's own softmax: the metric must not move
    when an agent edits a kernel. This is the measurement instrument, and it is
    stable, wide, and slow on purpose.
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1, logits.shape[-1])
    t = np.asarray(targets).reshape(-1).astype(np.int64)
    z = z - z.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=-1))
    nll = lse - z[np.arange(z.shape[0]), t]
    return float(nll.mean() / LN2)


def evaluate(forward, val: np.ndarray) -> float:
    """val_bpb over the fixed windows. ``forward(x_uint8) -> logits``."""
    starts = eval_windows(val)
    idx = starts[:, None] + np.arange(BLOCK_SIZE + 1, dtype=np.int64)[None, :]
    win = val[idx]
    x, y = win[:, :-1], win[:, 1:]
    total, n = 0.0, 0
    step = 8
    for i in range(0, len(x), step):
        logits = forward(x[i : i + step])
        chunk = x[i : i + step].size
        total += cross_entropy_bpb(logits, y[i : i + step]) * chunk
        n += chunk
    return total / n


class Budget:
    """Fixed wall-clock training budget, excluding everything before ``start``."""

    def __init__(self, seconds: float = TRAIN_SECONDS) -> None:
        self.seconds = seconds
        self.t0 = 0.0

    def start(self) -> None:
        self.t0 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def expired(self) -> bool:
        return self.elapsed >= self.seconds
