"""Candidate ReLU (skeleton — identical to reference until search runs)."""

from __future__ import annotations

import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def get_inputs(device: str = "cpu") -> torch.Tensor:
    return torch.randn(1024, device=device, dtype=torch.float32)
