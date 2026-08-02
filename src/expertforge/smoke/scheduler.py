"""Constant-with-step scheduler for the smoke gate (Issue #14).

Carries both a tiny tensor state (the learning-rate scalar as a one-element
float32 array) and a structured scalar state (``step_count``) so the checkpoint
exercises BOTH the scheduler tensor and ``scheduler_scalar_state`` paths.
The learning rate is constant (the smoke gate does not need a warmup/decay
schedule); the scheduler exists to be captured and restored truthfully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["SmokeScheduler", "SCHEDULER_TYPE"]

SCHEDULER_TYPE = "constant_with_step"


@dataclass
class _SchedulerSnapshot:
    state_tensor: np.ndarray  # [1] float32 — the current learning rate
    step_count: int


class SmokeScheduler:
    """A constant LR scheduler that advances a step counter each :meth:`step`."""

    def __init__(self, lr: float) -> None:
        self.lr = float(lr)
        self.step_count = 0

    def initialize(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1

    def current_lr(self) -> float:
        return self.lr

    # -- checkpoint state -------------------------------------------------

    def state_tensor(self) -> np.ndarray:
        """One-element float32 array carrying the current learning rate."""
        return np.array([self.lr], dtype=np.float32)

    def scalar_state(self) -> dict[str, Any]:
        return {"step_count": int(self.step_count), "lr": float(self.lr)}

    def apply_state(self, state_tensor: np.ndarray, scalar_state: dict[str, Any]) -> None:
        lr_from_tensor = float(np.asarray(state_tensor, dtype=np.float32).reshape(-1)[0])
        self.lr = lr_from_tensor
        self.step_count = int(scalar_state.get("step_count", 0))
