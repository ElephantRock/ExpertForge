"""Frozen D0 learning-rate schedule (baseline contract, amendment chain v3).

The schedule is a pure function of the one-based applied-update index ``u``
(``semantic_update_indexing: optimizer_update_index_u_is_one_based_for_applied_updates``):

- ``lr(0) = 0`` (the untrained control never applies an update);
- ``lr(u) = peak * u / warmup`` for ``1 <= u <= warmup`` (linear warmup);
- ``lr(u) = final + 0.5 * (peak - final) * (1 + cos(pi * (u - warmup) / (updates - warmup)))``
  for ``warmup < u <= updates`` (cosine decay).

Endpoint exactness is part of the ratified contract ("warmup reaches peak LR
exactly at the declared warmup update, and cosine decay reaches final LR
exactly at the last update"): ``u == warmup`` returns ``peak`` directly (the
algebraically identical ``peak * warmup / warmup`` is not guaranteed to be
bit-exact under IEEE-754 for arbitrary ``peak``), and the cosine branch
returns ``final`` exactly at ``u == updates`` because ``cos(pi) == -1.0``
exactly in IEEE-754.

The module is intentionally torch-free so the schedule contract is verifiable
on the fast test tier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from expertforge.config.d0_models import D0ScheduleConfig
from expertforge.d0.errors import ScheduleError

__all__ = [
    "SCHEDULER_TYPE",
    "D0Schedule",
    "ScheduleScalarState",
]

# Canonical scheduler identity recorded in checkpoint descriptors.
SCHEDULER_TYPE = "d0_warmup_cosine_v1"


@dataclass(frozen=True, slots=True)
class ScheduleScalarState:
    """Truthful scalar checkpoint state of the schedule.

    ``last_applied_update`` is the index of the most recently applied update
    (0 before any update). On restore the float64 ``last_learning_rate``
    scalar is authoritative so a restored schedule matches an uninterrupted
    one exactly.
    """

    last_applied_update: int
    last_learning_rate: float


@dataclass(frozen=True, slots=True)
class D0Schedule:
    """The frozen warmup+cosine schedule over one-based update indices."""

    peak_learning_rate: float
    final_learning_rate: float
    warmup_updates: int
    optimizer_updates: int

    @classmethod
    def from_config(cls, config: D0ScheduleConfig) -> D0Schedule:
        """Build the schedule from the resolved typed config section."""
        return cls(
            peak_learning_rate=float(config.peak_learning_rate),
            final_learning_rate=float(config.final_learning_rate),
            warmup_updates=int(config.warmup_updates),
            optimizer_updates=int(config.optimizer_updates),
        )

    def learning_rate_at(self, u: int) -> float:
        """Return the exact frozen learning rate for applied-update index ``u``.

        ``u == 0`` is the untrained control (learning rate 0, never applied).
        Indices outside ``0..optimizer_updates`` fail closed.
        """

        if type(u) is not int:
            raise ScheduleError(f"update index must be an exact integer; got {u!r}")
        if u == 0:
            return 0.0
        if u < 0 or u > self.optimizer_updates:
            raise ScheduleError(
                f"update index {u} outside the frozen range 0..{self.optimizer_updates}"
            )
        if u <= self.warmup_updates:
            if u == self.warmup_updates:
                return self.peak_learning_rate
            return self.peak_learning_rate * u / self.warmup_updates
        progress = (u - self.warmup_updates) / (self.optimizer_updates - self.warmup_updates)
        return self.final_learning_rate + 0.5 * (
            self.peak_learning_rate - self.final_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))

    def scalar_state(self, *, last_applied_update: int) -> ScheduleScalarState:
        """Capture the truthful scalar state after ``last_applied_update``."""

        if type(last_applied_update) is not int:
            raise ScheduleError(
                f"last_applied_update must be an exact integer; got {last_applied_update!r}"
            )
        if last_applied_update < 0 or last_applied_update > self.optimizer_updates:
            raise ScheduleError(
                f"last_applied_update {last_applied_update} outside 0..{self.optimizer_updates}"
            )
        return ScheduleScalarState(
            last_applied_update=last_applied_update,
            last_learning_rate=self.learning_rate_at(last_applied_update),
        )
