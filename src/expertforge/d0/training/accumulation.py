"""Gradient-accumulation accounting and frozen loss reduction (D0 batch).

Implements the ratified batch semantics over microsteps:

- one optimizer update = ``gradient_accumulation_steps`` microsteps;
- ``accumulation_position`` counts microsteps within the current update and
  must be 0 at every checkpoint save/restore boundary (V1 save boundary,
  mirrored from the smoke gate);
- the update-level loss is the exact
  ``mean_over_all_target_tokens_in_optimizer_update``: the sum of per-token
  losses across all microbatches divided by the total target-token count —
  never a mean of per-microbatch means (those differ whenever microbatches
  have different token counts).

Per-microbatch loss sums are accumulated in float32 (the frozen
``loss_reduction_dtype``) via NumPy so the reduction dtype is observable and
testable without torch. The module is torch-free.

Target-token accounting is exact: with no padding (frozen contract), every
microstep consumes exactly ``microbatch_sequences_per_device *
target_tokens_per_sequence`` target positions, and the per-update total must
equal ``target_tokens_per_update`` from the typed batch config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from expertforge.config.d0_models import D0BatchConfig
from expertforge.d0.errors import D0TrainingError

__all__ = ["UpdateAccumulator"]

_FLOAT32 = np.float32


@dataclass(frozen=True, slots=True)
class MicrostepRecord:
    """One microstep contribution as seen by the accumulator."""

    microstep_index: int
    token_loss_sum: float
    target_token_count: int
    microbatch_mean: float


class UpdateAccumulator:
    """Microstep accounting and exact update-level loss reduction.

    The accumulator is deliberately torch-free: callers hand in per-microstep
    float32 loss sums and target-token counts and receive the exact frozen
    update-level mean. Counter semantics mirror the smoke gate's
    update-boundary invariants (``global_update == completed_microsteps /
    microsteps_per_update``, ``accumulation_position == 0`` at boundaries).
    """

    def __init__(
        self,
        config: D0BatchConfig,
        *,
        target_tokens_per_sequence: int,
    ) -> None:
        if type(target_tokens_per_sequence) is not int or target_tokens_per_sequence <= 0:
            raise D0TrainingError(
                "target_tokens_per_sequence must be a positive exact integer; "
                f"got {target_tokens_per_sequence!r}"
            )
        self._microsteps_per_update = int(config.gradient_accumulation_steps)
        self._targets_per_microstep = (
            int(config.microbatch_sequences_per_device) * target_tokens_per_sequence
        )
        expected_per_update = self._targets_per_microstep * self._microsteps_per_update
        if expected_per_update != int(config.target_tokens_per_update):
            raise D0TrainingError(
                "batch arithmetic disagrees with the typed config: "
                f"microbatch_sequences_per_device * target_tokens_per_sequence * "
                f"gradient_accumulation_steps = {expected_per_update}, but "
                f"target_tokens_per_update = {config.target_tokens_per_update}"
            )
        self._global_update = 0
        self._completed_microsteps = 0
        self._accumulation_position = 0
        self._update_open = False
        self._loss_sum = _FLOAT32(0.0)
        self._token_count = 0

    # -- counters ---------------------------------------------------------

    @property
    def microsteps_per_update(self) -> int:
        return self._microsteps_per_update

    @property
    def targets_per_microstep(self) -> int:
        return self._targets_per_microstep

    @property
    def global_update(self) -> int:
        return self._global_update

    @property
    def completed_microsteps(self) -> int:
        return self._completed_microsteps

    @property
    def accumulation_position(self) -> int:
        return self._accumulation_position

    @property
    def processed_tokens(self) -> int:
        return self._global_update * self._targets_per_microstep * self._microsteps_per_update

    # -- lifecycle --------------------------------------------------------

    def start_update(self) -> None:
        """Open one optimizer update; fails if an update is already open."""

        if self._update_open:
            raise D0TrainingError(
                "cannot start an update while an update is open "
                f"(accumulation_position = {self._accumulation_position})"
            )
        self._update_open = True
        self._loss_sum = _FLOAT32(0.0)
        self._token_count = 0

    def record_microstep(
        self, *, token_loss_sum: float, target_token_count: int
    ) -> MicrostepRecord:
        """Record one microstep; returns the record with its microbatch mean."""

        if not self._update_open:
            raise D0TrainingError("cannot record a microstep before start_update opens the update")
        if self._accumulation_position >= self._microsteps_per_update:
            raise D0TrainingError(
                "microstep overflow: update already holds "
                f"{self._accumulation_position} of {self._microsteps_per_update} "
                "microsteps"
            )
        if target_token_count != self._targets_per_microstep:
            raise D0TrainingError(
                f"microstep target-token count {target_token_count} != frozen "
                f"{self._targets_per_microstep} (no-padding contract)"
            )
        loss_sum = _FLOAT32(token_loss_sum)
        microbatch_mean = _FLOAT32(loss_sum / _FLOAT32(target_token_count))
        # Accumulate in the declared float32 loss-reduction dtype.
        self._loss_sum = _FLOAT32(self._loss_sum + loss_sum)
        self._token_count += target_token_count
        self._accumulation_position += 1
        self._completed_microsteps += 1
        return MicrostepRecord(
            microstep_index=self._accumulation_position,
            token_loss_sum=float(loss_sum),
            target_token_count=target_token_count,
            microbatch_mean=float(microbatch_mean),
        )

    def complete_update(self) -> float:
        """Close the update and return the exact update-level mean loss.

        The mean divides the float32-accumulated token-loss sum by the total
        target-token count of the update and rounds once to float32.
        """

        if self._accumulation_position != self._microsteps_per_update:
            raise D0TrainingError(
                f"cannot complete update with accumulation_position = "
                f"{self._accumulation_position}; expected "
                f"{self._microsteps_per_update} microsteps"
            )
        if self._token_count == 0:
            raise D0TrainingError("cannot complete an update with zero target tokens")
        mean = _FLOAT32(self._loss_sum / _FLOAT32(self._token_count))
        self._global_update += 1
        self._accumulation_position = 0
        self._update_open = False
        self._loss_sum = _FLOAT32(0.0)
        self._token_count = 0
        return float(mean)

    def advance_to(self, global_update: int) -> None:
        """Fast-forward the counters to a resumed position (restore path).

        Only valid on a fresh accumulator at an update boundary: sets the
        counter state as if ``global_update`` updates had completed. This is
        the resume counterpart of the update-boundary invariants, never a
        mid-update skip.
        """

        if self._update_open or self._accumulation_position != 0:
            raise D0TrainingError("advance_to requires a closed update boundary")
        if self._global_update != 0:
            raise D0TrainingError(
                f"advance_to requires a fresh accumulator; global_update = {self._global_update}"
            )
        if type(global_update) is not int or global_update < 0:
            raise D0TrainingError(
                f"global_update must be a non-negative exact integer; got {global_update!r}"
            )
        self._global_update = global_update
        self._completed_microsteps = global_update * self._microsteps_per_update
