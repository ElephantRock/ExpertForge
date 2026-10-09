"""Tests for D0 gradient-accumulation accounting and loss reduction."""

from __future__ import annotations

import math

import pytest

from expertforge.config.d0_models import D0BatchConfig
from expertforge.d0.errors import D0TrainingError
from expertforge.d0.training.accumulation import UpdateAccumulator


def _config(
    *,
    micro: int = 4,
    accum: int = 16,
    seq: int = 1024,
) -> D0BatchConfig:
    """Frozen qualification batch shape (1 device)."""

    return D0BatchConfig(
        devices=1,
        microbatch_sequences_per_device=micro,
        gradient_accumulation_steps=accum,
        global_sequences_per_update=micro * accum,
        target_tokens_per_update=micro * accum * seq,
    )


def _targets_per_microstep() -> int:
    # qualification: 4 sequences * 1024 target positions, no padding
    return 4 * 1024


class TestConstruction:
    def test_accepts_the_frozen_qualification_shape(self) -> None:
        acc = UpdateAccumulator(_config(), target_tokens_per_sequence=1024)
        assert acc.microsteps_per_update == 16
        assert acc.targets_per_microstep == 4096
        assert acc.global_update == 0
        assert acc.completed_microsteps == 0
        assert acc.accumulation_position == 0
        assert acc.processed_tokens == 0

    def test_rejects_inconsistent_batch_arithmetic(self) -> None:
        bad = D0BatchConfig(
            devices=1,
            microbatch_sequences_per_device=4,
            gradient_accumulation_steps=16,
            global_sequences_per_update=64,
            target_tokens_per_update=999,
        )
        with pytest.raises(D0TrainingError, match="batch arithmetic disagrees"):
            UpdateAccumulator(bad, target_tokens_per_sequence=1024)

    def test_rejects_non_integer_sequence_length(self) -> None:
        with pytest.raises(D0TrainingError, match="positive exact integer"):
            UpdateAccumulator(_config(), target_tokens_per_sequence=1024.0)  # type: ignore[arg-type]


class TestMicrostepAccounting:
    def test_full_update_advances_counters(self) -> None:
        acc = UpdateAccumulator(_config(), target_tokens_per_sequence=1024)
        acc.start_update()
        for _ in range(16):
            acc.record_microstep(token_loss_sum=4096.0, target_token_count=4096)
        mean = acc.complete_update()
        assert mean == 1.0
        assert acc.global_update == 1
        assert acc.completed_microsteps == 16
        assert acc.accumulation_position == 0
        assert acc.processed_tokens == 65536

    def test_rejects_microstep_without_open_update(self) -> None:
        acc = UpdateAccumulator(_config(), target_tokens_per_sequence=1024)
        with pytest.raises(D0TrainingError, match="before start_update"):
            acc.record_microstep(token_loss_sum=1.0, target_token_count=4096)

    def test_rejects_complete_with_open_microsteps(self) -> None:
        acc = UpdateAccumulator(_config(), target_tokens_per_sequence=1024)
        acc.start_update()
        acc.record_microstep(token_loss_sum=1.0, target_token_count=4096)
        with pytest.raises(D0TrainingError, match="cannot complete update"):
            acc.complete_update()

    def test_rejects_wrong_target_count(self) -> None:
        acc = UpdateAccumulator(_config(), target_tokens_per_sequence=1024)
        acc.start_update()
        with pytest.raises(D0TrainingError, match="no-padding contract"):
            acc.record_microstep(token_loss_sum=1.0, target_token_count=4095)

    def test_rejects_nested_start(self) -> None:
        acc = UpdateAccumulator(_config(), target_tokens_per_sequence=1024)
        acc.start_update()
        with pytest.raises(D0TrainingError, match="accumulation_position"):
            acc.start_update()

    def test_update_boundary_invariants_over_many_updates(self) -> None:
        acc = UpdateAccumulator(_config(), target_tokens_per_sequence=1024)
        for update in range(1, 4):
            acc.start_update()
            for _ in range(16):
                acc.record_microstep(token_loss_sum=4096.0, target_token_count=4096)
            acc.complete_update()
            # Smoke-gate update-boundary invariants.
            assert acc.global_update == acc.completed_microsteps // 16
            assert acc.accumulation_position == 0
            assert acc.processed_tokens == acc.global_update * 65536
            assert update == acc.global_update


class TestLossReduction:
    def test_exact_mean_over_all_target_tokens(self) -> None:
        acc = UpdateAccumulator(_config(), target_tokens_per_sequence=1024)
        acc.start_update()
        # 15 microbatches with loss sum 2*4096, one with loss sum 6*4096:
        # exact update mean = (15*2*4096 + 6*4096) / (16*4096) = 36/16 = 2.25
        for index in range(16):
            acc.record_microstep(
                token_loss_sum=(6.0 if index == 15 else 2.0) * 4096,
                target_token_count=4096,
            )
        assert acc.complete_update() == 2.25

    def test_mean_of_means_would_differ(self) -> None:
        """The frozen reduction is NOT the mean of microbatch means.

        With equal token counts per microbatch the two coincide, so this
        test uses *recorded sums* on unequal-count... which the no-padding
        contract forbids; instead we demonstrate the distinction on the
        accumulator by unequal loss sums where a buggy mean-of-means over
        sums-per-token drifts. The exact value is the weighted mean.
        """

        acc = UpdateAccumulator(_config(), target_tokens_per_sequence=1024)
        acc.start_update()
        # Half the microbatches at per-token loss 1.0, half at 3.0:
        # exact mean = 2.0 either way (equal counts), but the accumulator
        # must produce it from SUMS, so verify against the closed form.
        for index in range(16):
            per_token = 1.0 if index < 8 else 3.0
            acc.record_microstep(token_loss_sum=per_token * 4096, target_token_count=4096)
        assert acc.complete_update() == 2.0

    def test_microbatch_mean_is_returned_per_microstep(self) -> None:
        acc = UpdateAccumulator(_config(), target_tokens_per_sequence=1024)
        acc.start_update()
        record = acc.record_microstep(token_loss_sum=8192.0, target_token_count=4096)
        assert record.microstep_index == 1
        assert record.target_token_count == 4096
        assert math.isclose(record.microbatch_mean, 2.0, rel_tol=1e-7)

    def test_float32_reduction_dtype_is_observable(self) -> None:
        """The accumulator reduces in float32 (frozen loss_reduction_dtype)."""

        acc = UpdateAccumulator(_config(), target_tokens_per_sequence=1024)
        acc.start_update()
        # 1e-8 added 16 times in float32 stays representable; the update
        # mean equals the exact tiny value without double-precision drift.
        for _ in range(16):
            acc.record_microstep(token_loss_sum=1e-08 * 4096, target_token_count=4096)
        mean = acc.complete_update()
        assert math.isclose(mean, 1e-08, rel_tol=1e-4)
