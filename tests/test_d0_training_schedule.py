"""Golden-vector tests for the frozen D0 learning-rate schedule.

All expected values below are literals computed once by an independent
one-off script (stdlib ``math`` with the frozen formulas) and reviewed before
being pasted. Qualification profile: peak 0.0006, warmup 200, final 6e-05,
updates 4000. Canonical profile: peak 0.0004, warmup 1600, final 4e-05,
updates 32000.
"""

from __future__ import annotations

import pytest

from expertforge.config.d0_models import D0ScheduleConfig
from expertforge.d0.errors import ScheduleError
from expertforge.d0.training.schedule import SCHEDULER_TYPE, D0Schedule, ScheduleScalarState


def _schedule_config(
    *,
    peak: float,
    warmup: int,
    final: float,
    updates: int,
    tokens: int,
    interval: int,
) -> D0ScheduleConfig:
    """Build one profile's schedule config with the frozen literal formulas.

    The Literal-typed formula fields are passed inline so static typing sees
    the exact frozen strings.
    """

    return D0ScheduleConfig(
        semantic_update_indexing=(
            "optimizer_update_index_u_is_one_based_for_applied_updates; "
            "u_in_1_through_optimizer_updates"
        ),
        learning_rate_at_update_zero=0.0,
        peak_learning_rate=peak,
        warmup_updates=warmup,
        warmup_formula="lr(u)=peak_learning_rate*u/warmup_updates for 1<=u<=warmup_updates",
        final_learning_rate=final,
        decay="cosine",
        cosine_formula=(
            "lr(u)=final_learning_rate+0.5*(peak_learning_rate-final_learning_rate)*"
            "(1+cos(pi*(u-warmup_updates)/(optimizer_updates-warmup_updates))) "
            "for warmup_updates<u<=optimizer_updates"
        ),
        training_target_tokens=tokens,
        optimizer_updates=updates,
        validation_interval_updates=interval,
        checkpoint_interval_updates=interval,
        generation_interval_updates=interval,
        profiling_interval_updates=interval,
    )


def _qualification_config() -> D0ScheduleConfig:
    return _schedule_config(
        peak=0.0006,
        warmup=200,
        final=6.0e-05,
        updates=4000,
        tokens=262144000,
        interval=500,
    )


def _canonical_config() -> D0ScheduleConfig:
    return _schedule_config(
        peak=0.0004,
        warmup=1600,
        final=4.0e-05,
        updates=32000,
        tokens=2097152000,
        interval=2000,
    )


class TestQualificationGoldenVectors:
    def test_from_config(self) -> None:
        schedule = D0Schedule.from_config(_qualification_config())
        assert schedule.peak_learning_rate == 0.0006
        assert schedule.final_learning_rate == 6.0e-05
        assert schedule.warmup_updates == 200
        assert schedule.optimizer_updates == 4000

    def test_update_zero_is_the_untrained_control(self) -> None:
        assert D0Schedule.from_config(_qualification_config()).learning_rate_at(0) == 0.0

    def test_warmup_start_literal(self) -> None:
        # 0.0006 * 1 / 200 = 2.9999999999999997e-06 (IEEE-754)
        assert D0Schedule.from_config(_qualification_config()).learning_rate_at(1) == (
            2.9999999999999997e-06
        )

    def test_warmup_middle_literal(self) -> None:
        assert D0Schedule.from_config(_qualification_config()).learning_rate_at(100) == 0.0003

    def test_warmup_last_before_peak_literal(self) -> None:
        assert D0Schedule.from_config(_qualification_config()).learning_rate_at(199) == 0.000597

    def test_warmup_peak_endpoint_is_exactly_peak(self) -> None:
        schedule = D0Schedule.from_config(_qualification_config())
        # Contract: warmup reaches peak LR exactly at the declared warmup update.
        assert schedule.learning_rate_at(200) == 0.0006

    def test_first_decay_step_literal(self) -> None:
        # final + 0.5*(peak-final)*(1+cos(pi*1/3800))
        assert D0Schedule.from_config(_qualification_config()).learning_rate_at(201) == (
            0.0005999999077287729
        )

    def test_decay_midpoint_literal(self) -> None:
        # u=2100 -> progress 0.5; cos(pi/2) = 6.123233995736766e-17 in IEEE-754
        assert D0Schedule.from_config(_qualification_config()).learning_rate_at(2100) == (
            0.00032999999999999994
        )

    def test_final_endpoint_is_exactly_final(self) -> None:
        schedule = D0Schedule.from_config(_qualification_config())
        # Contract: cosine decay reaches final LR exactly at the last update
        # (cos(pi) == -1.0 exactly in IEEE-754).
        assert schedule.learning_rate_at(4000) == 6.0e-05


class TestCanonicalGoldenVectors:
    def test_warmup_start_literal(self) -> None:
        # 0.0004 / 1600 = 2.5e-07
        assert D0Schedule.from_config(_canonical_config()).learning_rate_at(1) == 2.5e-07

    def test_warmup_peak_endpoint_is_exactly_peak(self) -> None:
        assert D0Schedule.from_config(_canonical_config()).learning_rate_at(1600) == 0.0004

    def test_decay_midpoint_literal(self) -> None:
        # u=16800 -> progress (16800-1600)/30400 = 0.5
        assert D0Schedule.from_config(_canonical_config()).learning_rate_at(16800) == 0.00022

    def test_final_endpoint_is_exactly_final(self) -> None:
        assert D0Schedule.from_config(_canonical_config()).learning_rate_at(32000) == 4.0e-05


class TestScheduleContract:
    def test_rejects_float_index(self) -> None:
        with pytest.raises(ScheduleError, match="exact integer"):
            D0Schedule.from_config(_qualification_config()).learning_rate_at(1.0)  # type: ignore[arg-type]

    def test_rejects_negative_index(self) -> None:
        with pytest.raises(ScheduleError, match="outside the frozen range"):
            D0Schedule.from_config(_qualification_config()).learning_rate_at(-1)

    def test_rejects_index_past_the_budget(self) -> None:
        with pytest.raises(ScheduleError, match="outside the frozen range"):
            D0Schedule.from_config(_qualification_config()).learning_rate_at(4001)

    def test_warmup_is_monotonically_increasing(self) -> None:
        schedule = D0Schedule.from_config(_qualification_config())
        rates = [schedule.learning_rate_at(u) for u in range(1, 201)]
        assert all(b > a for a, b in zip(rates, rates[1:], strict=False))

    def test_decay_is_monotonically_decreasing_to_final(self) -> None:
        schedule = D0Schedule.from_config(_qualification_config())
        rates = [schedule.learning_rate_at(u) for u in range(201, 4001)]
        # Non-increasing (float64 cos may plateau at ULP granularity), and
        # strictly decreasing overall.
        assert all(b <= a for a, b in zip(rates, rates[1:], strict=False))
        assert rates[0] > rates[-1]
        assert rates[-1] == schedule.final_learning_rate


class TestScalarState:
    def test_scalar_state_before_any_update(self) -> None:
        schedule = D0Schedule.from_config(_qualification_config())
        state = schedule.scalar_state(last_applied_update=0)
        assert isinstance(state, ScheduleScalarState)
        assert state.last_applied_update == 0
        assert state.last_learning_rate == 0.0

    def test_scalar_state_at_an_update(self) -> None:
        schedule = D0Schedule.from_config(_qualification_config())
        state = schedule.scalar_state(last_applied_update=2100)
        assert state.last_applied_update == 2100
        assert state.last_learning_rate == 0.00032999999999999994

    def test_scalar_state_rejects_out_of_range(self) -> None:
        with pytest.raises(ScheduleError, match="outside 0.."):
            D0Schedule.from_config(_qualification_config()).scalar_state(last_applied_update=4001)

    def test_scheduler_type_is_canonical(self) -> None:
        assert SCHEDULER_TYPE == "d0_warmup_cosine_v1"
