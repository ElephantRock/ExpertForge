"""Tests for the frozen D0 accept/fail/kill threshold matrix."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter

from expertforge.config.d0_models import D0ThresholdConfig
from expertforge.d0.training.thresholds import (
    RunThresholdSnapshot,
    ThresholdDecision,
    evaluate_thresholds,
)

# The frozen qualification and canonical threshold blocks (v3 configs).
QUALIFICATION: dict[str, Any] = {
    "minimum_final_validation_loss_improvement_nats": 0.5,
    "maximum_final_loss_above_best_prior_nats": None,
    "maximum_consecutive_regressing_validation_boundaries": None,
    "regression_boundary_delta_nats": None,
    "minimum_final_to_initial_throughput_ratio": None,
    "maximum_skipped_updates": 0,
    "maximum_peak_device_memory_fraction": 0.95,
    "requires_exact_checkpoint_round_trip": True,
    "requires_locked_environment_resume_equality": True,
    "maximum_checkpoint_write_seconds": 300,
    "maximum_checkpoint_read_seconds": 300,
    "maximum_peak_host_memory_fraction": 0.9,
    "minimum_loss_improvement_at_quarter_budget_nats": 0.1,
    "maximum_rejected_recovery_attempts_before_kill": 1,
    "kill_on_any_non_finite_value": True,
    "kill_on_any_skipped_optimizer_update": True,
    "kill_on_any_checkpoint_or_resume_state_mismatch": True,
    "maximum_out_of_memory_failures_after_remediation": 1,
    "maximum_failed_recovery_attempts": 2,
    "minimum_loss_improvement_at_half_budget_nats": 0.1,
}

CANONICAL: dict[str, Any] = {
    **QUALIFICATION,
    "minimum_final_validation_loss_improvement_nats": 1.0,
    "maximum_final_loss_above_best_prior_nats": 0.05,
    "maximum_consecutive_regressing_validation_boundaries": 3,
    "regression_boundary_delta_nats": 0.1,
    "minimum_final_to_initial_throughput_ratio": 0.8,
    "requires_exact_checkpoint_round_trip": None,
    "requires_locked_environment_resume_equality": None,
}

_adapter = TypeAdapter(D0ThresholdConfig)


def _qualification_config() -> D0ThresholdConfig:
    return _adapter.validate_python(QUALIFICATION)


def _canonical_config() -> D0ThresholdConfig:
    return _adapter.validate_python(CANONICAL)


def _healthy_snapshot(**overrides: Any) -> RunThresholdSnapshot:
    base: dict[str, Any] = {
        "update": 100,
        "optimizer_updates": 4000,
        "initial_validation_loss": 10.0,
        "current_validation_loss": 9.0,
    }
    base.update(overrides)
    return RunThresholdSnapshot(**base)


class TestDecisionRecord:
    def test_fail_requires_diagnostic_code(self) -> None:
        with pytest.raises(ValueError, match="require a diagnostic code"):
            ThresholdDecision(decision="fail_run", reason="x")

    def test_kill_requires_diagnostic_code(self) -> None:
        with pytest.raises(ValueError, match="require a diagnostic code"):
            ThresholdDecision(decision="kill", reason="x")

    def test_continue_forbids_diagnostic_code(self) -> None:
        with pytest.raises(ValueError, match="must not carry"):
            ThresholdDecision(decision="continue", reason="x", diagnostic_code="out_of_memory")

    def test_reject_acceptance_forbids_diagnostic_code(self) -> None:
        with pytest.raises(ValueError, match="must not carry"):
            ThresholdDecision(
                decision="reject_acceptance", reason="x", diagnostic_code="optimizer_failure"
            )


class TestKillConditions:
    def test_non_finite_kills(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(), _healthy_snapshot(non_finite_value_observed=True)
        )
        assert decision.decision == "kill"
        assert decision.diagnostic_code == "non_finite_loss"

    def test_skipped_update_kills(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(), _healthy_snapshot(skipped_updates=1)
        )
        assert decision.decision == "kill"
        assert decision.diagnostic_code == "optimizer_failure"

    def test_state_mismatch_kills(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(),
            _healthy_snapshot(checkpoint_or_resume_state_mismatch_observed=True),
        )
        assert decision.decision == "kill"
        assert decision.diagnostic_code == "checkpoint_failure"

    def test_oom_over_budget_kills(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(), _healthy_snapshot(out_of_memory_failures_after_remediation=2)
        )
        assert decision.decision == "kill"
        assert decision.diagnostic_code == "out_of_memory"

    def test_oom_at_budget_does_not_kill(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(), _healthy_snapshot(out_of_memory_failures_after_remediation=1)
        )
        assert decision.decision == "continue"

    def test_too_many_failed_recoveries_kill(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(), _healthy_snapshot(failed_recovery_attempts=3)
        )
        assert decision.decision == "kill"

    def test_too_many_rejected_recoveries_kill(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(), _healthy_snapshot(rejected_recovery_attempts=2)
        )
        assert decision.decision == "kill"

    def test_kill_precedes_fail(self) -> None:
        """A snapshot triggering both kill and fail conditions reports kill."""

        decision = evaluate_thresholds(
            _qualification_config(),
            _healthy_snapshot(
                non_finite_value_observed=True,
                maximum_checkpoint_write_seconds_observed=9999.0,
            ),
        )
        assert decision.decision == "kill"


class TestFailConditions:
    def test_slow_checkpoint_write_fails(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(),
            _healthy_snapshot(maximum_checkpoint_write_seconds_observed=301.0),
        )
        assert decision.decision == "fail_run"
        assert decision.diagnostic_code == "checkpoint_failure"

    def test_slow_checkpoint_read_fails(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(),
            _healthy_snapshot(maximum_checkpoint_read_seconds_observed=301.0),
        )
        assert decision.decision == "fail_run"
        assert decision.diagnostic_code == "checkpoint_failure"

    def test_host_memory_over_budget_fails(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(), _healthy_snapshot(peak_host_memory_fraction=0.91)
        )
        assert decision.decision == "fail_run"
        assert decision.diagnostic_code == "out_of_memory"

    def test_device_memory_over_budget_fails(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(), _healthy_snapshot(peak_device_memory_fraction=0.96)
        )
        assert decision.decision == "fail_run"

    def test_quarter_budget_loss_floor_fails(self) -> None:
        # update 1000/4000 = 25% of budget; improvement 0.09 < 0.1 floor.
        decision = evaluate_thresholds(
            _qualification_config(),
            _healthy_snapshot(update=1000, current_validation_loss=10.0 - 0.09),
        )
        assert decision.decision == "fail_run"
        assert decision.diagnostic_code == "optimizer_failure"

    def test_quarter_budget_met_continues(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(),
            _healthy_snapshot(update=1000, current_validation_loss=10.0 - 0.2),
        )
        assert decision.decision == "continue"

    def test_half_budget_loss_floor_fails(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(),
            _healthy_snapshot(update=2000, current_validation_loss=10.0 - 0.05),
        )
        assert decision.decision == "fail_run"

    def test_before_quarter_budget_no_floor_applies(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(),
            _healthy_snapshot(update=100, current_validation_loss=10.0),
        )
        assert decision.decision == "continue"


class TestAcceptanceGate:
    def _final(self, **overrides: Any) -> RunThresholdSnapshot:
        base: dict[str, Any] = {
            "update": 4000,
            "is_final_boundary": True,
            "current_validation_loss": 9.0,
        }
        base.update(overrides)
        return _healthy_snapshot(**base)

    def test_qualification_improvement_met_accepts(self) -> None:
        # improvement 1.0 >= 0.5
        decision = evaluate_thresholds(_qualification_config(), self._final())
        assert decision.decision == "continue"
        assert decision.diagnostic_code is None

    def test_qualification_improvement_missed_rejects(self) -> None:
        decision = evaluate_thresholds(
            _qualification_config(), self._final(current_validation_loss=10.0 - 0.4)
        )
        assert decision.decision == "reject_acceptance"
        assert "below" in decision.reason

    def test_canonical_requires_one_nat(self) -> None:
        # Improvement 0.9 passes qualification's 0.5 but misses canonical's 1.0.
        snapshot = self._final(current_validation_loss=10.0 - 0.9)
        decision = evaluate_thresholds(_canonical_config(), snapshot)
        assert decision.decision == "reject_acceptance"

    def test_canonical_exactly_one_nat_meets_the_gate(self) -> None:
        # The floor is inclusive: exactly 1.0 nats meets the canonical gate.
        snapshot = self._final(current_validation_loss=10.0 - 1.0)
        decision = evaluate_thresholds(_canonical_config(), snapshot)
        assert decision.decision == "continue"

    def test_canonical_final_above_best_prior_rejects(self) -> None:
        snapshot = self._final(current_validation_loss=10.0 - 1.2, best_validation_loss=10.0 - 1.25)
        decision = evaluate_thresholds(_canonical_config(), snapshot)
        assert decision.decision == "reject_acceptance"
        assert "best prior" in decision.reason

    def test_canonical_consecutive_regressions_reject(self) -> None:
        snapshot = self._final(consecutive_regressing_boundaries=4)
        decision = evaluate_thresholds(_canonical_config(), snapshot)
        assert decision.decision == "reject_acceptance"

    def test_canonical_throughput_ratio_reject(self) -> None:
        snapshot = self._final(
            first_window_median_throughput=100000.0, current_median_throughput=70000.0
        )
        decision = evaluate_thresholds(_canonical_config(), snapshot)
        assert decision.decision == "reject_acceptance"
        assert decision.diagnostic_code is None

    def test_canonical_all_gates_met_accepts(self) -> None:
        snapshot = self._final(
            current_validation_loss=10.0 - 1.2,
            best_validation_loss=10.0 - 1.15,
            consecutive_regressing_boundaries=0,
            first_window_median_throughput=100000.0,
            current_median_throughput=90000.0,
        )
        decision = evaluate_thresholds(_canonical_config(), snapshot)
        assert decision.decision == "continue"

    def test_qualification_ignores_canonical_gates(self) -> None:
        """Qualification pins the canonical gates to null: regressing
        boundaries and throughput loss do not reject."""

        snapshot = self._final(
            consecutive_regressing_boundaries=99,
            first_window_median_throughput=100000.0,
            current_median_throughput=1.0,
        )
        decision = evaluate_thresholds(_qualification_config(), snapshot)
        assert decision.decision == "continue"
