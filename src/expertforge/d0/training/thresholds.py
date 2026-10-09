"""Frozen D0 accept / fail / kill threshold matrix (pure, torch-free).

Encodes the ratified threshold semantics from the D0 baseline contract:

- **kill** — the run must terminate immediately (program-level): any
  non-finite value, any skipped optimizer update, any checkpoint/resume
  state mismatch, more than the allowed OOM failures after one documented
  batch-preserving remediation, or the allowed number of failed recovery
  attempts being exceeded. Kill outcomes close telemetry with a failure
  diagnostic code.
- **fail_run** — the run failed but the program records the outcome and
  stops gracefully: checkpoint write/read over budget, peak host memory
  over budget, or the mid-budget loss-improvement floors missed.
- **reject_acceptance** — the run completed normally but the final state
  does not meet the acceptance gate (final improvement, canonical-only
  extra gates). This is a gate verdict, not a telemetry failure: the run
  itself finished, so no diagnostic code applies.

Every decision carries its reason. Diagnostic codes are drawn from the
closed ``expertforge.telemetry`` ``DiagnosticCode`` domain; acceptance
rejections and "continue" carry no code.

The evaluator consumes the typed ``D0ThresholdConfig`` — values are never
re-hardcoded here. Canonical-only gates activate exactly when their config
fields are non-null (the qualification config pins them to null).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from expertforge.config.d0_models import D0ThresholdConfig

__all__ = [
    "ThresholdDecision",
    "RunThresholdSnapshot",
    "evaluate_thresholds",
]

ThresholdDecisionKind = Literal["continue", "fail_run", "kill", "reject_acceptance"]


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    """One threshold evaluation outcome."""

    decision: ThresholdDecisionKind
    reason: str
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        if self.decision in ("fail_run", "kill") and self.diagnostic_code is None:
            raise ValueError(f"{self.decision} decisions require a diagnostic code")
        if self.decision in ("continue", "reject_acceptance") and self.diagnostic_code is not None:
            raise ValueError(f"{self.decision} decisions must not carry a diagnostic code")


@dataclass(frozen=True, slots=True)
class RunThresholdSnapshot:
    """Observed run state at one evaluation point (boundary or final).

    ``None`` numeric fields mean "not yet observed" (e.g. validation has not
    run yet); gate logic treats them as not triggering rather than failing.
    """

    update: int
    optimizer_updates: int
    initial_validation_loss: float | None = None
    current_validation_loss: float | None = None
    best_validation_loss: float | None = None
    consecutive_regressing_boundaries: int = 0
    first_window_median_throughput: float | None = None
    current_median_throughput: float | None = None
    skipped_updates: int = 0
    rejected_recovery_attempts: int = 0
    failed_recovery_attempts: int = 0
    maximum_checkpoint_write_seconds_observed: float = 0.0
    maximum_checkpoint_read_seconds_observed: float = 0.0
    peak_device_memory_fraction: float | None = None
    peak_host_memory_fraction: float | None = None
    out_of_memory_failures_after_remediation: int = 0
    non_finite_value_observed: bool = False
    checkpoint_or_resume_state_mismatch_observed: bool = False
    is_final_boundary: bool = False


def _kill_checks(config: D0ThresholdConfig, s: RunThresholdSnapshot) -> ThresholdDecision | None:
    if config.kill_on_any_non_finite_value and s.non_finite_value_observed:
        return ThresholdDecision(
            decision="kill",
            reason="a non-finite value was observed (kill_on_any_non_finite_value)",
            diagnostic_code="non_finite_loss",
        )
    if config.kill_on_any_skipped_optimizer_update and s.skipped_updates > 0:
        return ThresholdDecision(
            decision="kill",
            reason=f"{s.skipped_updates} optimizer update(s) were skipped",
            diagnostic_code="optimizer_failure",
        )
    if (
        config.kill_on_any_checkpoint_or_resume_state_mismatch
        and s.checkpoint_or_resume_state_mismatch_observed
    ):
        return ThresholdDecision(
            decision="kill",
            reason="checkpoint or resume state mismatch observed",
            diagnostic_code="checkpoint_failure",
        )
    if (
        s.out_of_memory_failures_after_remediation
        > config.maximum_out_of_memory_failures_after_remediation
    ):
        return ThresholdDecision(
            decision="kill",
            reason=(
                f"{s.out_of_memory_failures_after_remediation} OOM failures after "
                f"remediation exceed the allowed "
                f"{config.maximum_out_of_memory_failures_after_remediation}"
            ),
            diagnostic_code="out_of_memory",
        )
    if s.failed_recovery_attempts > config.maximum_failed_recovery_attempts:
        return ThresholdDecision(
            decision="kill",
            reason=(
                f"{s.failed_recovery_attempts} failed recovery attempts exceed the "
                f"allowed {config.maximum_failed_recovery_attempts}"
            ),
            diagnostic_code="checkpoint_failure",
        )
    if s.rejected_recovery_attempts > config.maximum_rejected_recovery_attempts_before_kill:
        return ThresholdDecision(
            decision="kill",
            reason=(
                f"{s.rejected_recovery_attempts} rejected recovery attempts exceed the "
                f"allowed {config.maximum_rejected_recovery_attempts_before_kill}"
            ),
            diagnostic_code="checkpoint_failure",
        )
    return None


def _fail_checks(config: D0ThresholdConfig, s: RunThresholdSnapshot) -> ThresholdDecision | None:
    if s.maximum_checkpoint_write_seconds_observed > config.maximum_checkpoint_write_seconds:
        return ThresholdDecision(
            decision="fail_run",
            reason=(
                f"checkpoint write took {s.maximum_checkpoint_write_seconds_observed}s, "
                f"over the {config.maximum_checkpoint_write_seconds}s budget"
            ),
            diagnostic_code="checkpoint_failure",
        )
    if s.maximum_checkpoint_read_seconds_observed > config.maximum_checkpoint_read_seconds:
        return ThresholdDecision(
            decision="fail_run",
            reason=(
                f"checkpoint read took {s.maximum_checkpoint_read_seconds_observed}s, "
                f"over the {config.maximum_checkpoint_read_seconds}s budget"
            ),
            diagnostic_code="checkpoint_failure",
        )
    if s.peak_device_memory_fraction is not None:
        if s.peak_device_memory_fraction > config.maximum_peak_device_memory_fraction:
            return ThresholdDecision(
                decision="fail_run",
                reason=(
                    f"peak device memory fraction {s.peak_device_memory_fraction} exceeds "
                    f"{config.maximum_peak_device_memory_fraction}"
                ),
                diagnostic_code="out_of_memory",
            )
    if s.peak_host_memory_fraction is not None:
        if s.peak_host_memory_fraction > config.maximum_peak_host_memory_fraction:
            return ThresholdDecision(
                decision="fail_run",
                reason=(
                    f"peak host memory fraction {s.peak_host_memory_fraction} exceeds "
                    f"{config.maximum_peak_host_memory_fraction}"
                ),
                diagnostic_code="out_of_memory",
            )
    budget_fraction = s.update / s.optimizer_updates if s.optimizer_updates else 0.0
    if (
        s.initial_validation_loss is not None
        and s.current_validation_loss is not None
        and budget_fraction >= 0.25
        and budget_fraction < 0.5
    ):
        improvement = s.initial_validation_loss - s.current_validation_loss
        if improvement < config.minimum_loss_improvement_at_quarter_budget_nats:
            return ThresholdDecision(
                decision="fail_run",
                reason=(
                    f"loss improvement {improvement} nats at 25% of budget is below the "
                    f"{config.minimum_loss_improvement_at_quarter_budget_nats} floor"
                ),
                diagnostic_code="optimizer_failure",
            )
    if (
        s.initial_validation_loss is not None
        and s.current_validation_loss is not None
        and budget_fraction >= 0.5
        and not s.is_final_boundary
    ):
        improvement = s.initial_validation_loss - s.current_validation_loss
        if improvement < config.minimum_loss_improvement_at_half_budget_nats:
            return ThresholdDecision(
                decision="fail_run",
                reason=(
                    f"loss improvement {improvement} nats at 50% of budget is below the "
                    f"{config.minimum_loss_improvement_at_half_budget_nats} floor"
                ),
                diagnostic_code="optimizer_failure",
            )
    return None


def _canonical_final_checks(
    config: D0ThresholdConfig, s: RunThresholdSnapshot
) -> ThresholdDecision | None:
    """Canonical-only acceptance gates; each activates only when configured.

    Called only at the final boundary, where ``current_validation_loss`` /
    ``current_median_throughput`` hold the final values.
    """

    if config.maximum_final_loss_above_best_prior_nats is not None:
        if s.current_validation_loss is not None and s.best_validation_loss is not None:
            excess = s.current_validation_loss - s.best_validation_loss
            if excess > config.maximum_final_loss_above_best_prior_nats:
                return ThresholdDecision(
                    decision="reject_acceptance",
                    reason=(
                        f"final loss exceeds the best prior validation loss by {excess} "
                        f"nats, over the allowed "
                        f"{config.maximum_final_loss_above_best_prior_nats}"
                    ),
                )
    if (
        config.maximum_consecutive_regressing_validation_boundaries is not None
        and s.consecutive_regressing_boundaries
        > config.maximum_consecutive_regressing_validation_boundaries
    ):
        return ThresholdDecision(
            decision="reject_acceptance",
            reason=(
                f"{s.consecutive_regressing_boundaries} consecutive regressing validation "
                "boundaries exceed the allowed "
                f"{config.maximum_consecutive_regressing_validation_boundaries}"
            ),
        )
    if config.minimum_final_to_initial_throughput_ratio is not None:
        if (
            s.first_window_median_throughput is not None
            and s.current_median_throughput is not None
            and s.first_window_median_throughput > 0
        ):
            ratio = s.current_median_throughput / s.first_window_median_throughput
            if ratio < config.minimum_final_to_initial_throughput_ratio:
                return ThresholdDecision(
                    decision="reject_acceptance",
                    reason=(
                        f"final throughput {s.current_median_throughput} is "
                        f"{ratio} of the first window "
                        f"{s.first_window_median_throughput}, below the required "
                        f"{config.minimum_final_to_initial_throughput_ratio}"
                    ),
                )
    return None


def evaluate_thresholds(
    config: D0ThresholdConfig, snapshot: RunThresholdSnapshot
) -> ThresholdDecision:
    """Evaluate the frozen matrix at one point and return the decision.

    Evaluation order: kill conditions, then run-failure conditions, then —
    at the final boundary — the acceptance gate. ``continue`` means no
    threshold triggered.
    """

    killed = _kill_checks(config, snapshot)
    if killed is not None:
        return killed
    failed = _fail_checks(config, snapshot)
    if failed is not None:
        return failed
    if snapshot.is_final_boundary:
        if (
            snapshot.initial_validation_loss is not None
            and snapshot.current_validation_loss is not None
        ):
            improvement = snapshot.initial_validation_loss - snapshot.current_validation_loss
            if improvement < config.minimum_final_validation_loss_improvement_nats:
                return ThresholdDecision(
                    decision="reject_acceptance",
                    reason=(
                        f"final validation loss improvement {improvement} nats is below "
                        f"the required "
                        f"{config.minimum_final_validation_loss_improvement_nats} nats"
                    ),
                )
            canonical = _canonical_final_checks(config, snapshot)
            if canonical is not None:
                return canonical
            return ThresholdDecision(
                decision="continue",
                reason=(
                    f"acceptance gate met: final improvement {improvement} nats at or "
                    f"above {config.minimum_final_validation_loss_improvement_nats}"
                ),
            )
    return ThresholdDecision(decision="continue", reason="no threshold triggered")
