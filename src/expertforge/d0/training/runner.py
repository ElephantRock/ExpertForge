"""The D0 lifecycle training runner over a batch-source protocol.

Drives the frozen update loop (forward → exact loss reduction → backward →
clip+step at the scheduled LR), the checkpoint/validation interval
machinery, telemetry, and the frozen accept/fail/kill threshold matrix —
all over an abstract :class:`BatchSource`. The real D0.1b packed-window
source plugs into that protocol; tests supply a deterministic synthetic
source.

Frozen semantics implemented here:

- **Packed-window shift** (DEC-0003): a source window of
  ``packed_window_tokens`` positions yields ``inputs = window[:, :-1]`` and
  ``targets = window[:, 1:]`` — inputs are tokens ``0..L-1``, targets
  ``1..L``; every target position participates (no padding).
- **Loss reduction**: per-microstep cross-entropy summed over target
  tokens in float32 and handed to the T1 accumulator, which produces the
  exact ``mean_over_all_target_tokens_in_optimizer_update``.
- **Clip point**: after accumulation, before the step (T1 optimizer).
- **Checkpoint timing**: saves are timed with a monotonic clock; the
  observed write durations feed the frozen threshold budget (300 s). The
  smoke gate never timed saves; the D0 contract requires it.
- **Threshold outcomes**: ``kill`` closes telemetry failed and raises
  :class:`TrainingKilledError`; ``fail_run`` closes failed and raises
  :class:`RunFailedError`; ``reject_acceptance`` is recorded as the gate
  verdict (the run itself completed normally). A non-finite gradient from
  the optimizer maps directly to kill with ``non_finite_gradient``.
- **Precision**: when ``autocast_bf16`` is set, the forward runs under
  CUDA bf16 autocast over fp32 master parameters (no scaler). Enforcement
  of the preflight belongs to the orchestrator; the runner only executes
  the chosen strategy.

Out of scope (later tranches): provenance/manifest orchestration, the
real D0.1b batch source, distributed execution, generation, and any
material training run (D0.5 remains unauthorized).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch import nn

from expertforge.artifacts.models import ParentReference
from expertforge.artifacts.store import ArtifactStore
from expertforge.checkpoints.models import DataCursor, DataIdentity
from expertforge.checkpoints.restore import RestoreTransaction
from expertforge.checkpoints.store import CheckpointStore
from expertforge.config.d0_models import D0BatchConfig, D0ScheduleConfig, D0ThresholdConfig
from expertforge.d0.errors import (
    D0TrainingError,
    NonFiniteGradientError,
    RunFailedError,
    TrainingKilledError,
)
from expertforge.d0.training.accumulation import UpdateAccumulator
from expertforge.d0.training.state import D0StateFactory, D0StateProvider, D0TrainingState
from expertforge.d0.training.thresholds import (
    RunThresholdSnapshot,
    ThresholdDecision,
    evaluate_thresholds,
)
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.telemetry.models import MetricObservation, ProcessContext, ProgressPosition
from expertforge.telemetry.writer import TelemetryWriter

__all__ = [
    "BatchSource",
    "ValidationHook",
    "D0TrainingRunner",
]


class BatchSource(Protocol):
    """One microstep's packed windows plus the frozen data identity.

    ``next_window`` returns an int64 tensor of shape
    ``(microbatch_sequences_per_device, packed_window_tokens)`` — the
    runner performs the frozen shift into inputs/targets.
    """

    def next_window(self) -> torch.Tensor: ...

    def data_identity(self) -> DataIdentity: ...

    def cursor_snapshot(self) -> DataCursor: ...


class ValidationHook(Protocol):
    """Computes the validation loss (natural-log nats) for one boundary."""

    def validation_loss(self, model: nn.Module) -> float: ...


def _loss_observation(value: float) -> MetricObservation:
    return MetricObservation(
        namespace="training",
        name="cross_entropy_loss",
        unit="dimensionless",
        aggregation="gauge",
        window="point",
        value_status="finite",
        value=float(value),
    )


def _validation_observation(value: float) -> MetricObservation:
    return MetricObservation(
        namespace="validation",
        name="cross_entropy_loss",
        unit="dimensionless",
        aggregation="gauge",
        window="point",
        value_status="finite",
        value=float(value),
    )


class D0TrainingRunner:
    """Drives the frozen D0 update loop over a :class:`BatchSource`."""

    def __init__(
        self,
        *,
        identity: AttemptIdentityRecord,
        process_context: ProcessContext,
        artifact_store: ArtifactStore,
        checkpoint_store: CheckpointStore,
        telemetry: TelemetryWriter,
        configuration_envelope: Any,
        provenance: Any,
        state: D0TrainingState,
        accumulator: UpdateAccumulator,
        batch_source: BatchSource,
        validation_hook: ValidationHook,
        schedule_config: D0ScheduleConfig,
        batch_config: D0BatchConfig,
        threshold_config: D0ThresholdConfig,
        packed_window_tokens: int,
        rng_bundle_bytes_fn: Callable[[], bytes],
        new_state_fn: Callable[[], D0TrainingState],
        autocast_bf16: bool = False,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._identity = identity
        self._process_context = process_context
        self._artifact_store = artifact_store
        self._checkpoint_store = checkpoint_store
        self._telemetry = telemetry
        self._configuration_envelope = configuration_envelope
        self._provenance = provenance
        self.state = state
        self._accumulator = accumulator
        self._batch_source = batch_source
        self._validation_hook = validation_hook
        self._schedule_config = schedule_config
        self._batch_config = batch_config
        self._threshold_config = threshold_config
        if type(packed_window_tokens) is not int or packed_window_tokens < 2:
            raise D0TrainingError(
                "packed_window_tokens must be an integer >= 2 (inputs + targets); "
                f"got {packed_window_tokens!r}"
            )
        self._packed_window_tokens = packed_window_tokens
        self._rng_bundle_bytes_fn = rng_bundle_bytes_fn
        self._new_state_fn = new_state_fn
        self._autocast_bf16 = autocast_bf16
        self._monotonic_clock = monotonic_clock or time.monotonic

        self._provider = D0StateProvider(
            state=state,
            accumulator=accumulator,
            rng_bundle_bytes=self._rng_bundle_bytes_fn,
            data_identity=batch_source.data_identity(),
        )
        # Observed threshold inputs.
        self._initial_validation_loss: float | None = None
        self._current_validation_loss: float | None = None
        self._validation_is_fresh: bool = False
        self._validation_count: int = 0
        self._best_validation_loss: float | None = None
        self._consecutive_regressing: int = 0
        self._max_checkpoint_write_seconds: float = 0.0
        self.gate_verdict: ThresholdDecision | None = None
        self.checkpoint_records: list[Any] = []
        self.closed = False

    # -- update loop --------------------------------------------------------

    def run_updates(self, count: int) -> None:
        """Run ``count`` applied updates with intervals and thresholds.

        If no validation has happened yet, the update-zero control runs
        first: the frozen contract validates the untrained model before the
        first update (the smoke-gate loss-movement criterion), so the
        mid-budget improvement floors measure against a real update-0
        baseline rather than the first interval boundary.
        """

        if type(count) is not int or count < 0:
            raise D0TrainingError(f"count must be a non-negative exact integer; got {count!r}")
        if self._initial_validation_loss is None:
            self._validate_boundary()
        for _ in range(count):
            self._run_one_update()
            update = self.state.last_applied_update
            if (
                self._schedule_config.checkpoint_interval_updates > 0
                and update % self._schedule_config.checkpoint_interval_updates == 0
            ):
                self.save_checkpoint()
            if (
                self._schedule_config.validation_interval_updates > 0
                and update % self._schedule_config.validation_interval_updates == 0
            ):
                self._validate_boundary()
            is_final = update == self._schedule_config.optimizer_updates
            decision = self._evaluate_thresholds(is_final=is_final)
            if decision.decision == "kill":
                self.close(outcome="failed", diagnostic_code=decision.diagnostic_code)
                raise TrainingKilledError(
                    decision, diagnostic_code=decision.diagnostic_code or "optimizer_failure"
                )
            if decision.decision == "fail_run":
                self.close(outcome="failed", diagnostic_code=decision.diagnostic_code)
                raise RunFailedError(
                    decision, diagnostic_code=decision.diagnostic_code or "optimizer_failure"
                )

    def _run_one_update(self) -> None:
        state = self.state
        update = state.last_applied_update + 1
        if update > state.schedule.optimizer_updates:
            raise D0TrainingError(
                f"update {update} exceeds the frozen budget {state.schedule.optimizer_updates}"
            )
        learning_rate = state.schedule.learning_rate_at(update)
        # Fresh gradient buffer per update: microsteps accumulate WITHIN the
        # update through .grad; the previous update's gradients must never
        # leak into this one's clip+step.
        state.optimizer.zero_grad()
        self._accumulator.start_update()
        for _microstep in range(self._accumulator.microsteps_per_update):
            window = self._batch_source.next_window()
            if window.ndim != 2 or int(window.shape[1]) != self._packed_window_tokens:
                raise D0TrainingError(
                    f"batch source window shape {tuple(window.shape)} does not match the "
                    f"frozen packed window (micro_sequences, {self._packed_window_tokens})"
                )
            inputs = window[:, :-1]
            targets = window[:, 1:]
            if self._autocast_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = state.model(inputs)
            else:
                logits = state.model(inputs)
            loss_sum = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            loss_sum.backward()  # type: ignore[no-untyped-call]
            self._accumulator.record_microstep(
                token_loss_sum=float(loss_sum.detach()),
                target_token_count=targets.numel(),
            )
        mean_loss = self._accumulator.complete_update()
        try:
            state.optimizer.clip_and_step(learning_rate=learning_rate)
        except NonFiniteGradientError as exc:
            decision = ThresholdDecision(
                decision="kill",
                reason=str(exc),
                diagnostic_code="non_finite_gradient",
            )
            self.close(outcome="failed", diagnostic_code="non_finite_gradient")
            raise TrainingKilledError(decision, diagnostic_code="non_finite_gradient") from exc
        state.last_applied_update = update
        self._telemetry.emit_metric(
            component="training",
            observations=[_loss_observation(mean_loss)],
            progress=ProgressPosition(
                step=update,
                update=update,
                processed_tokens=self._accumulator.processed_tokens,
            ),
        )

    # -- intervals -----------------------------------------------------------

    def _validate_boundary(self) -> None:
        value = float(self._validation_hook.validation_loss(self.state.model))
        if self._initial_validation_loss is None:
            self._initial_validation_loss = value
        else:
            delta = self._threshold_config.regression_boundary_delta_nats
            previous = self._current_validation_loss
            if previous is not None and delta is not None and value - previous > delta:
                self._consecutive_regressing += 1
            else:
                self._consecutive_regressing = 0
        self._current_validation_loss = value
        self._validation_is_fresh = True
        self._validation_count += 1
        if self._best_validation_loss is None or value < self._best_validation_loss:
            self._best_validation_loss = value
        self._telemetry.emit_metric(
            component="validation",
            observations=[_validation_observation(value)],
            progress=ProgressPosition(
                step=self.state.last_applied_update,
                update=self.state.last_applied_update,
                processed_tokens=self._accumulator.processed_tokens,
            ),
        )
        self._telemetry.emit_event(
            component="validation",
            severity="INFO",
            event_name="validation.completed",
            progress=ProgressPosition(update=self.state.last_applied_update),
            fields=(("validation_loss", value),),
        )

    def save_checkpoint(self, *, parent: ParentReference | None = None) -> Any:
        """Capture and store one checkpoint, timed for the frozen budget."""

        started = self._monotonic_clock()
        captured = self._provider.capture_checkpoint_snapshot(
            data_cursor=self._batch_source.cursor_snapshot()
        )
        record = self._checkpoint_store.save(
            identity=self._identity,
            captured=captured,
            configuration_envelope=self._configuration_envelope,
            provenance=self._provenance,
            parent=parent,
        )
        duration = self._monotonic_clock() - started
        if duration > self._max_checkpoint_write_seconds:
            self._max_checkpoint_write_seconds = duration
        self.checkpoint_records.append(record)
        self._telemetry.emit_event(
            component="checkpoints",
            severity="INFO",
            event_name="checkpoint.saved",
            progress=ProgressPosition(update=self.state.last_applied_update),
            fields=(
                ("checkpoint_id", record.artifact_id),
                ("global_update", int(self.state.last_applied_update)),
                ("write_seconds", duration),
            ),
        )
        return record

    # -- thresholds -----------------------------------------------------------

    def _evaluate_thresholds(self, *, is_final: bool) -> ThresholdDecision:
        # Loss-improvement floors are meaningful only when the freshly
        # measured loss is the SECOND-or-later distinct measurement: the
        # first measurement alone would set initial == current (improvement
        # exactly 0.0), and stale between-boundary snapshots would compare
        # identical values for the same reason. The "initial" baseline is
        # this attempt's first validation — for a resumed attempt, the
        # post-restore state.
        if self._validation_is_fresh and self._validation_count >= 2:
            initial = self._initial_validation_loss
            current = self._current_validation_loss
            self._validation_is_fresh = False
        else:
            initial = None
            current = None
        snapshot = RunThresholdSnapshot(
            update=self.state.last_applied_update,
            optimizer_updates=self.state.schedule.optimizer_updates,
            initial_validation_loss=initial,
            current_validation_loss=current,
            best_validation_loss=self._best_validation_loss,
            consecutive_regressing_boundaries=self._consecutive_regressing,
            skipped_updates=0,
            maximum_checkpoint_write_seconds_observed=self._max_checkpoint_write_seconds,
            non_finite_value_observed=False,
            checkpoint_or_resume_state_mismatch_observed=False,
            is_final_boundary=is_final,
        )
        decision = evaluate_thresholds(self._threshold_config, snapshot)
        if decision.decision == "reject_acceptance":
            self.gate_verdict = decision
        return decision

    @property
    def max_checkpoint_write_seconds(self) -> float:
        return self._max_checkpoint_write_seconds

    # -- checkpoint restore -----------------------------------------------------

    def restore_from(
        self,
        archive: Any,
        *,
        expected_identity: AttemptIdentityRecord,
        rng_bundle_loader: Callable[[], Any],
        rng_consumer: Callable[[Any], None],
    ) -> None:
        """Restore this runner's state from a loaded checkpoint archive.

        A fresh :class:`D0TrainingState` is built through ``new_state_fn``,
        the transaction applies parameters/optimizer/scheduler/counters
        (RNG last via ``rng_consumer``), and the runner adopts the restored
        state, accumulator position, and cursor.
        """

        expected = self._provider.expected_compatibility(
            specification_fingerprint=expected_identity.fingerprint_digest_str(),
            cursor=self._batch_source.cursor_snapshot(),
        )
        factory = D0StateFactory(state_factory=self._new_state_fn)
        transaction = RestoreTransaction(
            archive=archive,
            factory=factory,
            rng_bundle_loader=rng_bundle_loader,
            rng_consumer=rng_consumer,
            expected_descriptor=expected,
        )
        transaction.prepare()
        transaction.commit()
        restored = factory.target
        if restored is None:  # pragma: no cover - transaction commits a target
            raise D0TrainingError("restore completed without a committed target")
        self.state = restored
        # The provider must wrap the ADOPTED state — the pre-restore bundle
        # would capture stale parameters against the advanced accumulator.
        self._provider = D0StateProvider(
            state=self.state,
            accumulator=self._accumulator,
            rng_bundle_bytes=self._rng_bundle_bytes_fn,
            data_identity=self._batch_source.data_identity(),
        )
        counters = factory.restored_counters
        self._accumulator.advance_to(counters.global_update)
        if counters.accumulation_position != 0:
            raise D0TrainingError(  # pragma: no cover - validated by the model
                "restored checkpoint is not quiescent"
            )
        self._telemetry.emit_event(
            component="checkpoints",
            severity="INFO",
            event_name="checkpoint.restored",
            progress=ProgressPosition(update=self.state.last_applied_update),
            fields=(("global_update", int(self.state.last_applied_update)),),
        )

    # -- lifecycle ----------------------------------------------------------------

    def close(self, *, outcome: str, diagnostic_code: str | None = None) -> None:
        """Close the telemetry stream; idempotent after the first call."""

        if self.closed:
            return
        self._telemetry.close(outcome=outcome, diagnostic_code=diagnostic_code)  # type: ignore[arg-type]
        self.closed = True

    def register_telemetry(self) -> Any:
        """Authoritatively validate and register the telemetry stream."""

        return self._artifact_store.register_telemetry(
            self._telemetry.path,
            process_context=self._process_context,
            producing_component="d0_training_runner",
        )
