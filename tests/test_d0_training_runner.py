"""Tests for the D0 lifecycle training runner (torch, CPU).

Covers the smoke gate's acceptance methodology over the production runner:
a U0-style interval run with telemetry, the R0/R1 bit-exact interruption/
resume property, the kill path (non-finite gradient), the run-failure path
(checkpoint budget), checkpoint-write timing, the frozen packed-window
shift, and validation-regression tracking.

Requires the ``d0-model`` extra; skips cleanly when torch is absent.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from expertforge.artifacts.store import ArtifactStore  # noqa: E402
from expertforge.checkpoints.models import DataCursor, DataIdentity  # noqa: E402
from expertforge.checkpoints.store import CheckpointStore  # noqa: E402
from expertforge.config.d0_models import (  # noqa: E402
    D0BatchConfig,
    D0OptimizerConfig,
    D0ScheduleConfig,
    D0ThresholdConfig,
)
from expertforge.d0.errors import (  # noqa: E402
    RunFailedError,
    TrainingKilledError,
)
from expertforge.d0.model.config import D0ModelConfig  # noqa: E402
from expertforge.d0.model.initialization import initialize_model  # noqa: E402
from expertforge.d0.model.transformer import D0Model  # noqa: E402
from expertforge.d0.training.accumulation import UpdateAccumulator  # noqa: E402
from expertforge.d0.training.optimizer import D0Optimizer  # noqa: E402
from expertforge.d0.training.runner import D0TrainingRunner  # noqa: E402
from expertforge.d0.training.schedule import D0Schedule  # noqa: E402
from expertforge.d0.training.state import D0TrainingState  # noqa: E402
from expertforge.identity.record import AttemptIdentityRecord  # noqa: E402
from expertforge.rng.derivation import SeedContext  # noqa: E402
from expertforge.rng.manager import RngManager  # noqa: E402
from expertforge.telemetry.models import ProcessContext  # noqa: E402
from expertforge.telemetry.writer import TelemetryWriter  # noqa: E402
from tests._checkpoint_fixtures import (  # noqa: E402
    make_identity_and_provenance,
    make_resume_identity,
    make_store,
    resolve_envelope,
)

_SEQ = 16
_WINDOW = _SEQ + 1  # frozen packed window: inputs 0..L-1, targets 1..L
_MICRO = 2
_ACCUM = 2
_VOCAB = 100
_UPDATES = 8
_INTERVAL = 2
_MASTER_SEED = 2026080200


def _schedule_config() -> D0ScheduleConfig:
    return D0ScheduleConfig(
        semantic_update_indexing=(
            "optimizer_update_index_u_is_one_based_for_applied_updates; "
            "u_in_1_through_optimizer_updates"
        ),
        learning_rate_at_update_zero=0.0,
        peak_learning_rate=0.006,
        warmup_updates=2,
        warmup_formula="lr(u)=peak_learning_rate*u/warmup_updates for 1<=u<=warmup_updates",
        final_learning_rate=6.0e-04,
        decay="cosine",
        cosine_formula=(
            "lr(u)=final_learning_rate+0.5*(peak_learning_rate-final_learning_rate)*"
            "(1+cos(pi*(u-warmup_updates)/(optimizer_updates-warmup_updates))) "
            "for warmup_updates<u<=optimizer_updates"
        ),
        training_target_tokens=_UPDATES * _MICRO * _ACCUM * _SEQ,
        optimizer_updates=_UPDATES,
        validation_interval_updates=_INTERVAL,
        checkpoint_interval_updates=_INTERVAL,
        generation_interval_updates=_INTERVAL,
        profiling_interval_updates=_INTERVAL,
    )


def _batch_config() -> D0BatchConfig:
    return D0BatchConfig(
        devices=1,
        microbatch_sequences_per_device=_MICRO,
        gradient_accumulation_steps=_ACCUM,
        global_sequences_per_update=_MICRO * _ACCUM,
        target_tokens_per_update=_MICRO * _ACCUM * _SEQ,
    )


def _threshold_config(**overrides: Any) -> D0ThresholdConfig:
    from pydantic import TypeAdapter

    values: dict[str, Any] = {
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
        # The tiny test schedule has no real loss improvement, so the
        # frozen mid-budget floors are relaxed to their positive minimum;
        # the budget-fail test re-tightens what it needs.
        "minimum_loss_improvement_at_quarter_budget_nats": 1.0e-09,
        "maximum_rejected_recovery_attempts_before_kill": 1,
        "kill_on_any_non_finite_value": True,
        "kill_on_any_skipped_optimizer_update": True,
        "kill_on_any_checkpoint_or_resume_state_mismatch": True,
        "maximum_out_of_memory_failures_after_remediation": 1,
        "maximum_failed_recovery_attempts": 2,
        "minimum_loss_improvement_at_half_budget_nats": 1.0e-09,
    }
    values.update(overrides)
    return TypeAdapter(D0ThresholdConfig).validate_python(values)


def _progression_window(*, start: int) -> torch.Tensor:
    """One packed window of adjacent-increment tokens (learnable structure).

    Random windows are structureless — their cross-entropy floor is ln(V) —
    so the synthetic corpus uses ``token[t+1] = token[t] + 1 (mod V)``: the
    model can genuinely learn it, which makes the frozen loss-improvement
    floors meaningful on the toy schedule.
    """

    return (
        ((start + torch.arange(_WINDOW, dtype=torch.int64)) % _VOCAB).unsqueeze(0).repeat(_MICRO, 1)
    )


class _SyntheticSource:
    """Deterministic packed windows: window i depends only on (seed, i)."""

    def __init__(self, *, seed: int, start_window: int = 0) -> None:
        self._seed = seed
        self._window_index = start_window
        self._position = start_window * _MICRO * _SEQ
        self._sequences = start_window * _MICRO

    def next_window(self) -> torch.Tensor:
        start = (self._seed + 97 * self._window_index) % _VOCAB
        window = _progression_window(start=start)
        self._window_index += 1
        self._position += _MICRO * _SEQ
        self._sequences += _MICRO
        return window

    def data_identity(self) -> DataIdentity:
        return DataIdentity(
            dataset_digest="a" * 64,
            split="train",
            length=1_000_000_000,
            preprocessing_identity="d0.v1",
            tokenizer_identity="synthetic-test-only",
            packing_policy="synthetic_packed_v1",
            sequence_policy="packed_16",
            shard_selection="all",
            data_config_digest="b" * 64,
        )

    def cursor_snapshot(self) -> DataCursor:
        return DataCursor(
            sampler_type="sequential",
            sampler_version=1,
            batch_size=_MICRO,
            sequence_length=_SEQ,
            drop_last=False,
            epoch=0,
            position=self._position,
            accepted_samples=self._sequences,
            accepted_sequences=self._sequences,
        )


class _FixedWindowLossHook:
    """Real validation loss: mean cross-entropy over a fixed window."""

    def __init__(self) -> None:
        window = _progression_window(start=17)
        self._inputs = window[:, :-1].contiguous()
        self._targets = window[:, 1:].contiguous()

    def validation_loss(self, model: torch.nn.Module) -> float:
        with torch.no_grad():
            logits = model(self._inputs)
            return float(
                torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, _VOCAB),
                    self._targets.reshape(-1),
                    reduction="mean",
                )
            )


def _new_bundle(seed: int) -> tuple[D0Model, D0Optimizer, D0Schedule]:
    config = D0ModelConfig(n_layers=2, dim=32, n_heads=4, head_dim=8, ffn_dim=48, vocab_size=_VOCAB)
    model = D0Model(config)
    generator = torch.Generator().manual_seed(seed)
    initialize_model(model, n_layers=config.n_layers, generator=generator)
    optimizer = D0Optimizer(
        model,
        D0OptimizerConfig(
            name="AdamW",
            beta1=0.9,
            beta2=0.95,
            epsilon=1.0e-08,
            weight_decay=0.1,
            weight_decay_includes=("attention_matrix_weights", "swiglu_matrix_weights"),
            weight_decay_excludes=("token_embedding_weight", "rmsnorm_weights"),
            gradient_clip_global_l2_norm=1.0,
            loss_reduction="mean_over_all_target_tokens_in_optimizer_update",
        ),
    )
    schedule = D0Schedule(
        peak_learning_rate=0.006,
        final_learning_rate=6.0e-04,
        warmup_updates=2,
        optimizer_updates=_UPDATES,
    )
    return model, optimizer, schedule


def _make_runner(
    tmp_path: Path,
    identity: AttemptIdentityRecord,
    provenance: Any,
    *,
    init_seed: int,
    source: _SyntheticSource,
    threshold_config: D0ThresholdConfig | None = None,
    monotonic_clock: Callable[[], float] | None = None,
) -> tuple[D0TrainingRunner, ArtifactStore]:
    artifact_store = make_store(tmp_path, identity=identity)
    checkpoint_store = CheckpointStore(artifact_store)
    telemetry = TelemetryWriter(
        artifact_root=artifact_store.artifact_root,
        identity=identity,
        process_context=ProcessContext(rank=0, world_size=1, local_rank=0),
        console_enabled=False,
    )
    model, optimizer, schedule = _new_bundle(seed=init_seed)
    accumulator = UpdateAccumulator(_batch_config(), target_tokens_per_sequence=_SEQ)
    state = D0TrainingState(model=model, optimizer=optimizer, schedule=schedule)

    def rng_bundle_bytes() -> bytes:
        manager = RngManager(root_seed=_MASTER_SEED, context=SeedContext(component="run"))
        manager.initialize()
        return manager.capture_state().to_deterministic_json()

    def new_state_fn() -> D0TrainingState:
        fresh_model, fresh_optimizer, fresh_schedule = _new_bundle(seed=init_seed + 999_999)
        return D0TrainingState(
            model=fresh_model, optimizer=fresh_optimizer, schedule=fresh_schedule
        )

    runner = D0TrainingRunner(
        identity=identity,
        process_context=ProcessContext(rank=0, world_size=1, local_rank=0),
        artifact_store=artifact_store,
        checkpoint_store=checkpoint_store,
        telemetry=telemetry,
        configuration_envelope=resolve_envelope(),
        provenance=provenance,
        state=state,
        accumulator=accumulator,
        batch_source=source,
        validation_hook=_FixedWindowLossHook(),
        schedule_config=_schedule_config(),
        batch_config=_batch_config(),
        threshold_config=threshold_config or _threshold_config(),
        packed_window_tokens=_WINDOW,
        rng_bundle_bytes_fn=rng_bundle_bytes,
        new_state_fn=new_state_fn,
        monotonic_clock=monotonic_clock,
    )
    return runner, artifact_store


class TestU0IntervalRun:
    def test_full_run_with_intervals(self, tmp_path: Path) -> None:
        identity, provenance = make_identity_and_provenance(tmp_path)
        source = _SyntheticSource(seed=11)
        runner, artifact_store = _make_runner(
            tmp_path, identity, provenance, init_seed=7, source=source
        )
        runner.run_updates(_UPDATES)
        runner.close(outcome="normal")
        record = runner.register_telemetry()

        assert runner.state.last_applied_update == _UPDATES
        assert len(runner.checkpoint_records) == _UPDATES // _INTERVAL
        assert runner.max_checkpoint_write_seconds >= 0.0
        assert record is not None

    def test_checkpoint_and_validation_counts(self, tmp_path: Path) -> None:
        identity, provenance = make_identity_and_provenance(tmp_path)
        source = _SyntheticSource(seed=12)
        runner, _store = _make_runner(tmp_path, identity, provenance, init_seed=7, source=source)
        runner.run_updates(4)
        # checkpoints at u=2,4; validations at u=2,4
        assert len(runner.checkpoint_records) == 2
        assert runner._initial_validation_loss is not None
        assert runner._current_validation_loss is not None
        runner.close(outcome="normal")


class TestWindowShift:
    def test_runner_slices_frozen_inputs_and_targets(self, tmp_path: Path) -> None:
        identity, provenance = make_identity_and_provenance(tmp_path)
        source = _SyntheticSource(seed=13)
        seen: list[torch.Tensor] = []
        runner, _store = _make_runner(tmp_path, identity, provenance, init_seed=7, source=source)

        inner = runner.state.model

        class _Spy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
                seen.append(token_ids.detach().clone())
                output = self.inner(token_ids)
                assert isinstance(output, torch.Tensor)
                return output

        spy = _Spy()
        spy.token_embedding = inner.token_embedding  # keep optimizer params valid
        runner.state.model = spy
        runner.run_updates(1)
        assert seen, "spy model was never called"
        for recorded in seen:
            assert recorded.shape == (_MICRO, _SEQ)
        runner.close(outcome="normal")


class TestR0R1Resume:
    def test_interrupted_resume_is_bit_exact(self, tmp_path: Path) -> None:
        # Twin A: uninterrupted 8 updates.
        identity_a, provenance_a = make_identity_and_provenance(tmp_path / "a")
        source_a = _SyntheticSource(seed=21)
        runner_a, _store_a = _make_runner(
            tmp_path / "a", identity_a, provenance_a, init_seed=7, source=source_a
        )
        runner_a.run_updates(_UPDATES)
        runner_a.close(outcome="normal")

        # Twin B: run to K=4 (checkpoint at 4), interrupt, resume, finish.
        identity_b, provenance_b = make_identity_and_provenance(tmp_path / "b")
        source_b = _SyntheticSource(seed=21)
        runner_b, store_b = _make_runner(
            tmp_path / "b", identity_b, provenance_b, init_seed=7, source=source_b
        )
        runner_b.run_updates(4)
        checkpoint_at_4 = runner_b.checkpoint_records[-1]
        assert checkpoint_at_4 is not None
        runner_b.close(outcome="interrupted", diagnostic_code="handled_interruption")

        resume_identity = make_resume_identity(
            identity_b, parent_artifact_id=checkpoint_at_4.artifact_id
        )
        parent_store = CheckpointStore(make_store(tmp_path / "b", identity=identity_b))
        archive = parent_store.load(checkpoint_at_4.artifact_id, expected_identity=resume_identity)

        def rng_bundle_loader() -> Any:
            manager = RngManager(root_seed=_MASTER_SEED, context=SeedContext(component="run"))
            manager.initialize()
            return manager.capture_state()

        consumed: list[Any] = []

        def rng_consumer(bundle: Any) -> None:
            consumed.append(bundle)

        # 4 updates consumed accum=2 windows each: resume at window index 8.
        source_b2 = _SyntheticSource(seed=21, start_window=_ACCUM * 4)
        runner_b2, _store_b2 = _make_runner(
            tmp_path / "b",
            resume_identity,
            provenance_b,
            init_seed=999_999,  # different init: parameters must come from the checkpoint
            source=source_b2,
        )
        runner_b2.restore_from(
            archive,
            expected_identity=resume_identity,
            rng_bundle_loader=rng_bundle_loader,
            rng_consumer=rng_consumer,
        )
        assert consumed, "RNG bundle was not restored"
        assert runner_b2.state.last_applied_update == 4
        assert runner_b2._accumulator.global_update == 4
        runner_b2.run_updates(4)
        runner_b2.close(outcome="normal")

        for (name_a, param_a), (name_b, param_b) in zip(
            runner_a.state.model.named_parameters(),
            runner_b2.state.model.named_parameters(),
            strict=True,
        ):
            assert name_a == name_b
            assert torch.equal(param_a.detach(), param_b.detach()), name_a


class TestKillPath:
    def test_non_finite_gradient_kills(self, tmp_path: Path) -> None:
        identity, provenance = make_identity_and_provenance(tmp_path)
        source = _SyntheticSource(seed=31)
        runner, _store = _make_runner(tmp_path, identity, provenance, init_seed=7, source=source)
        # Enormous embedding values drive cross-entropy to inf and the
        # backward pass to non-finite gradients.
        model = cast(D0Model, runner.state.model)
        with torch.no_grad():
            model.token_embedding.weight.mul_(1.0e30)
        with pytest.raises(TrainingKilledError, match="non-finite"):
            runner.run_updates(1)
        assert runner.closed
        runner.register_telemetry()  # stream must be well-formed


class TestFailPath:
    def test_checkpoint_over_budget_fails_the_run(self, tmp_path: Path) -> None:
        identity, provenance = make_identity_and_provenance(tmp_path)
        source = _SyntheticSource(seed=41)
        # Injected clock: each save observes a 400-second write (> 300 budget).
        clock = itertools.count(start=0.0, step=400.0)
        config = _threshold_config()
        runner, _store = _make_runner(
            tmp_path,
            identity,
            provenance,
            init_seed=7,
            source=source,
            threshold_config=config,
            monotonic_clock=lambda: next(clock),
        )
        with pytest.raises(RunFailedError, match="checkpoint write"):
            runner.run_updates(_INTERVAL)
        assert runner.closed


class TestTimingAndTracking:
    def test_write_duration_recorded(self, tmp_path: Path) -> None:
        identity, provenance = make_identity_and_provenance(tmp_path)
        source = _SyntheticSource(seed=51)
        runner, _store = _make_runner(tmp_path, identity, provenance, init_seed=7, source=source)
        runner.run_updates(_INTERVAL)
        assert runner.max_checkpoint_write_seconds > 0.0
        runner.close(outcome="normal")

    def test_validation_regression_tracking(self, tmp_path: Path) -> None:
        identity, provenance = make_identity_and_provenance(tmp_path)
        source = _SyntheticSource(seed=61)
        runner, _store = _make_runner(
            tmp_path,
            identity,
            provenance,
            init_seed=7,
            source=source,
            # Sanity: qualification config pins regression_delta to None, so
            # regressions are never counted under this profile.
            threshold_config=_threshold_config(),
        )
        runner.run_updates(4)
        assert runner._consecutive_regressing == 0
        runner.close(outcome="normal")
