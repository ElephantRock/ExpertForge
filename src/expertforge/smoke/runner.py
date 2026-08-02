"""Smoke gate orchestrator: U0/R0/R1 topology with exact restoration (Issue #14).

Wires the toy model, optimizer, scheduler, data cursor, RNG, telemetry, artifacts,
checkpoints, and experiment manifests through the real substrate subsystems to
prove the Milestone 0 substrate works end-to-end.

Topology (amendment C/D):

- **U0** (baseline, ``AllocationMode.INDEPENDENT``): train 0..K → checkpoint@K →
  continue K..N → validate → generate → final checkpoint → close(normal) →
  completed manifest.
- **R0** (interrupted, ``AllocationMode.INDEPENDENT``, same spec fingerprint,
  distinct run): train 0..K → parent checkpoint@K → close(interrupted) →
  partial manifest. No validate/generate after K.
- **R1** (resumed, ``AllocationMode.RESUME``): authoritative load of R0's K
  checkpoint → full :class:`RestoreTransaction` (RNG last) → continue K..N →
  validate → generate → final checkpoint → close(normal) → publish comparison
  report → completed resumed manifest with ``resume_checkpoint``.

Per-attempt artifact ordering (correction #8):

1. publish resolved configuration + provenance;
2. train + publish checkpoints;
3. publish generated output (where applicable);
4. close + authoritatively validate telemetry, then register it;
5. (R1) publish the comparison report;
6. manifest finalization LAST.

The attempt-scoped publication object enforces that terminal guard: every
publication method raises :class:`SmokeTerminalStateError` once the attempt's
terminal manifest has been finalized (correction #8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from expertforge.artifacts import ArtifactRecord, ArtifactStore, ExternalReference, ParentReference
from expertforge.checkpoints import (
    CheckpointStore,
    RestoreTransaction,
)
from expertforge.config.resolve import ResolutionEnvelope, canonical_bytes
from expertforge.experiments import (
    EvaluationSummary,
    ManifestBindingError,
    ManifestGenerator,
    ModelIdentity,
    ParentCheckpointResolver,
    TrainingBudget,
)
from expertforge.identity.emit import AllocationMode
from expertforge.identity.lineage import ResumeLineage
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.record import ProvenanceRecord
from expertforge.rng.manager import RngManager
from expertforge.smoke import fixtures
from expertforge.smoke.comparison import ComputationalState, compare, extract_computational_state
from expertforge.smoke.data import SmokeDataCursor
from expertforge.smoke.model import MODEL_ARCHITECTURE, SmokeModel
from expertforge.smoke.optimizer import SmokeAdamW
from expertforge.smoke.report import ComparisonEnvironment, ComparisonReport
from expertforge.smoke.scheduler import SmokeScheduler
from expertforge.smoke.state import SmokeRuntime, SmokeStateFactory, SmokeStateProvider
from expertforge.telemetry.models import ProcessContext
from expertforge.telemetry.writer import TelemetryWriter

__all__ = [
    "SmokeAttemptResult",
    "SmokeTerminalStateError",
    "AttemptScope",
    "SmokeRunner",
]


class SmokeTerminalStateError(Exception):
    """Raised when any artifact is published after an attempt's terminal manifest."""


@dataclass
class SmokeAttemptResult:
    """The terminal result of one smoke attempt (U0/R0/R1).

    F8/B1: this result deliberately does NOT expose a publish-capable
    :class:`ArtifactStore`. The attempt-scoped terminal guard lives on the
    :class:`AttemptScope` that produced the attempt; that scope is returned as
    ``sealed_scope`` so callers/tests can prove the guard rejects further
    publication on the REAL finalized scope (not by reconstructing a fresh
    store). Callers that need read-only access to a prior attempt's artifacts
    reconstruct the store from the attempt's identity (read-only
    ``ArtifactStore`` usage does not bypass the guard, which governs
    publication, not reads).
    """

    identity: AttemptIdentityRecord
    manifest_record: ArtifactRecord
    # The sealed (terminal) scope used for this attempt. Exposing it lets tests
    # prove post-manifest publication is rejected on the real scope. It is
    # terminal, so every publication method on it raises.
    sealed_scope: AttemptScope
    # Checkpoints produced by this attempt (K and/or final), keyed by role.
    checkpoints: dict[str, ArtifactRecord] = field(default_factory=dict)
    telemetry_record: ArtifactRecord | None = None
    generated_record: ArtifactRecord | None = None
    # Computational state at N (U0 and R1 only).
    computational_state: ComputationalState | None = None
    initial_validation_loss: float | None = None
    final_validation_loss: float | None = None
    generated_sample: NDArray[np.int64] | None = None
    # The U0@N vs R1@N comparison report (R1 only).
    comparison_report: Any = None


class AttemptScope:
    """Attempt-scoped publication guard (correction #8).

    All smoke artifact publication for one attempt flows through this object.
    After :meth:`finalize_manifest` succeeds, every subsequent publication method
    raises :class:`SmokeTerminalStateError`.
    """

    def __init__(
        self,
        identity: AttemptIdentityRecord,
        artifact_store: ArtifactStore,
        config_envelope: ResolutionEnvelope,
        provenance: ProvenanceRecord,
        process_context: ProcessContext,
    ) -> None:
        self.identity = identity
        self.store = artifact_store
        self.config_envelope = config_envelope
        self.provenance = provenance
        self.process_context = process_context
        self._terminal = False
        self._telemetry_writer: TelemetryWriter | None = None
        self._telemetry_path: Path | None = None

    @property
    def terminal(self) -> bool:
        return self._terminal

    def _guard(self) -> None:
        if self._terminal:
            raise SmokeTerminalStateError(
                f"attempt {self.identity.attempt_id!r} is terminal; no further publication allowed."
            )

    # -- telemetry lifecycle (owned by TelemetryWriter) ------------------

    def open_telemetry(
        self, *, console_enabled: bool, fsync_interval: int, level: str = "INFO"
    ) -> TelemetryWriter:
        self._guard()
        # Constructing TelemetryWriter emits logging.stream_opened automatically.
        writer = TelemetryWriter(
            artifact_root=self.store.artifact_root,
            identity=self.identity,
            process_context=self.process_context,
            console_enabled=console_enabled,
            fsync_interval_records=fsync_interval,
            level=level,
        )
        self._telemetry_writer = writer
        self._telemetry_path = writer.path
        return writer

    def close_telemetry(self, *, outcome: str, diagnostic_code: str | None = None) -> Path:
        self._guard()
        if self._telemetry_writer is None:
            raise RuntimeError("telemetry writer not opened")
        self._telemetry_writer.close(outcome=outcome, diagnostic_code=diagnostic_code)  # type: ignore[arg-type]
        return self._telemetry_path or self._telemetry_writer.path

    def register_telemetry(self, path: Path) -> ArtifactRecord:
        """Authoritatively validate the stream THEN register it (correction #6)."""
        self._guard()
        record = self.store.register_telemetry(
            path,
            process_context=self.process_context,
            producing_component="telemetry",
        )
        return record

    # -- artifact publication (guarded) ----------------------------------

    def publish_configuration(self) -> ArtifactRecord:
        self._guard()
        return self.store.publish(
            canonical_bytes(self.config_envelope),
            category="resolved_configuration",
            format="json",
            format_version=1,
            producing_component="config",
        )

    def publish_provenance(self) -> ArtifactRecord:
        self._guard()
        return self.store.publish(
            self.provenance.to_deterministic_json(),
            category="provenance",
            format="json",
            format_version=1,
            producing_component="provenance",
        )

    def save_checkpoint(
        self,
        checkpoint_store: CheckpointStore,
        runtime: SmokeRuntime,
        parent: ParentReference | None = None,
        *,
        writer: TelemetryWriter | None = None,
    ) -> ArtifactRecord:
        self._guard()
        captured = SmokeStateProvider(runtime).capture_checkpoint_snapshot()
        record = checkpoint_store.save(
            identity=self.identity,
            captured=captured,
            configuration_envelope=self.config_envelope,
            provenance=self.provenance,
            parent=parent,
        )
        if writer is not None:
            writer.emit_event(
                component="checkpoints",
                severity="INFO",
                event_name="checkpoint.saved",
                progress=_progress(runtime.counters),
                fields=(
                    ("checkpoint_id", record.artifact_id),
                    ("global_update", int(runtime.counters.global_update)),
                ),
            )
        return record

    def publish_generated_output(self, sample: NDArray[np.int64]) -> ArtifactRecord:
        self._guard()
        payload = json.dumps(
            {
                "schema": "expertforge.smoke-generated-sample",
                "schema_version": 1,
                "tokens": [int(x) for x in sample.tolist()],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return self.store.publish(
            payload,
            category="generated_sample",
            format="json",
            format_version=1,
            producing_component="smoke",
        )

    def publish_comparison_report(self, report: ComparisonReport) -> ArtifactRecord:
        self._guard()
        return self.store.publish(
            report.canonical_bytes(),
            category="report",
            format="json",
            format_version=1,
            producing_component="smoke",
        )

    # -- terminal manifest (LAST; activates the guard) -------------------

    def finalize_manifest(
        self,
        *,
        generator: ManifestGenerator,
        parent_checkpoint_resolver: ParentCheckpointResolver | None = None,
        **generate_kwargs: Any,
    ) -> ArtifactRecord:
        self._guard()
        manifest = generator.generate(**generate_kwargs)
        # Atomic publication-and-sealing (review finding B1): ``seal=True`` writes
        # the seal marker INSIDE the store's publish critical section, under the
        # same AttemptLock that serializes the artifact commit + registry append,
        # with a directory fsync so the marker survives a crash. This makes
        # terminal-manifest publication + sealing one serialized, crash-recoverable
        # operation. Opt-in: ordinary Issue #12 manifests (seal=False) remain
        # publishable for later review reports. Any store bound to the same
        # root+identity — including a freshly reconstructed one — observes the
        # marker and is rejected at every publish/register_* entry. Reads are
        # unaffected. is_sealed is fail-closed on I/O error.
        record = generator.publish(
            manifest, parent_checkpoint_resolver=parent_checkpoint_resolver, seal=True
        )
        self._terminal = True
        return record


class SmokeRunner:
    """Orchestrates the U0/R0/R1 smoke gate over a shared artifact root."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        config_envelope: ResolutionEnvelope,
        repo: Path,
        prepare_run_fn: Any,
        loss_threshold: float,
        k_updates: int,
        n_updates: int,
        tokens_per_update: int,
        allow_dirty: bool = False,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.config_envelope = config_envelope
        self.repo = Path(repo)
        self.prepare_run_func = prepare_run_fn
        self.loss_threshold = float(loss_threshold)
        self.k_updates = int(k_updates)
        self.n_updates = int(n_updates)
        self.tokens_per_update = int(tokens_per_update)
        self.allow_dirty = bool(allow_dirty)
        self.config = config_envelope.config

    # -- shared helpers --------------------------------------------------

    def _new_runtime(self, *, initialized: bool = True) -> tuple[SmokeRuntime, RngManager]:
        cfg = self.config
        seq_len = cfg.training.seq_len or cfg.data.seq_len
        mgr = RngManager.from_config(cfg, component="smoke")
        mgr.initialize()
        model = SmokeModel(cfg.model)
        if initialized:
            model.initialize(mgr.generator)
        opt = SmokeAdamW(lr=cfg.training.lr)
        if initialized:
            opt.initialize(model.parameters)
        sched = SmokeScheduler(lr=cfg.training.lr)
        sched.initialize()
        cursor = SmokeDataCursor.create(batch_size=cfg.training.batch_size, sequence_length=seq_len)
        from expertforge.checkpoints.models import CounterSnapshot

        counters = CounterSnapshot(
            global_update=0,
            completed_microsteps=0,
            accumulation_position=0,
            accepted_samples=0,
            accepted_sequences=0,
            processed_tokens=0,
        )
        rt = SmokeRuntime(
            model_params=model.parameters,
            model_buffers={},
            optimizer_state=opt,
            scheduler=sched,
            cursor=cursor,
            counters=counters,
            rng_manager=mgr,
        )
        return rt, mgr

    def _model(self, rt: SmokeRuntime) -> SmokeModel:
        m = SmokeModel(self.config.model)
        m.parameters = rt.model_params
        m._initialized = True
        return m

    def _train_updates(
        self, rt: SmokeRuntime, *, start: int, stop: int, writer: TelemetryWriter
    ) -> None:
        """Train updates ``[start, stop)`` advancing counters/cursor/scheduler."""
        model = self._model(rt)
        for _ in range(start, stop):
            batch = rt.cursor.next_batch()
            x = batch[:, :-1]
            targets = batch[:, 1:]
            logits, cache = model.forward(x)
            loss = _cross_entropy(logits, targets)
            grads = model.backward(cache, targets)
            rt.optimizer_state.update(model.parameters, grads)
            rt.scheduler.step()
            new_update = rt.counters.global_update + 1
            rt.counters = _bump_counters(rt.counters, new_update, self.tokens_per_update, rt.cursor)
            # Emit a training-step metric with processed_tokens as the progress axis.
            writer.emit_metric(
                component="training",
                observations=[_loss_observation(float(loss))],
                progress=_progress(rt.counters),
            )

    def _validate(
        self,
        rt: SmokeRuntime,
        *,
        writer: TelemetryWriter | None = None,
        is_initial: bool = False,
    ) -> float:
        model = self._model(rt)
        # Held-out validation slice DISJOINT from the consumed training prefix
        # (finding F2). Training consumes the first `total_training_tokens`
        # tokens; validation reads a fixed-length corpus suffix that never
        # overlaps that prefix.
        from expertforge.smoke.data import validation_tokens

        total_training_tokens = self.n_updates * self.tokens_per_update
        val = validation_tokens(total_training_tokens)
        x = val[:-1][None, :]
        targets = val[1:][None, :]
        logits, _ = model.forward(x)
        loss = _cross_entropy(logits, targets)
        if writer is not None:
            writer.emit_event(
                component="evaluation",
                severity="INFO",
                event_name="validation.completed",
                progress=_progress(rt.counters),
                fields=(
                    ("phase", "initial" if is_initial else "final"),
                    ("loss", float(loss)),
                ),
            )
            writer.emit_metric(
                component="evaluation",
                observations=[_loss_observation(float(loss), namespace="validation")],
                progress=_progress(rt.counters),
            )
        return float(loss)

    def _generate(
        self, rt: SmokeRuntime, *, writer: TelemetryWriter | None = None
    ) -> NDArray[np.int64]:
        model = self._model(rt)
        from expertforge.smoke.data import load_corpus_tokens

        prompt = load_corpus_tokens()[:4]
        sample = model.generate(prompt, n_new=8)
        if writer is not None:
            writer.emit_event(
                component="generation",
                severity="INFO",
                event_name="generation.completed",
                progress=_progress(rt.counters),
                fields=(("n_tokens", int(sample.shape[0])),),
            )
        return sample

    # -- U0 -------------------------------------------------------------

    def run_u0(self) -> SmokeAttemptResult:
        identity, provenance, _ = self._prepare_independent()
        store = ArtifactStore(self.artifact_root, identity)
        checkpoint_store = CheckpointStore(store)
        process_context = ProcessContext(rank=0, world_size=1, local_rank=0)
        scope = AttemptScope(identity, store, self.config_envelope, provenance, process_context)

        # F4: publish resolved configuration + provenance FIRST (before any
        # training/checkpoint artifacts).
        config_record = scope.publish_configuration()
        provenance_record = scope.publish_provenance()

        writer = scope.open_telemetry(
            console_enabled=self.config.logging.console_enabled,
            fsync_interval=self.config.logging.fsync_interval_records,
        )
        rt, _ = self._new_runtime()

        # initial validation (before training) for the loss-movement criterion
        initial_loss = self._validate(rt, writer=writer, is_initial=True)

        # 0..K
        self._train_updates(rt, start=0, stop=self.k_updates, writer=writer)
        k_checkpoint = scope.save_checkpoint(checkpoint_store, rt, writer=writer)
        # K..N
        self._train_updates(rt, start=self.k_updates, stop=self.n_updates, writer=writer)
        final_loss = self._validate(rt, writer=writer)
        sample = self._generate(rt, writer=writer)
        final_checkpoint = scope.save_checkpoint(checkpoint_store, rt, writer=writer)

        # generated output
        generated_record = scope.publish_generated_output(sample)

        # close + validate + register telemetry
        tel_path = scope.close_telemetry(outcome="normal")
        telemetry_record = scope.register_telemetry(tel_path)

        # computational state at N
        comp = extract_computational_state(
            model_params=rt.model_params,
            model_buffers=rt.model_buffers,
            optimizer=rt.optimizer_state,
            scheduler=rt.scheduler,
            cursor=rt.cursor,
            counters=rt.counters,
            rng_manager=rt.rng_manager,
            validation_loss=final_loss,
            generated_sample=sample,
        )

        manifest = scope.finalize_manifest(
            generator=ManifestGenerator(store),
            identity=identity,
            finalized_at_utc=datetime.now(UTC),
            classification="smoke_test",
            maturity_stage="Milestone 0",
            research_family="none/not-applicable",
            status="completed",
            issue_number=14,
            configuration_artifact=config_record,
            provenance_artifact=provenance_record,
            dataset=fixtures.dataset_reference(),
            tokenizer=fixtures.tokenizer_reference(),
            model=self._model_identity(),
            training=self._training_budget(),
            evaluation=self._evaluation_summary(final_loss),
            smoke_objective=SMOKE_OBJECTIVE,
            smoke_acceptance_criteria=SMOKE_ACCEPTANCE,
            checkpoint_evidence_required=True,
            generated_output_evidence_required=True,
            telemetry_artifacts=(telemetry_record,),
            checkpoint_artifacts=(k_checkpoint, final_checkpoint),
            generated_output_artifacts=(generated_record,),
        )
        return SmokeAttemptResult(
            identity=identity,
            manifest_record=manifest,
            sealed_scope=scope,
            checkpoints={"k": k_checkpoint, "final": final_checkpoint},
            telemetry_record=telemetry_record,
            generated_record=generated_record,
            computational_state=comp,
            initial_validation_loss=initial_loss,
            final_validation_loss=final_loss,
            generated_sample=sample,
        )

    # -- R0 (interrupted) ----------------------------------------------

    def run_r0(self) -> SmokeAttemptResult:
        identity, provenance, _ = self._prepare_independent()
        store = ArtifactStore(self.artifact_root, identity)
        checkpoint_store = CheckpointStore(store)
        process_context = ProcessContext(rank=0, world_size=1, local_rank=0)
        scope = AttemptScope(identity, store, self.config_envelope, provenance, process_context)

        # F4: publish resolved configuration + provenance FIRST.
        config_record = scope.publish_configuration()
        provenance_record = scope.publish_provenance()

        writer = scope.open_telemetry(
            console_enabled=self.config.logging.console_enabled,
            fsync_interval=self.config.logging.fsync_interval_records,
        )
        rt, _ = self._new_runtime()
        # R0 does NOT validate before training (it terminates at K).
        self._train_updates(rt, start=0, stop=self.k_updates, writer=writer)
        k_checkpoint = scope.save_checkpoint(checkpoint_store, rt, writer=writer)
        # R0 terminates: no validate/generate after K.
        tel_path = scope.close_telemetry(
            outcome="interrupted", diagnostic_code="handled_interruption"
        )
        telemetry_record = scope.register_telemetry(tel_path)
        manifest = scope.finalize_manifest(
            generator=ManifestGenerator(store),
            identity=identity,
            finalized_at_utc=datetime.now(UTC),
            classification="smoke_test",
            maturity_stage="Milestone 0",
            research_family="none/not-applicable",
            status="interrupted",
            outcome_diagnostic="handled_interruption",
            issue_number=14,
            configuration_artifact=config_record,
            provenance_artifact=provenance_record,
            dataset=fixtures.dataset_reference(),
            tokenizer=fixtures.tokenizer_reference(),
            model=self._model_identity(),
            training=self._training_budget(),
            smoke_objective=SMOKE_OBJECTIVE,
            smoke_acceptance_criteria=SMOKE_ACCEPTANCE,
            checkpoint_evidence_required=True,
            generated_output_evidence_required=True,
            telemetry_artifacts=(telemetry_record,),
            checkpoint_artifacts=(k_checkpoint,),
        )
        return SmokeAttemptResult(
            identity=identity,
            manifest_record=manifest,
            sealed_scope=scope,
            checkpoints={"k": k_checkpoint},
            telemetry_record=telemetry_record,
        )

    # -- R1 (resumed) ---------------------------------------------------

    def run_r1(
        self,
        r0: SmokeAttemptResult,
        u0: SmokeAttemptResult,
        *,
        force_mismatch: bool = False,
    ) -> SmokeAttemptResult:
        """Run R1 (resumed from R0) and publish the U0@N vs R1@N comparison report.

        F1: the authoritative load + full restore transaction (RNG last) happens
        BEFORE any state-advancing work — R1 never trains before restoration. The
        comparison report is published as an R1 report artifact AFTER the
        comparison and BEFORE R1's terminal manifest (correction #7).
        """
        r0_identity = r0.identity
        r0_k_checkpoint = r0.checkpoints["k"]
        u0_state = u0.computational_state
        if u0_state is None:
            raise ValueError(
                "U0 result must carry a computational state for the comparison report."
            )
        identity, provenance, _ = self._prepare_resume(
            retained_run_id=r0_identity.run_id,
            parent_attempt_id=r0_identity.attempt_id,
            parent_checkpoint_id=r0_k_checkpoint.artifact_id,
            parent_specification_fingerprint=r0_identity.fingerprint_digest_str(),
        )
        store = ArtifactStore(self.artifact_root, identity)
        checkpoint_store = CheckpointStore(store)
        process_context = ProcessContext(rank=0, world_size=1, local_rank=0)
        scope = AttemptScope(identity, store, self.config_envelope, provenance, process_context)

        # F4: publish resolved configuration + provenance FIRST.
        config_record = scope.publish_configuration()
        provenance_record = scope.publish_provenance()

        writer = scope.open_telemetry(
            console_enabled=self.config.logging.console_enabled,
            fsync_interval=self.config.logging.fsync_interval_records,
        )

        # F1: authoritative load + full restore transaction (RNG last) BEFORE any
        # state-advancing work. The runtime is freshly initialized (setup, not
        # training) so its structural descriptor matches the parent checkpoint;
        # the restore then overwrites it with R0's exact K state. The parent
        # checkpoint is registered in R0's attempt registry, so it is loaded
        # through R0's checkpoint store (reconstructed read-only from R0's
        # identity, not exposed on the result — F8).
        live_rt, _ = self._new_runtime()
        r0_store = ArtifactStore(self.artifact_root, r0_identity)
        r0_checkpoint_store = CheckpointStore(r0_store)
        archive = r0_checkpoint_store.load(r0_k_checkpoint.artifact_id, expected_identity=identity)
        writer.emit_event(
            component="checkpoints",
            severity="INFO",
            event_name="checkpoint.loaded",
            progress=_progress(live_rt.counters),
            fields=(
                ("parent_run_id", r0_identity.run_id),
                ("parent_attempt_id", r0_identity.attempt_id),
                ("parent_checkpoint_id", r0_k_checkpoint.artifact_id),
            ),
        )

        expected_descriptor = _expected_descriptor_with_fingerprint(
            live_rt, identity.fingerprint_digest_str()
        )
        factory = SmokeStateFactory(
            live_rt, rng_factory=lambda: RngManager.from_config(self.config, component="smoke")
        )

        def _load_rng() -> Any:
            return live_rt.rng_manager.capture_state()

        def _consume_rng(bundle: Any) -> None:
            fresh = RngManager.from_config(self.config, component="smoke")
            fresh.restore_state(bundle)
            live_rt.rng_manager = fresh

        from typing import cast

        from expertforge.checkpoints import StateFactory

        transaction = RestoreTransaction(
            archive=archive,
            factory=cast(StateFactory, factory),
            rng_bundle_loader=_load_rng,
            rng_consumer=_consume_rng,
            expected_descriptor=expected_descriptor,
        )
        try:
            transaction.prepare()
            transaction.commit()
        except Exception as e:
            writer.emit_event(
                component="checkpoints",
                severity="CRITICAL",
                event_name="checkpoint.restore_failed",
                diagnostic_code="checkpoint_failure",
                operator_message=f"restore transaction failed: {type(e).__name__}",
                progress=_progress(live_rt.counters),
            )
            raise
        writer.emit_event(
            component="checkpoints",
            severity="INFO",
            event_name="checkpoint.restored",
            progress=_progress(live_rt.counters),
            fields=(
                ("parent_checkpoint_id", r0_k_checkpoint.artifact_id),
                ("global_update", int(live_rt.counters.global_update)),
            ),
        )

        # Now live_rt holds R0's exact K state. Continue K..N.
        if force_mismatch:
            # Fault-injection hook (review finding B2): perturb one parameter
            # AFTER the (correct) restore but BEFORE continuation, so R1@N
            # diverges from U0@N. This drives the canonical failing-comparison
            # path: the R1 report is published with decision="fail" and the CLI
            # exits nonzero. Defaults off; production gate runs are unaffected.
            live_rt.model_params["embed"][0, 0] += 1.0
        self._train_updates(live_rt, start=self.k_updates, stop=self.n_updates, writer=writer)
        final_loss = self._validate(live_rt, writer=writer)
        sample = self._generate(live_rt, writer=writer)
        final_checkpoint = scope.save_checkpoint(checkpoint_store, live_rt, writer=writer)
        generated_record = scope.publish_generated_output(sample)

        comp = extract_computational_state(
            model_params=live_rt.model_params,
            model_buffers=live_rt.model_buffers,
            optimizer=live_rt.optimizer_state,
            scheduler=live_rt.scheduler,
            cursor=live_rt.cursor,
            counters=live_rt.counters,
            rng_manager=live_rt.rng_manager,
            validation_loss=final_loss,
            generated_sample=sample,
        )

        # F3: canonicalize mismatch diagnostics before constructing the report so
        # a failed comparison reports failure rather than crashing on the
        # sorted-diagnostics invariant. Compute the report OBJECT (do not publish
        # yet); ordering requires telemetry closed+registered before the report.
        comparison = compare(u0_state, comp)
        computational_equal = bool(comparison["equal"])
        report = _build_comparison_report(
            u0=u0,
            r1_identity=identity,
            r1_state=comp,
            r1_final_loss=final_loss,
            runner=self,
            comparison=comparison,
        )

        # B2 truthful terminal lifecycle + B3-r2 artifact ordering: close +
        # validate + register telemetry BEFORE publishing the comparison report
        # and the terminal manifest. A failed comparison (e.g. the
        # --force-mismatch fault-injection path) closes telemetry as FAILED and
        # publishes a truthful non-completed terminal manifest, so an injected
        # failure can never be mistaken for a successful canonical run.
        if computational_equal:
            tel_path = scope.close_telemetry(outcome="normal")
            terminal_status = "completed"
            outcome_diag = None
        else:
            tel_path = scope.close_telemetry(outcome="failed", diagnostic_code="checkpoint_failure")
            terminal_status = "failed"
            outcome_diag = "compatibility_mismatch"
        telemetry_record = scope.register_telemetry(tel_path)

        # Publish the comparison report AFTER telemetry is registered, BEFORE the
        # terminal manifest.
        report_record = scope.publish_comparison_report(report)

        manifest = scope.finalize_manifest(
            generator=ManifestGenerator(store),
            parent_checkpoint_resolver=_r0_resolver(r0_store, r0_k_checkpoint),
            identity=identity,
            finalized_at_utc=datetime.now(UTC),
            classification="smoke_test",
            maturity_stage="Milestone 0",
            research_family="none/not-applicable",
            status=terminal_status,
            outcome_diagnostic=outcome_diag,
            issue_number=14,
            configuration_artifact=config_record,
            provenance_artifact=provenance_record,
            dataset=fixtures.dataset_reference(),
            tokenizer=fixtures.tokenizer_reference(),
            model=self._model_identity(),
            training=self._training_budget(),
            evaluation=self._evaluation_summary(final_loss),
            smoke_objective=SMOKE_OBJECTIVE,
            smoke_acceptance_criteria=SMOKE_ACCEPTANCE,
            checkpoint_evidence_required=True,
            generated_output_evidence_required=True,
            telemetry_artifacts=(telemetry_record,),
            checkpoint_artifacts=(final_checkpoint,),
            generated_output_artifacts=(generated_record,),
            review_report_artifacts=(report_record,),
            resume_checkpoint=r0_k_checkpoint,
        )
        return SmokeAttemptResult(
            identity=identity,
            manifest_record=manifest,
            sealed_scope=scope,
            checkpoints={"final": final_checkpoint},
            telemetry_record=telemetry_record,
            generated_record=generated_record,
            computational_state=comp,
            final_validation_loss=final_loss,
            generated_sample=sample,
            comparison_report=report,
        )

    # -- prepare_run wrappers -------------------------------------------

    def _prepare_independent(self) -> tuple[AttemptIdentityRecord, ProvenanceRecord, Path]:
        identity, provenance, path = self.prepare_run_func(
            artifact_root=self.artifact_root,
            config_envelope=self.config_envelope,
            repo=self.repo,
            allow_dirty=self.allow_dirty,
            mode=AllocationMode.INDEPENDENT,
            extra_immutable_inputs=fixtures.fixture_immutable_inputs(),
        )
        return identity, provenance, path

    def _prepare_resume(
        self,
        *,
        retained_run_id: str,
        parent_attempt_id: str,
        parent_checkpoint_id: str,
        parent_specification_fingerprint: str,
    ) -> tuple[AttemptIdentityRecord, ProvenanceRecord, Path]:
        lineage = ResumeLineage(
            parent_run_id=retained_run_id,
            parent_attempt_id=parent_attempt_id,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        identity, provenance, path = self.prepare_run_func(
            artifact_root=self.artifact_root,
            config_envelope=self.config_envelope,
            repo=self.repo,
            allow_dirty=self.allow_dirty,
            mode=AllocationMode.RESUME,
            lineage=lineage,
            retained_run_id=retained_run_id,
            parent_specification_fingerprint=parent_specification_fingerprint,
            extra_immutable_inputs=fixtures.fixture_immutable_inputs(),
        )
        return identity, provenance, path

    # -- manifest sub-models --------------------------------------------

    def _model_identity(self) -> ModelIdentity:
        cfg = self.config.model
        # Exact parameter count derived from the arrays.
        rt, _ = self._new_runtime()
        model = SmokeModel(cfg)
        model.initialize(rt.rng_manager.generator)
        count = model.parameter_count()
        return ModelIdentity(
            architecture=MODEL_ARCHITECTURE,
            dim=cfg.dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            parameter_count=count,
            parameter_count_kind="exact",
        )

    def _training_budget(self) -> TrainingBudget:
        t = self.config.training
        return TrainingBudget(
            token_budget=t.tokens,
            batch_size=t.batch_size,
            seq_len=t.seq_len or self.config.data.seq_len,
            learning_rate=t.lr,
            seed=t.seed,
        )

    def _evaluation_summary(self, final_loss: float) -> EvaluationSummary:
        return EvaluationSummary(
            eval_interval_tokens=self.config.evaluation.eval_interval_tokens,
            final_loss=float(final_loss),
            metrics_recorded=("loss",),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SMOKE_OBJECTIVE = (
    "Prove the Milestone 0 substrate (config, identity, provenance, RNG, telemetry, "
    "artifacts, checkpoints, manifests) works end-to-end: train a toy model, checkpoint "
    "at K, interrupt, and natively resume from the checkpoint with exact computational-state "
    "reproducibility versus an uninterrupted baseline."
)
SMOKE_ACCEPTANCE = (
    "U0@N and R1@N produce identical canonical computational-state digests (parameters, "
    "optimizer slots, scheduler, counters, cursor, isolated RNG/cursor probes, validation "
    "loss, generated sample); final validation loss improves over the initial value by at "
    "least the predeclared threshold; and every required terminal artifact is published."
)


def _cross_entropy(logits: NDArray[np.floating[Any]], targets: NDArray[np.integer[Any]]) -> float:
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=-1, keepdims=True)
    B, T, _ = logits.shape
    idx_b = np.arange(B)[:, None]
    idx_t = np.arange(T)[None, :]
    chosen = p[idx_b, idx_t, targets]
    return float(-np.log(chosen + 1e-12).mean())


def _loss_observation(value: float, *, namespace: str = "training") -> Any:
    from expertforge.telemetry.models import MetricObservation

    return MetricObservation(
        namespace=namespace,
        name="cross_entropy_loss",
        unit="dimensionless",
        aggregation="gauge",
        window="point",
        value_status="finite",
        value=float(value),
    )


def _progress(counters: Any) -> Any:
    from expertforge.telemetry.models import ProgressPosition

    return ProgressPosition(
        step=int(counters.global_update),
        update=int(counters.global_update),
        processed_tokens=int(counters.processed_tokens),
    )


def _bump_counters(
    counters: Any, new_update: int, tokens_per_update: int, cursor: SmokeDataCursor
) -> Any:
    from expertforge.checkpoints.models import CounterSnapshot

    return CounterSnapshot(
        global_update=int(new_update),
        completed_microsteps=int(new_update),
        accumulation_position=0,
        accepted_samples=int(cursor.accepted_samples),
        accepted_sequences=int(cursor.accepted_sequences),
        processed_tokens=int(new_update * tokens_per_update),
    )


def _expected_descriptor_with_fingerprint(runtime: SmokeRuntime, fingerprint: str) -> Any:
    from expertforge.smoke.state import build_expected_descriptor

    return build_expected_descriptor(runtime, specification_fingerprint=fingerprint)


def _build_comparison_report(
    *,
    u0: SmokeAttemptResult,
    r1_identity: AttemptIdentityRecord,
    r1_state: ComputationalState,
    r1_final_loss: float,
    runner: SmokeRunner,
    comparison: dict[str, Any],
) -> ComparisonReport:
    """Build the frozen U0@N vs R1@N comparison report from the comparison dict.

    F3: mismatch diagnostics are canonicalized (filtered to the closed
    ``ComparisonMismatchField`` domain, sorted, deduplicated) BEFORE constructing
    the report, so a failed comparison reports failure rather than crashing on
    the report's sorted-diagnostics invariant.
    """
    u0_state = u0.computational_state
    assert u0_state is not None
    equal = bool(comparison["equal"])
    decision: Any = "pass" if equal else "fail"
    # Canonicalize mismatches: keep only fields in the closed domain, then sort
    # and deduplicate so the ComparisonReport's sorted/unique invariant holds.
    from expertforge.smoke.report import COMPARISON_MISMATCH_DOMAIN

    raw_mismatches = list(comparison["mismatches"])
    canonical = sorted({m for m in raw_mismatches if m in COMPARISON_MISMATCH_DOMAIN})
    final_mismatches: tuple[Any, ...] = () if equal else tuple(canonical)
    import hashlib as _hashlib

    sample_payload = ",".join(
        str(int(x)) for x in np.asarray(u0.generated_sample).reshape(-1).tolist()
    ).encode("utf-8")
    sample_digest = _hashlib.sha256(sample_payload).hexdigest()
    return ComparisonReport(
        u0_run_id=u0.identity.run_id,
        u0_attempt_id=u0.identity.attempt_id,
        r1_run_id=r1_identity.run_id,
        r1_attempt_id=r1_identity.attempt_id,
        specification_fingerprint=r1_identity.fingerprint_digest_str(),
        n_updates=runner.n_updates,
        k_updates=runner.k_updates,
        tokens_per_update=runner.tokens_per_update,
        u0_computational_digest=u0_state.computational_digest,
        r1_computational_digest=r1_state.computational_digest,
        u0_validation_loss=float(u0.final_validation_loss),  # type: ignore[arg-type]
        r1_validation_loss=float(r1_final_loss),
        generated_sample_digest=sample_digest,
        decision=decision,
        mismatches=final_mismatches,
        environment=_capture_environment(),
    )


def _capture_environment() -> ComparisonEnvironment:
    import platform as _platform
    import sys as _sys

    import numpy as _np

    # platform.machine()/processor() may return empty strings on some systems;
    # the ComparisonEnvironment requires non-empty machine/platform. Fall back to
    # a sentinel so the report is always constructible.
    machine = _platform.machine() or "unknown"
    return ComparisonEnvironment(
        python_version=_sys.version.split()[0],
        numpy_version=_np.__version__,
        platform=_platform.system() or "unknown",
        machine=machine,
        processor=_platform.processor(),
        determinism_note=(
            "Exact-byte reproducibility is claimed only for the same locked "
            "ExpertForge/Python/NumPy environment on the supported CPU platform."
        ),
    )


def _r0_resolver(
    r0_store: ArtifactStore, r0_k_checkpoint: ArtifactRecord
) -> ParentCheckpointResolver:
    """Build a parent-checkpoint resolver bound to R0's store + K checkpoint.

    ``ParentCheckpointResolver`` is a ``Callable[[ResumeLineage],
    tuple[ArtifactStore, ArtifactRecord]]`` that returns the parent store and
    the resolved parent checkpoint record (re-resolved through the parent store
    so the registry is authoritative).
    """

    def resolve(lineage: ResumeLineage) -> tuple[ArtifactStore, ArtifactRecord]:
        # Verify the parent identity matches R0's store.
        if (
            r0_store.identity.run_id != lineage.parent_run_id
            or r0_store.identity.attempt_id != lineage.parent_attempt_id
        ):
            raise ManifestBindingError("R0 store identity does not match resume lineage.")
        inspected = r0_store.inspect(lineage.parent_checkpoint_id)
        if isinstance(inspected, ExternalReference):
            raise ManifestBindingError("R0 checkpoint resolved to an external reference.")
        record: ArtifactRecord = inspected
        if record.artifact_id != r0_k_checkpoint.artifact_id:
            raise ManifestBindingError("resolved R0 checkpoint does not match the K checkpoint.")
        return r0_store, record

    return resolve
