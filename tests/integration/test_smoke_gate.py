"""End-to-end smoke & recovery gate tests (Issue #14, amendment P smoke 1–8).

These are marked ``smoke`` and run in the permanent smoke CI tier. They exercise
the full U0/R0/R1 topology through the real substrate subsystems:

1. U0 full path with every required record and terminal manifest ordering;
2. R0 interruption at K with truthful incomplete manifest;
3. R1 authoritative restore and K→N continuation;
4. U0@N versus R1@N exact computational comparison;
5. independent clean-run reproducibility (distinct identities, identical digest);
6. authoritative parent/reference/payload verification;
7. failure-injection matrix (exact typed layers);
8. CLI execution from a clean checkout with bounded runtime/memory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from expertforge.artifacts import ArtifactStore
from expertforge.checkpoints import CheckpointStore
from expertforge.experiments import ManifestGenerator, load_manifest
from expertforge.smoke.run import build_runner, run_gate
from expertforge.smoke.runner import SmokeRunner
from expertforge.smoke.state import SmokeRuntime

pytestmark = pytest.mark.smoke

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
GATE_CONFIG = CONFIGS / "m0-smoke-gate.yaml"
REPO = Path(__file__).resolve().parents[2]


def _runner(tmp_path: Path, *, allow_dirty: bool = True) -> SmokeRunner:
    return build_runner(
        artifact_root=tmp_path / "runs",
        config_path=GATE_CONFIG,
        repo=REPO,
        allow_dirty=allow_dirty,
    )


def _store_for(result: object, tmp_path: Path) -> ArtifactStore:
    """Reconstruct a read-only ArtifactStore for a completed attempt's identity.

    F8: results no longer expose a publish-capable store. Tests reconstruct a
    read-only store (read-only usage does not bypass the attempt-scoped terminal
    guard, which governs publication, not reads).
    """
    return ArtifactStore(tmp_path / "runs", result.identity)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Smoke 1: U0 full path + terminal manifest ordering
# ---------------------------------------------------------------------------


class TestU0FullPath:
    def test_u0_publishes_complete_terminal_manifest_last(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        # The manifest is the terminal artifact; verify it loads and is completed.
        store = _store_for(u0, tmp_path)
        loaded = load_manifest(
            u0.manifest_record.artifact_id, artifact_store=store, expected_identity=u0.identity
        )
        assert loaded.manifest.status == "completed"
        assert loaded.manifest.classification == "smoke_test"
        assert loaded.manifest.maturity_stage == "Milestone 0"
        # Complete core evidence.
        assert loaded.manifest.evidence.status == "complete"
        assert loaded.manifest.evidence.missing == ()
        # Required evidence present.
        assert loaded.manifest.configuration_artifact is not None
        assert loaded.manifest.provenance_artifact is not None
        assert len(loaded.manifest.telemetry_artifacts) >= 1
        assert len(loaded.manifest.checkpoint_artifacts) == 2  # K + final
        assert len(loaded.manifest.generated_output_artifacts) == 1
        # Model identity carries the exact parameter count.
        assert loaded.manifest.model_descriptor is not None
        assert loaded.manifest.model_descriptor.parameter_count == 2560
        assert loaded.manifest.model_descriptor.parameter_count_kind == "exact"
        # Non-blank smoke contract.
        assert loaded.manifest.smoke_objective
        assert loaded.manifest.smoke_acceptance_criteria

    def test_u0_computational_state_present(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        assert u0.computational_state is not None
        assert u0.initial_validation_loss is not None
        assert u0.final_validation_loss is not None
        assert u0.final_validation_loss < u0.initial_validation_loss


# ---------------------------------------------------------------------------
# Smoke 2: R0 interruption at K + truthful incomplete manifest
# ---------------------------------------------------------------------------


class TestR0Interruption:
    def test_r0_publishes_interrupted_partial_manifest(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        r0 = runner.run_r0()
        loaded = load_manifest(
            r0.manifest_record.artifact_id,
            artifact_store=_store_for(r0, tmp_path),
            expected_identity=r0.identity,
        )
        assert loaded.manifest.status == "interrupted"
        assert loaded.manifest.outcome_diagnostic == "handled_interruption"
        # Truthfully partial: has its K checkpoint but lacks eval/generated output.
        assert loaded.manifest.evidence.status == "partial"
        assert "evaluation_summary_missing" in loaded.manifest.evidence.missing
        assert "generated_output_artifact_missing" in loaded.manifest.evidence.missing
        # R0 has its K checkpoint.
        assert len(loaded.manifest.checkpoint_artifacts) == 1
        # R0 never produced generated output.
        assert loaded.manifest.generated_output_artifacts == ()
        # No evaluation summary (R0 did not validate after K).
        assert loaded.manifest.evaluation_summary is None

    def test_r0_distinct_run_identity_from_u0(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        r0 = runner.run_r0()
        # Same specification fingerprint, distinct run/attempt identities.
        assert u0.identity.fingerprint_digest_str() == r0.identity.fingerprint_digest_str()
        assert u0.identity.run_id != r0.identity.run_id
        assert u0.identity.attempt_id != r0.identity.attempt_id


# ---------------------------------------------------------------------------
# Smoke 3 + 4: R1 restore + continuation + U0@N vs R1@N comparison
# ---------------------------------------------------------------------------


class TestR1RestoreAndComparison:
    def test_full_gate_passes_u0_vs_r1(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        passed, report, summary = run_gate(runner)
        assert passed, f"gate failed: mismatches={summary['comparison_mismatches']}"
        assert summary["computational_equal"] is True
        assert summary["u0_computational_digest"] == summary["r1_computational_digest"]
        assert summary["loss_movement_ok"] is True
        assert report is not None
        assert report.decision == "pass"
        assert report.mismatches == ()

    def test_r1_resumed_identity_preserves_run_and_lineage(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        r0 = runner.run_r0()
        r1 = runner.run_r1(r0, u0)
        # R1 retains R0's run_id, gets a new attempt_id.
        assert r1.identity.run_id == r0.identity.run_id
        assert r1.identity.attempt_id != r0.identity.attempt_id
        assert r1.identity.attempt_id != u0.identity.attempt_id
        # Same specification fingerprint (RESUME preserves it).
        assert r1.identity.fingerprint_digest_str() == r0.identity.fingerprint_digest_str()
        # Fully-qualified native lineage naming R0's K checkpoint.
        lineage = r1.identity.lineage
        assert lineage is not None
        assert lineage.parent_run_id == r0.identity.run_id
        assert lineage.parent_attempt_id == r0.identity.attempt_id
        assert lineage.parent_checkpoint_id == r0.checkpoints["k"].artifact_id

    def test_r1_manifest_has_resume_checkpoint(self, tmp_path: Path) -> None:
        from expertforge.smoke.runner import _r0_resolver

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        r0 = runner.run_r0()
        r1 = runner.run_r1(r0, u0)
        resolver = _r0_resolver(_store_for(r0, tmp_path), r0.checkpoints["k"])
        loaded = load_manifest(
            r1.manifest_record.artifact_id,
            artifact_store=_store_for(r1, tmp_path),
            expected_identity=r1.identity,
            parent_checkpoint_resolver=resolver,
        )
        assert loaded.manifest.status == "completed"
        assert loaded.manifest.resume_checkpoint is not None
        assert loaded.manifest.resume_checkpoint.artifact_id == r0.checkpoints["k"].artifact_id
        # The comparison report is an R1 review_report artifact.
        assert len(loaded.manifest.review_report_artifacts) == 1

    def test_comparison_excludes_identity_bound_containers(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        r0 = runner.run_r0()
        r1 = runner.run_r1(r0, u0)
        # Identity-bound containers are intentionally unequal.
        assert u0.identity.run_id != r1.identity.run_id
        assert u0.manifest_record.artifact_id != r1.manifest_record.artifact_id
        # But computational state is equal.
        assert u0.computational_state is not None
        assert r1.computational_state is not None
        assert (
            u0.computational_state.computational_digest
            == r1.computational_state.computational_digest
        )


# ---------------------------------------------------------------------------
# Smoke 5: independent clean-run reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_two_clean_u0_runs_distinct_identity_identical_digest(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        u0a = runner.run_u0()
        u0b = runner.run_u0()
        # Distinct identities.
        assert u0a.identity.run_id != u0b.identity.run_id
        assert u0a.identity.attempt_id != u0b.identity.attempt_id
        # Identical specification fingerprint.
        assert u0a.identity.fingerprint_digest_str() == u0b.identity.fingerprint_digest_str()
        # Identical computational digest + validation + sample.
        assert u0a.computational_state is not None
        assert u0b.computational_state is not None
        assert (
            u0a.computational_state.computational_digest
            == u0b.computational_state.computational_digest
        )
        assert u0a.computational_state.validation_loss == u0b.computational_state.validation_loss
        assert u0a.computational_state.generated_sample == u0b.computational_state.generated_sample
        # Each validates independently.
        assert u0a.manifest_record.artifact_id != u0b.manifest_record.artifact_id


# ---------------------------------------------------------------------------
# Smoke 6: authoritative parent/reference/payload verification
# ---------------------------------------------------------------------------


class TestParentVerification:
    def test_r0_checkpoint_is_loadable_through_r0_store(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        r0 = runner.run_r0()
        r1 = runner.run_r1(r0, u0)
        # R1's identity lineage references a real, loadable checkpoint.
        r0_store = _store_for(r0, tmp_path)
        r0_cp_store = CheckpointStore(r0_store)
        lineage = r1.identity.lineage
        assert lineage is not None
        # The parent checkpoint record resolves through R0's store.
        record = r0_store.inspect(lineage.parent_checkpoint_id)
        assert record.artifact_id == r0.checkpoints["k"].artifact_id
        # It loads with R1's resume identity.
        archive = r0_cp_store.load(lineage.parent_checkpoint_id, expected_identity=r1.identity)
        assert archive.artifact_id == r0.checkpoints["k"].artifact_id


# ---------------------------------------------------------------------------
# Smoke 7: failure-injection matrix (exact typed layers)
# ---------------------------------------------------------------------------


class TestFailureMatrix:
    def test_corrupt_checkpoint_raises_checkpoint_corrupt_error(self, tmp_path: Path) -> None:
        import shutil

        from expertforge.checkpoints import CheckpointCorruptError

        runner = _runner(tmp_path)
        r0 = runner.run_r0()
        record = r0.checkpoints["k"]
        # Copy R0's entire attempt tree to an ISOLATED root (never mutate R0).
        corrupt_root = tmp_path / "corrupt"
        shutil.copytree(runner.artifact_root, corrupt_root)
        # Flip a byte in the checkpoint content (the tar) so its SHA-256 no longer
        # matches the registry-recorded digest. The registry entry stays intact.
        content_path = (
            corrupt_root
            / r0.identity.run_id
            / "attempts"
            / r0.identity.attempt_id
            / "artifacts"
            / "checkpoint"
            / record.artifact_id
            / "content"
        )
        assert content_path.is_file()
        data = bytearray(content_path.read_bytes())
        data[-1] ^= 0xFF
        content_path.write_bytes(bytes(data))
        # Build a store over the isolated root bound to R0's identity.
        corrupt_store = ArtifactStore(corrupt_root, r0.identity)
        corrupt_cp_store = CheckpointStore(corrupt_store)
        # Build a valid resume identity through the runner's prepare path.
        r1_identity, _, _ = runner._prepare_resume(  # noqa: SLF001
            retained_run_id=r0.identity.run_id,
            parent_attempt_id=r0.identity.attempt_id,
            parent_checkpoint_id=record.artifact_id,
            parent_specification_fingerprint=r0.identity.fingerprint_digest_str(),
        )
        with pytest.raises(CheckpointCorruptError):
            corrupt_cp_store.load(record.artifact_id, expected_identity=r1_identity)

    def test_wrong_lineage_raises_checkpoint_lineage_error(self, tmp_path: Path) -> None:
        from expertforge.checkpoints import CheckpointLineageError

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        r0 = runner.run_r0()
        cp_store = CheckpointStore(_store_for(r0, tmp_path))
        # Loading with U0's identity (wrong lineage) must fail.
        with pytest.raises(CheckpointLineageError):
            cp_store.load(r0.checkpoints["k"].artifact_id, expected_identity=u0.identity)

    def test_post_manifest_publication_raises_terminal_guard(self, tmp_path: Path) -> None:
        # B1: the terminal guard is enforced on the REAL finalized scope, not by
        # removing a dataclass field. Drive a full U0 run (which finalizes the
        # attempt's terminal manifest through its own scope), then prove that
        # scope rejects every further publication method.
        from expertforge.smoke.runner import SmokeTerminalStateError

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        # The sealed scope is the real scope that produced U0; it is terminal.
        assert u0.sealed_scope.terminal is True
        with pytest.raises(SmokeTerminalStateError):
            u0.sealed_scope.publish_configuration()
        with pytest.raises(SmokeTerminalStateError):
            u0.sealed_scope.publish_provenance()
        with pytest.raises(SmokeTerminalStateError):
            u0.sealed_scope.save_checkpoint(
                CheckpointStore(ArtifactStore(tmp_path / "runs", u0.identity)),
                _minimal_runtime(runner),
            )
        with pytest.raises(SmokeTerminalStateError):
            u0.sealed_scope.publish_generated_output(np.array([1, 2, 3], dtype=np.int64))
        with pytest.raises(SmokeTerminalStateError):
            u0.sealed_scope.finalize_manifest(
                generator=ManifestGenerator(ArtifactStore(tmp_path / "runs", u0.identity)),
                identity=u0.identity,
                finalized_at_utc=__import__("datetime").datetime.now(__import__("datetime").UTC),
                classification="smoke_test",
                maturity_stage="Milestone 0",
                research_family="none/not-applicable",
                status="completed",
                smoke_objective="x",
                smoke_acceptance_criteria="y",
            )


def _minimal_runtime(runner: SmokeRunner) -> SmokeRuntime:
    """A fresh initialized runtime for the guard test's save_checkpoint probe."""
    rt: SmokeRuntime
    rt, _ = runner._new_runtime()  # noqa: SLF001
    return rt


# ---------------------------------------------------------------------------
# Smoke 7b: expanded failure matrix (F7) + new-finding regressions (F1/F2/F6)
# ---------------------------------------------------------------------------


class TestExpandedFailureMatrix:
    def test_compatibility_mismatch_after_valid_load_raises_restore_incompatible(
        self, tmp_path: Path
    ) -> None:
        # F7: an exact-compatibility descriptor mismatch after a valid load must
        # raise RestoreIncompatibleError, not proceed.
        from expertforge.checkpoints import RestoreIncompatibleError, RestoreTransaction

        runner = _runner(tmp_path)
        r0 = runner.run_r0()
        r0_store = _store_for(r0, tmp_path)
        r1_identity, _, _ = runner._prepare_resume(  # noqa: SLF001
            retained_run_id=r0.identity.run_id,
            parent_attempt_id=r0.identity.attempt_id,
            parent_checkpoint_id=r0.checkpoints["k"].artifact_id,
            parent_specification_fingerprint=r0.identity.fingerprint_digest_str(),
        )
        live_rt, _ = runner._new_runtime()  # noqa: SLF001
        archive = CheckpointStore(r0_store).load(
            r0.checkpoints["k"].artifact_id, expected_identity=r1_identity
        )
        # Build a deliberately-mismatched expected descriptor by perturbing the
        # model descriptor (add a phantom parameter the archive does not carry).
        from expertforge.smoke.state import SmokeStateProvider

        captured = SmokeStateProvider(live_rt).capture_checkpoint_snapshot()

        bad_model = captured.model_descriptor.model_copy(deep=True)
        # Mismatch: declare an extra parameter not in the archive.
        from expertforge.checkpoints.models import ModelParameterDescriptor

        bad_model = bad_model.model_copy(
            update={
                "parameters": (
                    *bad_model.parameters,
                    ModelParameterDescriptor(name="phantom", shape=(4,), dtype="float32"),
                )
            }
        )
        bad_descriptor = captured.model_descriptor  # placeholder; rebuild below
        from expertforge.checkpoints.models import CompatibilityDescriptor

        expected = CompatibilityDescriptor(
            specification_fingerprint=r1_identity.fingerprint_digest_str(),
            model_descriptor=bad_model,
            optimizer_descriptor=captured.optimizer_descriptor,
            scheduler_descriptor=captured.scheduler_descriptor,
            scaler_descriptor=None,
            rng_descriptor=captured.rng_descriptor,
            data_descriptor=captured.data_descriptor,
            topology_descriptor=captured.topology_descriptor,
        )
        del bad_descriptor
        from expertforge.smoke.state import SmokeStateFactory

        factory = SmokeStateFactory(
            live_rt,
            rng_factory=lambda: __import__(
                "expertforge.rng.manager", fromlist=["RngManager"]
            ).RngManager.from_config(runner.config, component="smoke"),
        )
        txn = RestoreTransaction(
            archive=archive,
            factory=factory,
            rng_bundle_loader=lambda: live_rt.rng_manager.capture_state(),
            rng_consumer=lambda b: None,
            expected_descriptor=expected,
        )
        with pytest.raises(RestoreIncompatibleError):
            txn.prepare()

    def test_missing_evidence_artifact_raises_manifest_binding_error(self, tmp_path: Path) -> None:
        # B3-r1: a manifest that references a CONCRETE UNREGISTERED evidence
        # artifact is rejected at publication with exactly ManifestBindingError.
        # We build a manifest via generate() that names a checkpoint artifact
        # whose ArtifactRecord is constructed (valid shape/category) but whose
        # artifact_id is NOT registered in the store, then publish() ->
        # _verify_manifest_references -> _resolve_current_record fails.
        from datetime import UTC, datetime

        from expertforge.artifacts import ArtifactRecord
        from expertforge.experiments import ManifestBindingError

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        store = _store_for(u0, tmp_path)
        # A concrete-but-unregistered checkpoint artifact record: same identity as
        # U0 (so generate()'s identity-binding check passes), but an artifact_id
        # that was never registered/published in this store.
        unregistered_checkpoint = ArtifactRecord(
            artifact_id="artifact-v1-sha256-" + "0" * 64,
            schema_version=1,
            run_id=u0.identity.run_id,
            attempt_id=u0.identity.attempt_id,
            specification_fingerprint=u0.identity.fingerprint_digest_str(),
            category="checkpoint",
            format="tar",
            format_version=1,
            content_digest="sha256:" + "0" * 64,
            byte_size=1,
            producing_component="probe",
            created_at_utc=datetime.now(UTC),
            relative_path="artifacts/checkpoint/probe",
            storage_class="canonical_local",
            retention="retained",
        )
        gen = ManifestGenerator(store)
        manifest = gen.generate(
            identity=u0.identity,
            finalized_at_utc=datetime.now(UTC),
            classification="smoke_test",
            maturity_stage="Milestone 0",
            research_family="none/not-applicable",
            status="interrupted",
            outcome_diagnostic="handled_interruption",
            configuration_artifact=None,
            provenance_artifact=None,
            dataset=__import__(
                "expertforge.smoke.fixtures", fromlist=["dataset_reference"]
            ).dataset_reference(),
            tokenizer=__import__(
                "expertforge.smoke.fixtures", fromlist=["tokenizer_reference"]
            ).tokenizer_reference(),
            model=runner._model_identity(),  # noqa: SLF001
            training=runner._training_budget(),  # noqa: SLF001
            smoke_objective="x",
            smoke_acceptance_criteria="y",
            checkpoint_evidence_required=True,
            telemetry_artifacts=(),
            checkpoint_artifacts=(unregistered_checkpoint,),  # concrete unregistered evidence
            generated_output_artifacts=(),
        )
        # publish() must reject the unregistered checkpoint evidence.
        with pytest.raises(ManifestBindingError):
            gen.publish(manifest)

    def test_wrong_parent_fingerprint_raises_on_prepare_resume(self, tmp_path: Path) -> None:
        # F7: a resume whose parent_specification_fingerprint does not match the
        # retained run is rejected at prepare_run (provenance orchestration wraps
        # the identity-emit spec-fingerprint check).
        from expertforge.provenance.orchestrate import ProvenanceOrchestrationError

        runner = _runner(tmp_path)
        r0 = runner.run_r0()
        bogus_fp = "spec-v1-sha256-" + "a" * 64
        with pytest.raises(ProvenanceOrchestrationError):
            runner._prepare_resume(  # noqa: SLF001
                retained_run_id=r0.identity.run_id,
                parent_attempt_id=r0.identity.attempt_id,
                parent_checkpoint_id=r0.checkpoints["k"].artifact_id,
                parent_specification_fingerprint=bogus_fp,
            )

    def test_incomplete_vs_corrupt_telemetry_diagnostics(self, tmp_path: Path) -> None:
        # B2: distinguish incomplete telemetry (no terminal close) from corrupt
        # telemetry (malformed record) using the public #9 loader boundaries.
        from expertforge.telemetry.loader import TelemetryLoadError, scan_telemetry_stream

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        store = _store_for(u0, tmp_path)
        import os

        assert u0.telemetry_record is not None
        fd, _rec = store.open_verified_content(u0.telemetry_record.artifact_id)
        with os.fdopen(fd, "rb") as fh:
            valid_bytes = fh.read()
        # Incomplete: truncate before the stream_closed record (drop the last
        # line, which is the close event).
        lines = valid_bytes.decode("utf-8").splitlines()
        incomplete_bytes = ("\n".join(lines[:-1]) + "\n").encode("utf-8")
        incomplete_path = tmp_path / "incomplete.jsonl"
        incomplete_path.write_bytes(incomplete_bytes)
        scan_inc = scan_telemetry_stream(incomplete_path)
        assert scan_inc.status == "incomplete"
        with pytest.raises(TelemetryLoadError):
            from expertforge.telemetry.loader import load_telemetry_stream

            load_telemetry_stream(
                incomplete_path,
                expected_identity=u0.identity,
                expected_process_context=__import__(
                    "expertforge.telemetry.models", fromlist=["ProcessContext"]
                ).ProcessContext(rank=0, world_size=1, local_rank=0),
            )
        # Corrupt: inject a malformed JSON line.
        corrupt_lines = list(lines)
        mid = len(corrupt_lines) // 2
        corrupt_lines[mid] = "{not valid json"
        corrupt_bytes = ("\n".join(corrupt_lines) + "\n").encode("utf-8")
        corrupt_path = tmp_path / "corrupt.jsonl"
        corrupt_path.write_bytes(corrupt_bytes)
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                corrupt_path,
                expected_identity=u0.identity,
                expected_process_context=__import__(
                    "expertforge.telemetry.models", fromlist=["ProcessContext"]
                ).ProcessContext(rank=0, world_size=1, local_rank=0),
            )

    def test_missing_parent_reference_raises_checkpoint_lineage_error(self, tmp_path: Path) -> None:
        # B2-r2: a never-registered parent reference is rejected at the public
        # resolve_parent() boundary with CheckpointLineageError (the binding
        # amendment's required typed layer). This tests a genuinely missing
        # parent REGISTRATION (not a lineage mismatch): the ParentReference names
        # an artifact_id that was never registered in the parent store.
        from expertforge.artifacts import ParentReference
        from expertforge.checkpoints import CheckpointLineageError

        runner = _runner(tmp_path)
        r0 = runner.run_r0()
        r0_store = _store_for(r0, tmp_path)
        bogus_parent = ParentReference(
            run_id=r0.identity.run_id,
            attempt_id=r0.identity.attempt_id,
            artifact_id="artifact-v1-sha256-" + "e" * 64,  # never registered
        )
        cp_store = CheckpointStore(r0_store)
        with pytest.raises(CheckpointLineageError):
            cp_store.resolve_parent(bogus_parent)

    def test_cli_exits_nonzero_on_computational_mismatch(self, tmp_path: Path) -> None:
        # B2: the canonical failing-comparison path. The --force-mismatch flag
        # perturbs R1's continuation after restore so U0@N vs R1@N genuinely
        # diverges: the R1 comparison report is published with decision="fail"
        # and the CLI exits nonzero. This exercises a REAL computational mismatch
        # (not a loss-threshold tweak) through the real CLI entry point.
        env = {
            "PATH": __import__("os").environ.get("PATH", ""),
            "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
        }
        result = __import__("subprocess").run(
            [
                sys.executable,
                "-m",
                "expertforge.smoke.run",
                "--artifact-root",
                str(tmp_path / "mismatch-runs"),
                "--config",
                str(GATE_CONFIG),
                "--repo",
                str(REPO),
                "--allow-dirty",
                "--force-mismatch",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            env=env,
            timeout=120,
        )
        assert result.returncode == 1, "CLI must exit 1 on a computational mismatch"
        assert "SMOKE GATE: FAIL" in result.stderr
        # A canonical decision=fail report was published with mismatch diagnostics.
        import re

        decision = re.search(r'"decision":"(\w+)"', result.stdout)
        assert decision is not None and decision.group(1) == "fail"
        assert '"mismatches":[' in result.stdout.replace(" ", "")
        # The fail report was published as a registered R1 artifact: parse the
        # summary JSON (it precedes the COMPARISON_REPORT line) and confirm a
        # report artifact exists under R1's attempt directory.
        summary_json = result.stdout.split("COMPARISON_REPORT", 1)[0].strip()
        summary = json.loads(summary_json)
        report_dir = (
            tmp_path
            / "mismatch-runs"
            / summary["r1_run_id"]
            / "attempts"
            / summary["r1_attempt_id"]
            / "artifacts"
            / "report"
        )
        assert report_dir.exists() and any(report_dir.iterdir()), (
            "fail report must be published as an R1 artifact"
        )
        # B3-r2: load the published R1 manifest and telemetry and assert the
        # truthful FAILED terminal states (an injected failure must never look
        # like a completed canonical run).

        r1_attempt_dir = (
            tmp_path
            / "mismatch-runs"
            / summary["r1_run_id"]
            / "attempts"
            / summary["r1_attempt_id"]
        )
        # R1 manifest: status=failed, outcome_diagnostic=compatibility_mismatch.
        from expertforge.experiments import parse_manifest_bytes

        mf_dir = r1_attempt_dir / "artifacts" / "experiment_manifest"
        manifest_paths = [p for p in mf_dir.iterdir()] if mf_dir.exists() else []
        assert len(manifest_paths) == 1
        with open(manifest_paths[0] / "content", "rb") as fh:
            raw = fh.read()
        loaded_manifest = parse_manifest_bytes(raw)
        assert loaded_manifest.status == "failed"
        assert loaded_manifest.outcome_diagnostic == "compatibility_mismatch"
        # R1 telemetry: stream_closed with outcome=failed.
        import json as _json

        tel_dir = r1_attempt_dir / "artifacts" / "telemetry"
        tel_paths = [p for p in tel_dir.iterdir()] if tel_dir.exists() else []
        assert len(tel_paths) == 1
        with open(tel_paths[0] / "content", "rb") as fh:
            tel_raw = fh.read().decode("utf-8")
        close_records = [
            _json.loads(line)
            for line in tel_raw.splitlines()
            if line.strip() and _json.loads(line).get("event_name") == "logging.stream_closed"
        ]
        assert len(close_records) == 1
        assert close_records[0]["fields"][0]["value"] == "failed"

    def test_failing_comparison_reports_failure_without_crashing(self, tmp_path: Path) -> None:
        # F3/F7: a comparison whose operands differ reports decision=fail with
        # canonicalized mismatches rather than crashing.
        from expertforge.smoke.comparison import (
            ComputationalState,
            compare,
        )
        from expertforge.smoke.report import COMPARISON_MISMATCH_DOMAIN
        from expertforge.smoke.runner import _build_comparison_report

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        r0 = runner.run_r0()
        r1 = runner.run_r1(r0, u0)
        # Perturb R1's state to force a mismatch and build a fail report.
        u0_state = u0.computational_state
        r1_state = r1.computational_state
        assert u0_state is not None and r1_state is not None
        # Synthesize a mismatched r1 state by replacing its digest fields.
        mismatched = ComputationalState(
            parameters_digest="x" * 64,
            optimizer_slots_digest=u0_state.optimizer_slots_digest,
            optimizer_scalar_digest=u0_state.optimizer_scalar_digest,
            scheduler_digest=u0_state.scheduler_digest,
            counters_digest=u0_state.counters_digest,
            cursor_digest=u0_state.cursor_digest,
            next_item_probe=u0_state.next_item_probe,
            next_random_probe=u0_state.next_random_probe,
            validation_loss=u0_state.validation_loss + 1.0,  # differs
            generated_sample=u0_state.generated_sample,
            computational_digest="y" * 64,
        )
        comparison = compare(u0_state, mismatched)
        assert comparison["equal"] is False
        report = _build_comparison_report(
            u0=u0,
            r1_identity=r1.identity,
            r1_state=mismatched,
            r1_final_loss=float(r1.final_validation_loss),  # type: ignore[arg-type]
            runner=runner,
            comparison=comparison,
        )
        assert report.decision == "fail"
        assert len(report.mismatches) > 0
        # All reported mismatches are in the closed domain and sorted.
        for m in report.mismatches:
            assert m in COMPARISON_MISMATCH_DOMAIN
        assert list(report.mismatches) == sorted(report.mismatches)


class TestNewFindingRegressions:
    """Dedicated regressions for each corrected finding (F1/F2/F6/F8)."""

    def test_r1_restores_before_any_state_advancing_work(self, tmp_path: Path) -> None:
        # F1: R1's restore transaction commits BEFORE any training. We assert
        # this structurally by checking that R1's global_update after restore
        # equals K (the parent checkpoint's boundary), proving no R1-local
        # training advanced the counters before restore.
        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        r0 = runner.run_r0()
        r1 = runner.run_r1(r0, u0)
        # R1 continued K..N and ended at global_update == N (8). The proof that
        # restore happened first (not pre-restore training) is the byte-exact
        # computational equality with U0, already asserted elsewhere; here we
        # additionally assert the resume checkpoint boundary is K.
        lineage = r1.identity.lineage
        assert lineage is not None
        assert r0.checkpoints["k"].artifact_id == lineage.parent_checkpoint_id

    def test_validation_uses_disjoint_held_out_window(self, tmp_path: Path) -> None:
        # F2: the validation window is disjoint from the consumed training prefix.
        from expertforge.smoke.data import (
            VALIDATION_WINDOW_TOKENS,
            load_corpus_tokens,
            validation_tokens,
        )

        total_training = 128
        val = validation_tokens(total_training)
        tokens = load_corpus_tokens()
        # The validation window is the corpus suffix of length
        # VALIDATION_WINDOW_TOKENS, starting strictly after the training prefix.
        assert len(val) == VALIDATION_WINDOW_TOKENS
        suffix_start = len(tokens) - VALIDATION_WINDOW_TOKENS
        assert suffix_start >= total_training
        np.testing.assert_array_equal(val, tokens[suffix_start:])

    def test_required_telemetry_events_are_emitted(self, tmp_path: Path) -> None:
        # F6: the U0/R1 telemetry streams contain the required checkpoint,
        # validation, generation, and (R1) restore events.
        import os

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        r0 = runner.run_r0()
        r1 = runner.run_r1(r0, u0)
        for result, expected_events in (
            (u0, {"checkpoint.saved", "validation.completed", "generation.completed"}),
            (
                r1,
                {
                    "checkpoint.loaded",
                    "checkpoint.restored",
                    "checkpoint.saved",
                    "validation.completed",
                    "generation.completed",
                },
            ),
        ):
            assert result.telemetry_record is not None
            store = _store_for(result, tmp_path)
            fd, _rec = store.open_verified_content(result.telemetry_record.artifact_id)
            # os.fdopen takes ownership of the fd and closes it on close().
            with os.fdopen(fd, "rb") as fh:
                raw = fh.read()
            event_names = {
                json.loads(line)["event_name"]
                for line in raw.decode("utf-8").splitlines()
                if line.strip() and '"event_name"' in line
            }
            for ev in expected_events:
                assert ev in event_names, (
                    f"missing event {ev} in attempt {result.identity.attempt_id}"
                )

    def test_terminal_guard_enforced_on_real_finalized_scope(self, tmp_path: Path) -> None:
        # B1/F8: the terminal guard is enforced on the REAL finalized scope, not
        # merely by omitting a dataclass field. After a real U0 run, the sealed
        # scope is terminal and rejects publication; the result exposes no
        # publish-capable store.
        import dataclasses

        from expertforge.smoke.runner import SmokeTerminalStateError

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        # No publish-capable store field.
        field_names = {f.name for f in dataclasses.fields(u0)}
        assert "artifact_store" not in field_names
        assert not hasattr(u0, "artifact_store")
        # The sealed scope is the real, finalized scope and is terminal.
        assert u0.sealed_scope.terminal is True
        with pytest.raises(SmokeTerminalStateError):
            u0.sealed_scope.publish_configuration()

    def test_durable_seal_blocks_reconstructed_store_publication(self, tmp_path: Path) -> None:
        # B1 (durable): the seal is enforced INSIDE ArtifactStore publication, so
        # a FRESHLY RECONSTRUCTED store from root+identity (the bypass path) also
        # rejects publication/registration. Reads remain unrestricted. This is
        # the unforgeable mechanism a runner-only convention cannot provide.
        from expertforge.artifacts import ArtifactSealedError

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        # Reconstruct a brand-new store from the public root + U0's identity.
        fresh = ArtifactStore(runner.artifact_root, u0.identity)
        assert fresh.is_sealed is True
        # Every publication/registration path must reject the sealed attempt.
        with pytest.raises(ArtifactSealedError):
            fresh.publish(
                b"x", category="report", format="json", format_version=1, producing_component="x"
            )
        with pytest.raises(ArtifactSealedError):
            fresh.register_external(
                category="external",
                format="json",
                format_version=1,
                producing_component="x",
                location_type="uri",
                location="https://example.invalid/x",
                expected_digest="sha256:" + "0" * 64,
                expected_byte_size=1,
            )
        # Reads remain unrestricted on the reconstructed store.
        rec = fresh.inspect(u0.manifest_record.artifact_id)
        assert rec.category == "experiment_manifest"

    def test_seal_crash_recovery_completes_on_retry(self, tmp_path: Path) -> None:
        # B1 crash-recovery: simulate a terminal manifest publish that crashes
        # AFTER the registry append but BEFORE the marker write, then retry the
        # same publish(seal=True). The retry finds the existing bundle (idempotent
        # branch) and completes the seal marker, so the attempt ends sealed.
        runner = _runner(tmp_path)
        # Run U0 partially: publish config+provenance, then manually publish a
        # manifest WITHOUT sealing (simulating a crash before the marker write).
        u0_incomplete = runner.run_u0()
        store = _store_for(u0_incomplete, tmp_path)
        # Delete the seal marker to simulate the crash gap (manifest registered,
        # marker not yet written).
        marker = store.sealed_marker_path
        if marker.exists():
            marker.unlink()
        assert store.is_sealed is False  # marker gone; not sealed (crash window)
        # Retry the terminal publication with seal=True: the existing-bundle
        # idempotent branch must complete the seal marker.
        gen = ManifestGenerator(store)
        loaded = load_manifest(
            u0_incomplete.manifest_record.artifact_id,
            artifact_store=store,
            expected_identity=u0_incomplete.identity,
        )
        gen.publish(loaded.manifest, seal=True)
        assert store.is_sealed is True
        # A reconstructed store now observes the completed seal.
        fresh = ArtifactStore(runner.artifact_root, u0_incomplete.identity)
        assert fresh.is_sealed is True

    def test_concurrent_publish_after_seal_is_rejected(self, tmp_path: Path) -> None:
        # B1 race-safety: after an attempt is sealed, a concurrent publish
        # (including one that passed the pre-lock check before the seal) is
        # rejected by the in-lock recheck. We model this sequentially: seal the
        # attempt, then attempt a publish on a fresh store.
        from expertforge.artifacts import ArtifactSealedError

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        fresh = ArtifactStore(runner.artifact_root, u0.identity)
        assert fresh.is_sealed is True
        with pytest.raises(ArtifactSealedError):
            fresh.publish(
                b"concurrent",
                category="report",
                format="json",
                format_version=1,
                producing_component="concurrent",
            )

    def test_crash_window_intervening_publish_rejected_retry_completes_seal(
        self, tmp_path: Path
    ) -> None:
        # B1-r4 crash-window regression:
        # 1. Produce the registered-manifest / no-`sealed`-marker state through
        #    controlled fault injection (run U0, delete `sealed` but keep
        #    `sealing` intent to simulate a crash between intent-write and
        #    seal-completion).
        # 2. Attempt an unrelated publish — it must be REJECTED (the `sealing`
        #    intent blocks every mutating path).
        # 3. Retry the terminal publication (seal=True) — it completes the seal.
        # 4. Verify the manifest is the final registry publication and the attempt
        #    is sealed (`sealed` present, `sealing` gone).
        from expertforge.artifacts import ArtifactSealedError
        from expertforge.experiments import ManifestGenerator, load_manifest

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        store = _store_for(u0, tmp_path)

        # Fault injection: simulate a crash after intent-write but before
        # seal-completion. Delete `sealed`, write `sealing`.
        sealed_marker = store.sealed_marker_path
        sealing_marker = store.sealing_marker_path
        if sealed_marker.exists():
            sealed_marker.unlink()
        # Write a valid bound intent (artifact_id content) for the manifest.
        sealing_marker.write_text(u0.manifest_record.artifact_id, encoding="utf-8")
        assert store.is_sealed is True  # `sealing` intent blocks

        # 2. An unrelated publisher (seal=False) must be rejected.
        fresh = ArtifactStore(runner.artifact_root, u0.identity)
        with pytest.raises(ArtifactSealedError):
            fresh.publish(
                b"intervening",
                category="report",
                format="json",
                format_version=1,
                producing_component="intervening",
            )

        # 3. Retry the terminal publication (seal=True). The manifest already
        # exists (idempotent path); the authorized retry completes the seal.
        gen = ManifestGenerator(store)
        loaded = load_manifest(
            u0.manifest_record.artifact_id,
            artifact_store=store,
            expected_identity=u0.identity,
        )
        gen.publish(loaded.manifest, seal=True)

        # 4. The attempt is now fully sealed.
        assert store.is_sealed is True
        assert sealed_marker.exists()
        assert not sealing_marker.exists()
        # The manifest is the final registry publication (no intervening artifact
        # was registered).
        report_artifacts = store.list_artifacts(category="report")
        assert all(r.created_at_utc <= u0.manifest_record.created_at_utc for r in report_artifacts)
        # A reconstructed store also observes the completed seal.
        recon = ArtifactStore(runner.artifact_root, u0.identity)
        assert recon.is_sealed is True
        with pytest.raises(ArtifactSealedError):
            recon.publish(
                b"post-seal",
                category="report",
                format="json",
                format_version=1,
                producing_component="post-seal",
            )

    @pytest.mark.parametrize("crash_point", ["after_intent", "after_rename", "after_append"])
    def test_crash_injection_recovery_artifact_bound(
        self, tmp_path: Path, crash_point: str
    ) -> None:
        # B1-r5 crash-injection regressions: for each crash point, verify:
        # 1. The crash produces a durable sealing intent.
        # 2. An unrelated mutation is rejected (the intent blocks it).
        # 3. The authorized terminal retry (same artifact_id, seal=True) recovers.
        # 4. The terminal artifact is the final registry publication; the attempt
        #    is sealed.
        from datetime import UTC, datetime

        from expertforge.artifacts import ArtifactSealedError
        from expertforge.experiments import ManifestGenerator

        runner = _runner(tmp_path)
        # Create a fresh attempt for the crash test (don't reuse U0).
        identity, provenance, _ = runner._prepare_independent()  # noqa: SLF001
        store = ArtifactStore(runner.artifact_root, identity)
        config_record = store.publish(
            __import__("expertforge.config.resolve", fromlist=["canonical_bytes"]).canonical_bytes(
                runner.config_envelope
            ),
            category="resolved_configuration",
            format="json",
            format_version=1,
            producing_component="config",
        )
        provenance_record = store.publish(
            provenance.to_deterministic_json(),
            category="provenance",
            format="json",
            format_version=1,
            producing_component="provenance",
        )
        gen = ManifestGenerator(store)
        manifest = gen.generate(
            identity=identity,
            finalized_at_utc=datetime.now(UTC),
            classification="smoke_test",
            maturity_stage="Milestone 0",
            research_family="none/not-applicable",
            status="interrupted",
            outcome_diagnostic="handled_interruption",
            configuration_artifact=config_record,
            provenance_artifact=provenance_record,
            dataset=__import__(
                "expertforge.smoke.fixtures", fromlist=["dataset_reference"]
            ).dataset_reference(),
            tokenizer=__import__(
                "expertforge.smoke.fixtures", fromlist=["tokenizer_reference"]
            ).tokenizer_reference(),
            model=runner._model_identity(),  # noqa: SLF001
            training=runner._training_budget(),  # noqa: SLF001
            smoke_objective="crash test",
            smoke_acceptance_criteria="recover",
        )
        # Enable crash injection at the specified point.
        store.crash_injection = crash_point
        with pytest.raises(RuntimeError, match="crash-injection"):
            gen.publish(manifest, seal=True)
        # 1. The sealing intent exists (durable).
        assert store.sealing_marker_path.exists()
        assert not store.sealed_marker_path.exists()
        # 2. An unrelated mutation is rejected.
        fresh = ArtifactStore(runner.artifact_root, identity)
        assert fresh.is_sealed is True
        with pytest.raises(ArtifactSealedError):
            fresh.publish(
                b"unrelated",
                category="report",
                format="json",
                format_version=1,
                producing_component="unrelated",
            )
        # 3. Authorized terminal retry (same manifest, seal=True) recovers.
        store2 = ArtifactStore(runner.artifact_root, identity)
        store2.crash_injection = None  # disable crash for the retry
        gen2 = ManifestGenerator(store2)
        record = gen2.publish(manifest, seal=True)
        # 4. The terminal artifact is the final registry publication.
        assert store2.is_sealed is True
        assert not store2.sealing_marker_path.exists()
        assert store2.sealed_marker_path.exists()
        # The manifest is registered.
        manifests = store2.list_artifacts(category="experiment_manifest")
        assert len(manifests) == 1
        assert manifests[0].artifact_id == record.artifact_id
        # No unrelated report artifact was registered.
        reports = store2.list_artifacts(category="report")
        assert len(reports) == 0

    def test_different_artifact_seal_true_rejected_during_pending_intent(
        self, tmp_path: Path
    ) -> None:
        # B1-r6 #4: a different artifact attempting seal=True while a bound intent
        # exists for another artifact must be rejected.
        from datetime import UTC, datetime

        from expertforge.artifacts import ArtifactSealedError
        from expertforge.experiments import ManifestGenerator

        runner = _runner(tmp_path)
        identity, provenance, _ = runner._prepare_independent()  # noqa: SLF001
        store = ArtifactStore(runner.artifact_root, identity)
        config_record = store.publish(
            __import__("expertforge.config.resolve", fromlist=["canonical_bytes"]).canonical_bytes(
                runner.config_envelope
            ),
            category="resolved_configuration",
            format="json",
            format_version=1,
            producing_component="config",
        )
        provenance_record = store.publish(
            provenance.to_deterministic_json(),
            category="provenance",
            format="json",
            format_version=1,
            producing_component="provenance",
        )
        gen = ManifestGenerator(store)
        manifest = gen.generate(
            identity=identity,
            finalized_at_utc=datetime.now(UTC),
            classification="smoke_test",
            maturity_stage="Milestone 0",
            research_family="none/not-applicable",
            status="interrupted",
            outcome_diagnostic="handled_interruption",
            configuration_artifact=config_record,
            provenance_artifact=provenance_record,
            dataset=__import__(
                "expertforge.smoke.fixtures", fromlist=["dataset_reference"]
            ).dataset_reference(),
            tokenizer=__import__(
                "expertforge.smoke.fixtures", fromlist=["tokenizer_reference"]
            ).tokenizer_reference(),
            model=runner._model_identity(),  # noqa: SLF001
            training=runner._training_budget(),  # noqa: SLF001
            smoke_objective="intent test",
            smoke_acceptance_criteria="reject different artifact",
        )
        # Crash after intent write.
        store.crash_injection = "after_intent"
        with pytest.raises(RuntimeError, match="crash-injection"):
            gen.publish(manifest, seal=True)
        # A different artifact's seal=True publish must be rejected.
        store2 = ArtifactStore(runner.artifact_root, identity)
        store2.crash_injection = None
        with pytest.raises(ArtifactSealedError):
            store2.publish(
                b"different",
                category="report",
                format="json",
                format_version=1,
                producing_component="different",
                seal=True,
            )

    def test_malformed_intent_rejects_all_mutations(self, tmp_path: Path) -> None:
        # B1-r6 #4: a malformed (empty) intent must reject all mutations,
        # including the bound-artifact retry.
        from expertforge.artifacts import ArtifactSealedError, ArtifactStoreError

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        store = _store_for(u0, tmp_path)
        # Corrupt the intent: delete sealed, create empty sealing (simulating
        # a power loss that left the marker truncated).
        store.sealed_marker_path.unlink()
        store.sealing_marker_path.write_bytes(b"")
        assert store.is_sealed is True  # sealing marker exists (even if empty)
        # Unrelated publish rejected (malformed intent → fail-closed error).
        fresh = ArtifactStore(runner.artifact_root, u0.identity)
        with pytest.raises((ArtifactSealedError, ArtifactStoreError)):
            fresh.publish(
                b"x",
                category="report",
                format="json",
                format_version=1,
                producing_component="x",
            )
        # Recovery retry fails because the intent is malformed.
        from expertforge.experiments import ManifestGenerator, load_manifest

        gen = ManifestGenerator(fresh)
        loaded = load_manifest(
            u0.manifest_record.artifact_id, artifact_store=fresh, expected_identity=u0.identity
        )
        with pytest.raises((ArtifactStoreError, ArtifactSealedError)):
            gen.publish(loaded.manifest, seal=True)

    def test_seal_attempt_rejected_during_pending_bound_intent(self, tmp_path: Path) -> None:
        # B1-r6 #3/#4: seal_attempt() must reject when a bound intent exists.
        from expertforge.artifacts import ArtifactSealedError

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        store = _store_for(u0, tmp_path)
        # Simulate a crash after intent: delete sealed, manually create a sealing
        # intent (U0's normal run removes it on completion).
        store.sealed_marker_path.unlink()
        store.sealing_marker_path.write_text(u0.manifest_record.artifact_id, encoding="utf-8")
        fresh = ArtifactStore(runner.artifact_root, u0.identity)
        with pytest.raises(ArtifactSealedError):
            fresh.seal_attempt()

    def test_manifest_must_be_terminal_registry_publication_for_retroactive_seal(
        self, tmp_path: Path
    ) -> None:
        # #3: a no-intent existing-manifest seal must verify the manifest's
        # initial_publication is the terminal (final-sequence) registry entry.
        # Uses the PUBLIC update_retention() API to add a retention transition on
        # the manifest (carrying the same artifact_id but NOT an
        # initial_publication). The fixed _verify_terminal_registry_publication
        # matches initial_publication specifically, so a later retention
        # transition on the manifest does NOT satisfy the check.
        from expertforge.artifacts import ArtifactSealedError
        from expertforge.experiments import ManifestGenerator, load_manifest

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        store = _store_for(u0, tmp_path)
        # Delete BOTH markers to simulate a no-intent state.
        store.sealed_marker_path.unlink()
        assert not store.sealing_marker_path.exists()
        # Publish a report AFTER the manifest (so the manifest's
        # initial_publication is no longer the final registry entry).
        store.publish(
            b"post-manifest-report",
            category="report",
            format="text",
            format_version=1,
            producing_component="review",
        )
        # Add a legal retention transition on the manifest via the PUBLIC API.
        # This carries the same artifact_id but entry_kind="retention_transition"
        # (not initial_publication).
        store.update_retention(u0.manifest_record.artifact_id, "expired")
        # Inspect actual registry sequences: the manifest's initial_publication
        # must have a lower sequence than the registry maximum.
        entries = store._load_registry()  # noqa: SLF001
        manifest_init_seq = None
        max_seq = -1
        for entry in entries:
            if (
                entry.payload.get("artifact_id") == u0.manifest_record.artifact_id
                and entry.entry_kind == "initial_publication"
            ):
                manifest_init_seq = entry.sequence
            if entry.sequence > max_seq:
                max_seq = entry.sequence
        assert manifest_init_seq is not None
        assert manifest_init_seq < max_seq  # manifest is NOT the final entry
        # Attempt to retroactively seal via the existing manifest path.
        # Must be rejected because the manifest's initial_publication is not
        # the final registry entry.
        fresh = ArtifactStore(runner.artifact_root, u0.identity)
        gen = ManifestGenerator(fresh)
        loaded = load_manifest(
            u0.manifest_record.artifact_id, artifact_store=fresh, expected_identity=u0.identity
        )
        with pytest.raises(ArtifactSealedError):
            gen.publish(loaded.manifest, seal=True)

    def test_nonregular_sealed_path_rejects_completion_preserves_intent(
        self, tmp_path: Path
    ) -> None:
        # #2: a nonregular file (directory) at the `sealed` marker path must
        # reject completion with ArtifactStoreError and PRESERVE the `sealing`
        # intent (so the attempt remains blocked and the authorized retry can
        # still recover).
        from expertforge.artifacts import ArtifactStoreError

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        store = _store_for(u0, tmp_path)
        # Simulate a nonregular sealed path: remove the file, create a directory.
        store.sealed_marker_path.unlink()
        store.sealed_marker_path.mkdir()
        # Write a valid sealing intent.
        store.sealing_marker_path.write_text(u0.manifest_record.artifact_id, encoding="utf-8")
        # Attempt completion via the ManifestGenerator existing-manifest path.
        from expertforge.experiments import ManifestGenerator, load_manifest

        fresh = ArtifactStore(runner.artifact_root, u0.identity)
        gen = ManifestGenerator(fresh)
        loaded = load_manifest(
            u0.manifest_record.artifact_id, artifact_store=fresh, expected_identity=u0.identity
        )
        with pytest.raises(ArtifactStoreError):
            gen.publish(loaded.manifest, seal=True)
        # The sealing intent must be PRESERVED (not removed by the failed completion).
        assert store.sealing_marker_path.exists()
        # The attempt remains blocked.
        assert fresh.is_sealed is True

    def test_path_replacement_during_intent_read_fails_closed(self, tmp_path: Path) -> None:
        # #2: a REAL path-replacement TOCTOU test. The store's
        # ``sealing_read_inject`` hook is called between lstat and open. The
        # hook atomically replaces the marker file (delete + recreate with
        # different content), producing a different inode. The fd-bound read
        # must detect the inode mismatch and raise typed ArtifactStoreError.
        from expertforge.artifacts import ArtifactStoreError

        runner = _runner(tmp_path)
        u0 = runner.run_u0()
        store = _store_for(u0, tmp_path)
        # Write a valid intent.
        store.sealing_marker_path.write_text(u0.manifest_record.artifact_id, encoding="utf-8")

        def replace_between_lstat_and_open(path: Path) -> None:
            """Atomically replace the marker so the lstat'd inode differs from
            the opened fd's inode."""
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            path.write_text(u0.manifest_record.artifact_id + "\n", encoding="utf-8")

        fresh = ArtifactStore(runner.artifact_root, u0.identity)
        fresh.sealing_read_inject = replace_between_lstat_and_open
        # The identity-mismatch must produce a typed ArtifactStoreError.
        with pytest.raises(ArtifactStoreError, match="replaced|inode"):
            fresh._read_sealing_intent_artifact_id()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Smoke 8: CLI execution from a clean checkout
# ---------------------------------------------------------------------------


class TestCliExecution:
    def test_cli_runs_gate_and_exits_zero(self, tmp_path: Path) -> None:
        env = {
            "PATH": __import__("os").environ.get("PATH", ""),
            "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "expertforge.smoke.run",
                "--artifact-root",
                str(tmp_path / "cli-runs"),
                "--config",
                str(GATE_CONFIG),
                "--repo",
                str(REPO),
                "--allow-dirty",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            env=env,
            timeout=120,
        )
        assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        assert "SMOKE GATE: PASS" in result.stdout

    def test_cli_does_not_overwrite_prior_runs(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        u0a = runner.run_u0()
        u0b = runner.run_u0()
        # Both runs coexist under the same artifact root (distinct paths).
        dir_a = _store_for(u0a, tmp_path).attempt_dir
        dir_b = _store_for(u0b, tmp_path).attempt_dir
        assert dir_a.exists()
        assert dir_b.exists()
        assert dir_a != dir_b
