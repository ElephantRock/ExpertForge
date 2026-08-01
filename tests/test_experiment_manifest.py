"""Fast contract tests for versioned experiment manifests (Issue #12)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from expertforge.artifacts import ArtifactRecord, ArtifactStore
from expertforge.experiments import (
    DatasetReference,
    EvaluationSummary,
    ExperimentManifest,
    ManifestCorruptError,
    ManifestGenerator,
    ManifestVersionError,
    ModelIdentity,
    TokenizerReference,
    TrainingBudget,
    canonical_manifest_bytes,
    experiment_manifest_json_schema,
    parse_manifest_bytes,
)
from expertforge.identity.fingerprint import ImmutableInput, specification_fingerprint
from expertforge.identity.record import AttemptIdentityRecord

RUN_ID = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"
ATTEMPT_ID = "attempt-20260101t000000z-cccccccccccccccccccc"
DATASET_HEX = "1" * 64
TOKENIZER_HEX = "2" * 64


def _identity() -> AttemptIdentityRecord:
    return AttemptIdentityRecord(
        specification_fingerprint=specification_fingerprint(
            b'{"x":1}',
            [
                ImmutableInput(
                    name="dataset.fixture",
                    algorithm="sha256",
                    digest=DATASET_HEX,
                ),
                ImmutableInput(
                    name="tokenizer.fixture",
                    algorithm="sha256",
                    digest=TOKENIZER_HEX,
                ),
            ],
        ),
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        created_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "runs", _identity())


def _publish_evidence(store: ArtifactStore) -> dict[str, ArtifactRecord]:
    return {
        "configuration": store.publish(
            b'{"resolved":true}',
            category="resolved_configuration",
            format="json",
            format_version=1,
            producing_component="config",
        ),
        "provenance": store.publish(
            b'{"source":"fixture"}',
            category="provenance",
            format="json",
            format_version=1,
            producing_component="provenance",
        ),
        "telemetry": store.publish(
            b'{"record":"fixture"}\n',
            category="telemetry",
            format="jsonl",
            format_version=1,
            producing_component="telemetry",
        ),
        "checkpoint": store.publish(
            b"fixture-tar",
            category="checkpoint",
            format="tar",
            format_version=1,
            producing_component="checkpoints",
        ),
    }


def _dataset() -> DatasetReference:
    return DatasetReference(
        dataset_id="fixture-corpus-v1",
        split="train",
        immutable_input_name="dataset.fixture",
        content_digest=f"sha256:{DATASET_HEX}",
        is_fixture=True,
    )


def _tokenizer() -> TokenizerReference:
    return TokenizerReference(
        tokenizer_id="fixture-tokenizer-v1",
        vocab_size=128,
        immutable_input_name="tokenizer.fixture",
        content_digest=f"sha256:{TOKENIZER_HEX}",
        is_fixture=True,
    )


def _model() -> ModelIdentity:
    return ModelIdentity(
        architecture="decoder-only Transformer",
        dim=32,
        n_layers=2,
        n_heads=4,
        ffn_dim=64,
        parameter_count=4096,
    )


def _training() -> TrainingBudget:
    return TrainingBudget(
        token_budget=1024,
        batch_size=2,
        seq_len=32,
        learning_rate=1e-3,
        seed=7,
    )


def _complete_manifest(tmp_path: Path) -> tuple[ManifestGenerator, ExperimentManifest]:
    store = _store(tmp_path)
    evidence = _publish_evidence(store)
    generator = ManifestGenerator(store)
    manifest = generator.generate(
        identity=store.identity,
        classification="smoke_test",
        maturity_stage="Milestone 0",
        research_family="none/not-applicable",
        status="completed",
        issue_number=12,
        configuration_artifact=evidence["configuration"],
        provenance_artifact=evidence["provenance"],
        dataset=_dataset(),
        tokenizer=_tokenizer(),
        fixture=True,
        model=_model(),
        training=_training(),
        evaluation=EvaluationSummary(
            eval_interval_tokens=128,
            final_loss=1.25,
            final_perplexity=3.49,
            metrics_recorded=("loss", "perplexity"),
        ),
        telemetry_artifacts=(evidence["telemetry"],),
        checkpoint_artifacts=(evidence["checkpoint"],),
    )
    return generator, manifest


def test_manifest_is_frozen_and_has_no_circular_manifest_id(tmp_path: Path) -> None:
    _, manifest = _complete_manifest(tmp_path)
    payload = json.loads(canonical_manifest_bytes(manifest))
    assert "manifest_id" not in payload
    with pytest.raises(ValidationError):
        manifest.status = "failed"  # type: ignore[misc]


def test_canonical_round_trip(tmp_path: Path) -> None:
    _, manifest = _complete_manifest(tmp_path)
    raw = canonical_manifest_bytes(manifest)
    assert parse_manifest_bytes(raw) == manifest
    assert raw == canonical_manifest_bytes(parse_manifest_bytes(raw))


def test_noncanonical_and_duplicate_json_rejected(tmp_path: Path) -> None:
    _, manifest = _complete_manifest(tmp_path)
    raw = canonical_manifest_bytes(manifest)
    with pytest.raises(ManifestCorruptError, match="canonical"):
        parse_manifest_bytes(b" " + raw)
    duplicate = raw[:-1] + b',"schema":"expertforge.experiment-manifest"}'
    with pytest.raises(ManifestCorruptError, match="duplicate"):
        parse_manifest_bytes(duplicate)


def test_unknown_versions_are_unsupported(tmp_path: Path) -> None:
    _, manifest = _complete_manifest(tmp_path)
    data = json.loads(canonical_manifest_bytes(manifest))
    data["schema_version"] = 2
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ManifestVersionError):
        parse_manifest_bytes(raw)


def test_formal_contract_required_and_smoke_contract_forbidden(tmp_path: Path) -> None:
    store = _store(tmp_path)
    generator = ManifestGenerator(store)
    with pytest.raises(ValidationError, match="formal_experiment"):
        generator.generate(
            identity=store.identity,
            classification="formal_experiment",
            maturity_stage="D0",
            research_family="F0",
            status="failed",
            outcome_diagnostic="data_failure",
        )
    with pytest.raises(ValidationError, match="smoke_test"):
        generator.generate(
            identity=store.identity,
            classification="smoke_test",
            maturity_stage="Milestone 0",
            research_family="none/not-applicable",
            status="failed",
            outcome_diagnostic="data_failure",
            hypothesis="forbidden",
        )


def test_incomplete_failed_manifest_declares_exact_missing_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifest = ManifestGenerator(store).generate(
        identity=store.identity,
        classification="smoke_test",
        maturity_stage="Milestone 0",
        research_family="none/not-applicable",
        status="failed",
        outcome_diagnostic="configuration_error",
        known_limitations=("failed-before-evidence-publication",),
    )
    assert manifest.evidence.status == "partial"
    assert manifest.evidence.missing == (
        "configuration_artifact_missing",
        "dataset_identity_missing",
        "model_identity_missing",
        "provenance_artifact_missing",
        "telemetry_artifact_missing",
        "tokenizer_identity_missing",
        "training_budget_missing",
    )


def test_completed_manifest_cannot_omit_core_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValidationError, match="completed attempts"):
        ManifestGenerator(store).generate(
            identity=store.identity,
            classification="smoke_test",
            maturity_stage="Milestone 0",
            research_family="none/not-applicable",
            status="completed",
        )


def test_dataset_and_tokenizer_bind_to_fingerprint(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValidationError, match="fingerprint immutable input"):
        ManifestGenerator(store).generate(
            identity=store.identity,
            classification="smoke_test",
            maturity_stage="Milestone 0",
            research_family="none/not-applicable",
            status="failed",
            outcome_diagnostic="data_failure",
            dataset=DatasetReference(
                dataset_id="bad",
                split="train",
                immutable_input_name="dataset.fixture",
                content_digest="sha256:" + "f" * 64,
                is_fixture=True,
            ),
        )


def test_artifact_role_and_attempt_identity_are_bound(tmp_path: Path) -> None:
    store = _store(tmp_path)
    report = store.publish(
        b"report",
        category="report",
        format="text",
        format_version=1,
        producing_component="test",
    )
    with pytest.raises(ValidationError, match="telemetry artifact requires"):
        ManifestGenerator(store).generate(
            identity=store.identity,
            classification="smoke_test",
            maturity_stage="Milestone 0",
            research_family="none/not-applicable",
            status="failed",
            outcome_diagnostic="telemetry_write_failed",
            telemetry_artifacts=(report,),
        )


def test_schema_document_tracks_top_level_model_fields() -> None:
    schema = experiment_manifest_json_schema()
    aliases = {field.alias or name for name, field in ExperimentManifest.model_fields.items()}
    assert aliases == set(schema["properties"])
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["allOf"]
    committed = json.loads(
        (Path(__file__).resolve().parents[1] / "schemas" / "experiment-manifest-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(committed["properties"]) == aliases
    assert committed["allOf"] == schema["allOf"]


def test_public_surface_is_explicit() -> None:
    import expertforge.experiments as experiments

    assert "ExperimentManifest" in experiments.__all__
    assert "ManifestGenerator" in experiments.__all__
    assert "load_manifest" in experiments.__all__
