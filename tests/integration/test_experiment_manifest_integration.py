"""Portable generate/publish/load integration coverage for Issue #12."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from expertforge.artifacts import ArtifactRecord, ArtifactStore
from expertforge.experiments import (
    DatasetReference,
    EvaluationSummary,
    LoadedExperimentManifest,
    ManifestBindingError,
    ManifestCorruptError,
    ManifestGenerator,
    ManifestPublicationError,
    ModelIdentity,
    TokenizerReference,
    TrainingBudget,
    canonical_manifest_bytes,
    load_manifest,
    scan_manifest,
)
from expertforge.identity.fingerprint import (
    ImmutableInput,
    SpecificationFingerprintRecord,
    specification_fingerprint,
)
from expertforge.identity.lineage import ResumeLineage
from expertforge.identity.record import AttemptIdentityRecord

pytestmark = pytest.mark.integration

RUN_ID = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"
PARENT_ATTEMPT = "attempt-20260101t000000z-cccccccccccccccccccc"
CHILD_ATTEMPT = "attempt-20260101t000001z-dddddddddddddddddddd"
DATASET_HEX = "1" * 64
TOKENIZER_HEX = "2" * 64
FINALIZED_AT = datetime(2026, 1, 1, 0, 2, tzinfo=UTC)


def _fingerprint() -> SpecificationFingerprintRecord:
    return specification_fingerprint(
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
    )


def _identity(attempt_id: str, lineage: ResumeLineage | None = None) -> AttemptIdentityRecord:
    return AttemptIdentityRecord(
        specification_fingerprint=_fingerprint(),
        run_id=RUN_ID,
        attempt_id=attempt_id,
        created_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
        lineage=lineage,
    )


def _records(store: ArtifactStore) -> dict[str, ArtifactRecord]:
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


def _generate(
    store: ArtifactStore,
    records: dict[str, ArtifactRecord],
    **kwargs: Any,
):
    values: dict[str, Any] = {
        "identity": store.identity,
        "finalized_at_utc": FINALIZED_AT,
        "classification": "smoke_test",
        "maturity_stage": "none/not-applicable",
        "research_family": "none/not-applicable",
        "status": "completed",
        "issue_number": 12,
        "configuration_artifact": records["configuration"],
        "provenance_artifact": records["provenance"],
        "dataset": DatasetReference(
            dataset_id="fixture-corpus-v1",
            split="train",
            immutable_input_name="dataset.fixture",
            content_digest=f"sha256:{DATASET_HEX}",
            is_fixture=True,
        ),
        "tokenizer": TokenizerReference(
            tokenizer_id="fixture-tokenizer-v1",
            vocab_size=128,
            immutable_input_name="tokenizer.fixture",
            content_digest=f"sha256:{TOKENIZER_HEX}",
            is_fixture=True,
        ),
        "model": ModelIdentity(
            architecture="decoder-only Transformer",
            dim=32,
            n_layers=2,
            n_heads=4,
            ffn_dim=64,
            parameter_count=4096,
            parameter_count_kind="exact",
        ),
        "training": TrainingBudget(
            token_budget=1024,
            batch_size=2,
            seq_len=32,
            learning_rate=1e-3,
            seed=7,
        ),
        "evaluation": EvaluationSummary(
            eval_interval_tokens=256,
            final_loss=2.0,
            final_perplexity=7.4,
            metrics_recorded=("loss", "perplexity"),
        ),
        "smoke_objective": "Verify manifest publication and restoration.",
        "smoke_acceptance_criteria": "Exact typed round trip succeeds.",
        "checkpoint_evidence_required": True,
        "telemetry_artifacts": (records["telemetry"],),
        "checkpoint_artifacts": (records["checkpoint"],),
    }
    values.update(kwargs)
    return ManifestGenerator(store).generate(**values)


def test_generate_publish_load_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", _identity(PARENT_ATTEMPT))
    manifest = _generate(store, _records(store))
    record = ManifestGenerator(store).publish(manifest)
    loaded = load_manifest(
        record.artifact_id,
        artifact_store=store,
        expected_identity=store.identity,
    )
    assert isinstance(loaded, LoadedExperimentManifest)
    assert loaded.manifest == manifest
    assert loaded.artifact_record == record
    assert "manifest_id" not in json.loads(canonical_manifest_bytes(loaded.manifest))


def test_failed_attempt_publishes_explicit_incomplete_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", _identity(PARENT_ATTEMPT))
    generator = ManifestGenerator(store)
    manifest = generator.generate(
        identity=store.identity,
        finalized_at_utc=FINALIZED_AT,
        classification="smoke_test",
        maturity_stage="none/not-applicable",
        research_family="none/not-applicable",
        status="interrupted",
        outcome_diagnostic="handled_interruption",
        smoke_objective="Verify interruption evidence.",
        smoke_acceptance_criteria="An explicit partial manifest is published.",
    )
    record = generator.publish(manifest)
    loaded = load_manifest(
        record.artifact_id,
        artifact_store=store,
        expected_identity=store.identity,
    )
    assert loaded.manifest.status == "interrupted"
    assert loaded.manifest.evidence.status == "partial"
    assert "evaluation_summary_missing" in loaded.manifest.evidence.missing


@pytest.mark.parametrize(
    ("status", "diagnostic"),
    [
        ("completed", None),
        ("failed", "checkpoint_restore_failed"),
        ("interrupted", "handled_interruption"),
    ],
)
def test_resumed_attempt_outcome_is_orthogonal(
    tmp_path: Path,
    status: str,
    diagnostic: str | None,
) -> None:
    root = tmp_path / "runs"
    parent_store = ArtifactStore(root, _identity(PARENT_ATTEMPT))
    parent_checkpoint = _records(parent_store)["checkpoint"]
    lineage = ResumeLineage(
        parent_run_id=RUN_ID,
        parent_attempt_id=PARENT_ATTEMPT,
        parent_checkpoint_id=parent_checkpoint.artifact_id,
    )
    child_store = ArtifactStore(root, _identity(CHILD_ATTEMPT, lineage))
    records = _records(child_store)
    manifest = _generate(
        child_store,
        records,
        status=status,
        outcome_diagnostic=diagnostic,
        resume_checkpoint=parent_checkpoint,
    )
    resolver = lambda observed: (  # noqa: E731 - compact typed fixture resolver
        parent_store,
        parent_checkpoint if observed == lineage else records["checkpoint"],
    )
    record = ManifestGenerator(child_store).publish(
        manifest,
        parent_checkpoint_resolver=resolver,
    )
    loaded = load_manifest(
        record.artifact_id,
        artifact_store=child_store,
        expected_identity=child_store.identity,
        parent_checkpoint_resolver=resolver,
    )
    assert loaded.manifest.status == status
    assert loaded.manifest.identity.lineage == lineage
    assert loaded.manifest.resume_checkpoint == parent_checkpoint


def test_registry_coverage_rejects_unrepresented_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", _identity(PARENT_ATTEMPT))
    records = _records(store)
    store.publish(
        b"unrepresented output",
        category="generated_sample",
        format="text",
        format_version=1,
        producing_component="generation",
    )
    manifest = _generate(store, records)
    with pytest.raises(ManifestBindingError, match="coverage mismatch"):
        ManifestGenerator(store).publish(manifest)


def test_referenced_artifact_payload_must_verify_before_publication(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", _identity(PARENT_ATTEMPT))
    records = _records(store)
    manifest = _generate(store, records)
    configuration = store.locate(records["configuration"].artifact_id)
    assert configuration is not None
    configuration.write_bytes(b'{"resolved":false}')
    with pytest.raises(ManifestBindingError, match="failed verification"):
        ManifestGenerator(store).publish(manifest)


def test_scan_classifies_valid_unsupported_and_corrupt(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", _identity(PARENT_ATTEMPT))
    manifest = _generate(store, _records(store))
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_manifest_bytes(manifest))
    assert scan_manifest(path).status == "valid"
    data = json.loads(path.read_bytes())
    data["format_version"] = 2
    path.write_text(
        json.dumps(data, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    assert scan_manifest(path).status == "unsupported"
    path.write_bytes(b"not-json")
    assert scan_manifest(path).status == "corrupt"


def test_load_rejects_content_tampering(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", _identity(PARENT_ATTEMPT))
    manifest = _generate(store, _records(store))
    record = ManifestGenerator(store).publish(manifest)
    content = store.locate(record.artifact_id)
    assert content is not None
    raw = bytearray(content.read_bytes())
    raw[-2] = ord("0") if raw[-2] != ord("0") else ord("1")
    content.write_bytes(raw)
    with pytest.raises((ManifestBindingError, ManifestCorruptError)):
        load_manifest(
            record.artifact_id,
            artifact_store=store,
            expected_identity=store.identity,
        )


def test_different_second_terminal_manifest_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", _identity(PARENT_ATTEMPT))
    records = _records(store)
    generator = ManifestGenerator(store)
    first = _generate(store, records)
    generator.publish(first)
    second = _generate(store, records, known_limitations=("different",))
    with pytest.raises(ManifestPublicationError, match="different terminal"):
        generator.publish(second)
