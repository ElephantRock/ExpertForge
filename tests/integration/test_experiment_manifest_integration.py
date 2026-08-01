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
    ExperimentManifest,
    ManifestBindingError,
    ManifestCorruptError,
    ManifestGenerator,
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
) -> ExperimentManifest:
    return ManifestGenerator(store).generate(
        identity=store.identity,
        classification="smoke_test",
        maturity_stage="Milestone 0",
        research_family="none/not-applicable",
        status="completed",
        issue_number=12,
        configuration_artifact=records["configuration"],
        provenance_artifact=records["provenance"],
        dataset=DatasetReference(
            dataset_id="fixture-corpus-v1",
            split="train",
            immutable_input_name="dataset.fixture",
            content_digest=f"sha256:{DATASET_HEX}",
            is_fixture=True,
        ),
        tokenizer=TokenizerReference(
            tokenizer_id="fixture-tokenizer-v1",
            vocab_size=128,
            immutable_input_name="tokenizer.fixture",
            content_digest=f"sha256:{TOKENIZER_HEX}",
            is_fixture=True,
        ),
        fixture=True,
        model=ModelIdentity(
            architecture="decoder-only Transformer",
            dim=32,
            n_layers=2,
            n_heads=4,
            ffn_dim=64,
            parameter_count=4096,
        ),
        training=TrainingBudget(
            token_budget=1024,
            batch_size=2,
            seq_len=32,
            learning_rate=1e-3,
            seed=7,
        ),
        telemetry_artifacts=(records["telemetry"],),
        checkpoint_artifacts=(records["checkpoint"],),
        **kwargs,
    )


def test_generate_publish_load_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", _identity(PARENT_ATTEMPT))
    manifest = _generate(store, _records(store))
    record = ManifestGenerator(store).publish(manifest)
    assert record.category == "experiment_manifest"
    assert record.format == "json"
    assert store.list_artifacts()[-1].artifact_id == record.artifact_id
    loaded = load_manifest(
        record.artifact_id,
        artifact_store=store,
        expected_identity=store.identity,
    )
    assert loaded == manifest
    assert "manifest_id" not in json.loads(canonical_manifest_bytes(loaded))


def test_failed_attempt_publishes_explicit_incomplete_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", _identity(PARENT_ATTEMPT))
    generator = ManifestGenerator(store)
    manifest = generator.generate(
        identity=store.identity,
        classification="smoke_test",
        maturity_stage="Milestone 0",
        research_family="none/not-applicable",
        status="interrupted",
        outcome_diagnostic="handled_interruption",
    )
    record = generator.publish(manifest)
    loaded = load_manifest(
        record.artifact_id,
        artifact_store=store,
        expected_identity=store.identity,
    )
    assert loaded.status == "interrupted"
    assert loaded.evidence.status == "partial"


def test_resumed_attempt_requires_and_resolves_parent_checkpoint(tmp_path: Path) -> None:
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
    manifest = _generate(child_store, records, resume_checkpoint=parent_checkpoint)
    record = ManifestGenerator(child_store).publish(manifest)
    with pytest.raises(ManifestBindingError, match="parent_checkpoint_resolver"):
        load_manifest(
            record.artifact_id,
            artifact_store=child_store,
            expected_identity=child_store.identity,
        )
    loaded = load_manifest(
        record.artifact_id,
        artifact_store=child_store,
        expected_identity=child_store.identity,
        parent_checkpoint_resolver=lambda observed: (
            parent_checkpoint if observed == lineage else records["checkpoint"]
        ),
    )
    assert loaded.resume_checkpoint == parent_checkpoint
    inspected = child_store.inspect(record.artifact_id)
    assert isinstance(inspected, ArtifactRecord)
    assert inspected.parent is not None
    assert inspected.parent.artifact_id == parent_checkpoint.artifact_id


def test_scan_classifies_complete_unsupported_and_corrupt(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", _identity(PARENT_ATTEMPT))
    manifest = _generate(store, _records(store))
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_manifest_bytes(manifest))
    assert scan_manifest(path).status == "complete"
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
