"""Regression for post-manifest review-report publication (Issue #12 amendment L)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from expertforge.artifacts import ArtifactStore
from expertforge.experiments import (
    DatasetReference,
    EvaluationSummary,
    ManifestGenerator,
    ModelIdentity,
    TokenizerReference,
    TrainingBudget,
    load_manifest,
)
from expertforge.identity.fingerprint import ImmutableInput, specification_fingerprint
from expertforge.identity.record import AttemptIdentityRecord

pytestmark = pytest.mark.integration


def test_later_review_report_does_not_invalidate_manifest_load(tmp_path: Path) -> None:
    dataset_hex = "1" * 64
    tokenizer_hex = "2" * 64
    identity = AttemptIdentityRecord(
        specification_fingerprint=specification_fingerprint(
            b'{"x":1}',
            [
                ImmutableInput(
                    name="dataset.fixture",
                    algorithm="sha256",
                    digest=dataset_hex,
                ),
                ImmutableInput(
                    name="tokenizer.fixture",
                    algorithm="sha256",
                    digest=tokenizer_hex,
                ),
            ],
        ),
        run_id="run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb",
        attempt_id="attempt-20260101t000000z-cccccccccccccccccccc",
        created_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    store = ArtifactStore(tmp_path / "runs", identity)
    configuration = store.publish(
        b'{"resolved":true}',
        category="resolved_configuration",
        format="json",
        format_version=1,
        producing_component="config",
    )
    provenance = store.publish(
        b'{"source":"fixture"}',
        category="provenance",
        format="json",
        format_version=1,
        producing_component="provenance",
    )
    telemetry = store.publish(
        b'{"record":"fixture"}\n',
        category="telemetry",
        format="jsonl",
        format_version=1,
        producing_component="telemetry",
    )
    generator = ManifestGenerator(store)
    manifest = generator.generate(
        identity=identity,
        finalized_at_utc=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        classification="smoke_test",
        maturity_stage="none/not-applicable",
        research_family="none/not-applicable",
        status="completed",
        configuration_artifact=configuration,
        provenance_artifact=provenance,
        dataset=DatasetReference(
            dataset_id="fixture-corpus-v1",
            split="train",
            immutable_input_name="dataset.fixture",
            content_digest=f"sha256:{dataset_hex}",
            is_fixture=True,
        ),
        tokenizer=TokenizerReference(
            tokenizer_id="fixture-tokenizer-v1",
            vocab_size=128,
            immutable_input_name="tokenizer.fixture",
            content_digest=f"sha256:{tokenizer_hex}",
            is_fixture=True,
        ),
        model=ModelIdentity(
            architecture="decoder-only Transformer",
            dim=32,
            n_layers=2,
            n_heads=4,
            ffn_dim=64,
            parameter_count=4096,
            parameter_count_kind="exact",
        ),
        training=TrainingBudget(
            token_budget=1024,
            batch_size=2,
            seq_len=32,
            learning_rate=1e-3,
            seed=7,
        ),
        evaluation=EvaluationSummary(
            eval_interval_tokens=256,
            final_loss=2.0,
            metrics_recorded=("loss",),
        ),
        smoke_objective="Verify manifest/report lifecycle separation.",
        smoke_acceptance_criteria="A later report does not invalidate loading.",
        telemetry_artifacts=(telemetry,),
    )
    manifest_record = generator.publish(manifest)
    store.publish(
        b"reviewed result",
        category="report",
        format="text",
        format_version=1,
        producing_component="review",
    )
    loaded = load_manifest(
        manifest_record.artifact_id,
        artifact_store=store,
        expected_identity=identity,
    )
    assert loaded.manifest == manifest
    assert loaded.artifact_record == manifest_record
