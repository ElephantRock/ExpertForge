"""Fast contract tests for versioned experiment manifests (Issue #12)."""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from expertforge.artifacts import ArtifactRecord, ArtifactStore
from expertforge.experiments import (
    MAX_MANIFEST_BYTES,
    DatasetReference,
    EvaluationSummary,
    ExperimentManifest,
    ManifestCorruptError,
    ManifestGenerator,
    ManifestPublicationError,
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
FINALIZED_AT = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)


class SchemaValidationError(AssertionError):
    """Failure from the dependency-free Draft 2020-12 keyword subset used here."""


def _resolve_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise SchemaValidationError(f"unsupported external schema reference {reference!r}")
    value: Any = root
    for part in reference[2:].split("/"):
        value = value[part.replace("~1", "/").replace("~0", "~")]
    assert isinstance(value, dict)
    return value


def _schema_type_matches(instance: Any, expected: str) -> bool:
    if expected == "null":
        return instance is None
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "object":
        return isinstance(instance, dict)
    raise SchemaValidationError(f"unsupported schema type {expected!r}")


def _validate_schema(instance: Any, schema: dict[str, Any], root: dict[str, Any]) -> None:
    if "$ref" in schema:
        _validate_schema(instance, _resolve_ref(root, schema["$ref"]), root)
        return
    if "const" in schema and instance != schema["const"]:
        raise SchemaValidationError(f"expected const {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaValidationError(f"{instance!r} is outside enum {schema['enum']!r}")
    if "anyOf" in schema:
        errors: list[Exception] = []
        for option in schema["anyOf"]:
            try:
                _validate_schema(instance, option, root)
            except SchemaValidationError as exc:
                errors.append(exc)
            else:
                break
        else:
            raise SchemaValidationError(f"no anyOf branch matched: {errors!r}")
    if "allOf" in schema:
        for item in schema["allOf"]:
            _validate_schema(instance, item, root)
    if "if" in schema:
        try:
            _validate_schema(instance, schema["if"], root)
        except SchemaValidationError:
            branch = schema.get("else")
        else:
            branch = schema.get("then")
        if branch is not None:
            _validate_schema(instance, branch, root)

    expected_type = schema.get("type")
    if expected_type is not None and not _schema_type_matches(instance, expected_type):
        raise SchemaValidationError(f"expected {expected_type}, got {type(instance).__name__}")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        missing = [name for name in required if name not in instance]
        if missing:
            raise SchemaValidationError(f"missing required properties {missing!r}")
        properties = schema.get("properties", {})
        for name, child in properties.items():
            if name in instance:
                _validate_schema(instance[name], child, root)
        if schema.get("additionalProperties") is False:
            extras = sorted(set(instance) - set(properties))
            if extras:
                raise SchemaValidationError(f"unexpected properties {extras!r}")

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            raise SchemaValidationError("array shorter than minItems")
        maximum = schema.get("maxItems")
        if maximum is not None and len(instance) > maximum:
            raise SchemaValidationError("array longer than maxItems")
        if schema.get("uniqueItems") and len(
            {json.dumps(value, sort_keys=True) for value in instance}
        ) != len(instance):
            raise SchemaValidationError("array items are not unique")
        item_schema = schema.get("items")
        if item_schema is not None:
            for value in instance:
                _validate_schema(value, item_schema, root)

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise SchemaValidationError("string shorter than minLength")
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(pattern, instance) is None:
            raise SchemaValidationError(f"string {instance!r} does not match {pattern!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise SchemaValidationError("number below minimum")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            raise SchemaValidationError("number not above exclusiveMinimum")
        if "maximum" in schema and instance > schema["maximum"]:
            raise SchemaValidationError("number above maximum")


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


def _generator_kwargs(records: dict[str, ArtifactRecord]) -> dict[str, Any]:
    return {
        "identity": _identity(),
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
        "smoke_objective": "Verify the complete attempt evidence path.",
        "smoke_acceptance_criteria": "Manifest publishes and reloads exactly.",
        "checkpoint_evidence_required": True,
        "telemetry_artifacts": (records["telemetry"],),
        "checkpoint_artifacts": (records["checkpoint"],),
    }


def _manifest(store: ArtifactStore, records: dict[str, ArtifactRecord]) -> ExperimentManifest:
    kwargs = _generator_kwargs(records)
    kwargs["identity"] = store.identity
    return ManifestGenerator(store).generate(**kwargs)


def test_complete_manifest_requires_all_core_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    records = _publish_evidence(store)
    kwargs = _generator_kwargs(records)
    kwargs["identity"] = store.identity
    kwargs["evaluation"] = None
    with pytest.raises(ValidationError, match="complete core evidence"):
        ManifestGenerator(store).generate(**kwargs)


def test_optional_expected_artifacts_have_closed_missing_codes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    generator = ManifestGenerator(store)
    manifest = generator.generate(
        identity=store.identity,
        finalized_at_utc=FINALIZED_AT,
        classification="smoke_test",
        maturity_stage="none/not-applicable",
        research_family="none/not-applicable",
        status="failed",
        outcome_diagnostic="data_failure",
        smoke_objective="Exercise failure reporting.",
        smoke_acceptance_criteria="Missing evidence is explicit.",
        checkpoint_evidence_required=True,
        generated_output_evidence_required=True,
    )
    assert "checkpoint_artifact_missing" in manifest.evidence.missing
    assert "generated_output_artifact_missing" in manifest.evidence.missing
    assert "evaluation_summary_missing" in manifest.evidence.missing


def test_classification_contracts_are_complete(tmp_path: Path) -> None:
    store = _store(tmp_path)
    records = _publish_evidence(store)
    kwargs = _generator_kwargs(records)
    kwargs["identity"] = store.identity
    kwargs["smoke_objective"] = None
    with pytest.raises(ValidationError, match="smoke_test requires"):
        ManifestGenerator(store).generate(**kwargs)

    formal = _generator_kwargs(records)
    formal.update(
        {
            "identity": store.identity,
            "classification": "formal_experiment",
            "smoke_objective": None,
            "smoke_acceptance_criteria": None,
            "hypothesis": "A controlled change improves the metric.",
            "control": "Current baseline.",
            "fixed_constraints": ("seed=7",),
            "independent_variable": "optimizer",
            "dependent_variables": ("loss",),
            "minimum_useful_effect": "loss improves by at least 0.01",
            "failure_threshold": "loss regresses by 0.02",
            "kill_criterion": "stop on non-finite loss",
        }
    )
    assert ManifestGenerator(store).generate(**formal).classification == "formal_experiment"
    formal["kill_criterion"] = None
    with pytest.raises(ValidationError, match="kill_criterion"):
        ManifestGenerator(store).generate(**formal)


def test_finalization_time_and_review_snapshot_are_explicit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    records = _publish_evidence(store)
    manifest = _manifest(store, records)
    assert manifest.finalized_at_utc == FINALIZED_AT
    assert manifest.model_descriptor is not None
    assert manifest.model_descriptor.parameter_count_kind == "exact"
    assert manifest.fixture is True
    encoded = json.loads(canonical_manifest_bytes(manifest))
    assert "fixture" not in encoded
    assert "manifest_retention_intent" not in encoded
    assert "finalized_at_utc" in encoded

    kwargs = _generator_kwargs(records)
    kwargs.update(
        {
            "identity": store.identity,
            "result": "Observed stable completion.",
            "decision": "Proceed to the smoke gate.",
        }
    )
    reviewed = ManifestGenerator(store).generate(**kwargs)
    assert reviewed.result is not None
    kwargs.update({"status": "failed", "outcome_diagnostic": "data_failure"})
    with pytest.raises(ValidationError, match="completed attempts"):
        ManifestGenerator(store).generate(**kwargs)


def test_generation_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    records = _publish_evidence(store)
    first = _manifest(store, records)
    second = _manifest(store, records)
    assert canonical_manifest_bytes(first) == canonical_manifest_bytes(second)


def test_publish_rejects_oversized_manifest_before_registration(tmp_path: Path) -> None:
    store = _store(tmp_path)
    records = _publish_evidence(store)
    kwargs = _generator_kwargs(records)
    kwargs.update(
        {
            "identity": store.identity,
            "known_limitations": ("x" * MAX_MANIFEST_BYTES,),
        }
    )
    manifest = ManifestGenerator(store).generate(**kwargs)
    with pytest.raises(ManifestPublicationError, match="MAX_MANIFEST_BYTES"):
        ManifestGenerator(store).publish(manifest)
    assert store.list_artifacts(category="experiment_manifest") == []


def test_parser_rejects_versions_duplicates_noncanonical_and_nonfinite(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifest = _manifest(store, _publish_evidence(store))
    raw = canonical_manifest_bytes(manifest)
    assert parse_manifest_bytes(raw) == manifest

    data = json.loads(raw)
    data["schema_version"] = 2
    with pytest.raises(ManifestVersionError):
        parse_manifest_bytes(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ManifestCorruptError, match="duplicate JSON key"):
        parse_manifest_bytes(b'{"schema":"x","schema":"y"}')
    with pytest.raises(ManifestCorruptError, match="not canonical"):
        parse_manifest_bytes(raw + b"\n")
    with pytest.raises(ManifestCorruptError, match="non-finite"):
        parse_manifest_bytes(raw.replace(b'"final_loss":2.0', b'"final_loss":NaN'))


def test_schema_bytes_match_and_independently_validate_matrix(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifest = _manifest(store, _publish_evidence(store))
    generated = experiment_manifest_json_schema()
    expected_bytes = json.dumps(
        generated,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "experiment-manifest-v1.json"
    committed = schema_path.read_bytes()
    assert committed == expected_bytes, expected_bytes.decode()

    valid = json.loads(canonical_manifest_bytes(manifest))
    _validate_schema(valid, generated, generated)

    invalid: list[dict[str, Any]] = []
    case = copy.deepcopy(valid)
    case["evaluation_summary"] = None
    invalid.append(case)
    case = copy.deepcopy(valid)
    case["smoke_objective"] = None
    invalid.append(case)
    case = copy.deepcopy(valid)
    case["evidence"] = {"status": "complete", "missing": ["evaluation_summary_missing"]}
    invalid.append(case)
    case = copy.deepcopy(valid)
    case["resume_checkpoint"] = valid["checkpoint_artifacts"][0]
    invalid.append(case)
    case = copy.deepcopy(valid)
    case["tokenizer_identity"]["is_fixture"] = False
    invalid.append(case)
    case = copy.deepcopy(valid)
    case["configuration_artifact"]["content_digest"] = "sha256:not-a-digest"
    invalid.append(case)

    for value in invalid:
        with pytest.raises(SchemaValidationError):
            _validate_schema(value, generated, generated)
        with pytest.raises(ValidationError):
            ExperimentManifest.model_validate(value, strict=False)


def test_public_surface_is_explicit() -> None:
    import expertforge.experiments as experiments

    assert "LoadedExperimentManifest" in experiments.__all__
    assert "ParameterCountKind" in experiments.__all__
    assert "ManifestInspectionDiagnosticCode" in experiments.__all__
