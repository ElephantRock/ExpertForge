"""Validate the prospective D0 formal experiment definition without publishing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from expertforge.config.resolve import resolve_config
from expertforge.experiments import (
    DatasetReference,
    EvaluationSummary,
    EvidenceCompleteness,
    ExperimentManifest,
    MissingEvidenceCode,
    ModelIdentity,
    TokenizerReference,
    TrainingBudget,
    canonical_manifest_bytes,
    parse_manifest_bytes,
)
from expertforge.identity.fingerprint import SpecificationFingerprintRecord
from expertforge.identity.record import AttemptIdentityRecord
from scripts.validate_d0_config_binding import (
    CONFIG_PATHS,
    CONTRACT_PATH,
    FINGERPRINT_PATHS,
    ROOT,
    validate_all as validate_config_bindings,
)
from scripts.validate_d0_source_manifests import ContractValidationError, load_json_object

DEFINITION_PATH = ROOT / "experiments/d0/formal-experiment-definition-v1.json"
Profile = Literal["qualification", "canonical"]
_PROFILES: tuple[Profile, ...] = ("qualification", "canonical")
_FORBIDDEN_PLACEHOLDERS = ("tbd", "approximately", "approximate", "todo")
_EXPECTED_MISSING: tuple[MissingEvidenceCode, ...] = (
    "checkpoint_artifact_missing",
    "configuration_artifact_missing",
    "generated_output_artifact_missing",
    "provenance_artifact_missing",
    "telemetry_artifact_missing",
)
_SYNTHETIC_IDS = {
    "qualification": (
        "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb",
        "attempt-20260101t000000z-cccccccccccccccccccc",
    ),
    "canonical": (
        "run-20260101t000000z-dddddddddddd-eeeeeeeeeeeeeeeeeeeeeeee",
        "attempt-20260101t000000z-ffffffffffffffffffff",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _walk_strings(value: object) -> Sequence[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, Mapping):
        for key, member in value.items():
            strings.extend(_walk_strings(key))
            strings.extend(_walk_strings(member))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for member in value:
            strings.extend(_walk_strings(member))
    return strings


def _load_fingerprint(path: Path) -> SpecificationFingerprintRecord:
    record = SpecificationFingerprintRecord.model_validate_json(path.read_text(encoding="utf-8"))
    record.verify_digest()
    return record


def _expected_profile(
    contract: Mapping[str, Any],
    profile: Profile,
    fingerprint: SpecificationFingerprintRecord,
) -> dict[str, Any]:
    model = contract["models"][profile]
    schedule = contract["schedules"][profile]
    thresholds = contract["thresholds"]
    loss = thresholds[profile]["minimum_final_validation_loss_improvement_nats"]
    loss_text = f"{loss:.2f}"

    fixed_constraints = sorted(
        [
            f"architecture={contract['architecture']['family']}",
            f"attention={contract['architecture']['attention']}",
            (
                "batch.global_sequences_per_update="
                f"{contract['batch']['global_sequences_per_update']}"
            ),
            f"batch.target_tokens_per_update={contract['batch']['target_tokens_per_update']}",
            f"configuration.specification_fingerprint={fingerprint.digest_str}",
            f"dataset.manifest.sha256={contract['dataset']['source_manifest_sha256']}",
            (
                "evaluation.validation_target_tokens_per_boundary="
                f"{contract['evaluation']['validation_target_tokens_per_boundary']}"
            ),
            f"model.model_id={model['model_id']}",
            f"model.trainable_parameters={model['trainable_parameters']}",
            (
                "optimizer=AdamW("
                f"beta1={contract['optimizer']['beta1']},"
                f"beta2={contract['optimizer']['beta2']},"
                f"epsilon={contract['optimizer']['epsilon']},"
                f"weight_decay={contract['optimizer']['weight_decay']},"
                "gradient_clip_global_l2_norm="
                f"{contract['optimizer']['gradient_clip_global_l2_norm']})"
            ),
            (
                "precision.parameter_and_compute_dtype="
                f"{contract['precision_and_environment']['parameter_and_compute_dtype']}"
            ),
            f"seeds.master_seed={contract['seeds']['master_seed']}",
            (
                "sequence.model_context_tokens="
                f"{contract['sequence_semantics']['model_context_tokens']}"
            ),
            (
                "sequence.processed_token_definition="
                f"{contract['sequence_semantics']['processed_token_definition']}"
            ),
            f"tokenizer.manifest.sha256={contract['tokenizer']['source_manifest_sha256']}",
            f"training.optimizer_updates={schedule['optimizer_updates']}",
            f"training.target_tokens={schedule['training_target_tokens']}",
        ]
    )
    dependent_variables = sorted(
        [
            "checkpoint_read_seconds",
            "checkpoint_write_seconds",
            "generation_completion",
            "non_finite_event_count",
            "peak_device_memory_fraction",
            "peak_host_memory_fraction",
            "perplexity",
            "recovery_state_equality",
            "skipped_optimizer_updates",
            "throughput_target_tokens_per_second",
            "validation_loss_nats",
        ]
    )
    known_limitations = sorted(
        [
            "This prospective definition is not a terminal attempt manifest.",
            "Training seed variance is not claimed.",
            "Validation uses synthetic interrupted identities and publishes no artifacts.",
        ]
    )

    return {
        "config_path": f"configs/d0/{profile}.yaml",
        "fingerprint_path": f"experiments/d0/{profile}-fingerprint.json",
        "specification_fingerprint": fingerprint.digest_str,
        "hypothesis": (
            f"Training the frozen {profile} model for {schedule['training_target_tokens']} "
            f"target tokens will improve fixed-validation loss by at least {loss_text} "
            "natural-log nats relative to update zero while satisfying every declared "
            "stability, recovery, systems, checkpoint, and generation gate."
        ),
        "control": (
            "The same frozen model at update zero under the identical specification "
            "fingerprint and fixed validation protocol."
        ),
        "fixed_constraints": fixed_constraints,
        "independent_variable": (
            "optimizer_update_count_and_equivalent_accumulated_target_token_exposure"
        ),
        "dependent_variables": dependent_variables,
        "minimum_useful_effect": (
            f"Final fixed-validation loss improves by at least {loss_text} natural-log "
            "nats relative to update zero."
        ),
        "failure_threshold": _canonical_json(
            {"common": thresholds["failure"], "profile": thresholds[profile]}
        ),
        "kill_criterion": _canonical_json(thresholds["kill"]),
        "known_limitations": known_limitations,
        "checkpoint_evidence_required": True,
        "generated_output_evidence_required": True,
    }


def _validate_definition_envelope(definition: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "issue",
        "parent_issue",
        "pull_request",
        "manifest_contract",
        "profiles",
    }
    _require(set(definition) == expected_keys, "formal definition top-level fields changed")
    _require(
        definition["schema_version"] == "expertforge-d0-formal-experiment-definition/1",
        "formal definition schema version changed",
    )
    _require(
        definition["status"] == "prospective_not_executed",
        "definition must remain prospective",
    )
    _require(definition["issue"] == 42, "formal definition must bind Issue #42")
    _require(definition["parent_issue"] == 41, "formal definition must bind parent Issue #41")
    _require(definition["pull_request"] == 43, "formal definition must bind draft PR #43")
    expected_manifest_contract = {
        "schema": "expertforge.experiment-manifest",
        "schema_version": 1,
        "format_version": 1,
        "classification": "formal_experiment",
        "validation_fixture_status": "interrupted",
        "validation_fixture_outcome_diagnostic": "handled_interruption",
        "publication_authorized": False,
    }
    _require(
        definition["manifest_contract"] == expected_manifest_contract,
        "manifest compatibility boundary changed",
    )
    profiles = definition["profiles"]
    _require(isinstance(profiles, Mapping), "profiles must be an object")
    _require(set(profiles) == set(_PROFILES), "formal definition profiles changed")


def _synthetic_identity(
    profile: Profile,
    fingerprint: SpecificationFingerprintRecord,
) -> AttemptIdentityRecord:
    run_id, attempt_id = _SYNTHETIC_IDS[profile]
    return AttemptIdentityRecord(
        specification_fingerprint=fingerprint,
        run_id=run_id,
        attempt_id=attempt_id,
        created_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )


def build_validation_manifest(
    contract: Mapping[str, Any],
    profile: Profile,
    profile_definition: Mapping[str, Any],
    fingerprint: SpecificationFingerprintRecord,
) -> ExperimentManifest:
    """Construct a synthetic interrupted manifest entirely in memory."""

    envelope = resolve_config(CONFIG_PATHS[profile])
    config = envelope.config
    if config.d0 is None:
        raise ContractValidationError(f"{profile}: resolved D0 configuration is missing")
    model = contract["models"][profile]

    manifest = ExperimentManifest(
        identity=_synthetic_identity(profile, fingerprint),
        finalized_at_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=1),
        issue_number=42,
        pull_request_number=43,
        maturity_stage="D0",
        research_family="none/not-applicable",
        classification="formal_experiment",
        status="interrupted",
        outcome_diagnostic="handled_interruption",
        known_limitations=tuple(profile_definition["known_limitations"]),
        evidence=EvidenceCompleteness(status="partial", missing=_EXPECTED_MISSING),
        dataset_identity=DatasetReference(
            dataset_id=config.data.dataset_id,
            split=config.data.split,
            immutable_input_name="dataset.manifest",
            content_digest=f"sha256:{contract['dataset']['source_manifest_sha256']}",
            is_fixture=False,
        ),
        tokenizer_identity=TokenizerReference(
            tokenizer_id=config.tokenizer.tokenizer_id,
            vocab_size=config.tokenizer.vocab_size,
            immutable_input_name="tokenizer.manifest",
            content_digest=f"sha256:{contract['tokenizer']['source_manifest_sha256']}",
            is_fixture=False,
        ),
        model_descriptor=ModelIdentity(
            architecture=(
                "decoder-only Transformer; exact causal MHA; pre-norm RMSNorm; "
                "full-head RoPE; SwiGLU; tied embeddings; bias-free; dropout-free"
            ),
            dim=config.model.dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            ffn_dim=config.model.ffn_dim,
            parameter_count=model["trainable_parameters"],
            parameter_count_kind="exact",
        ),
        training_budget=TrainingBudget(
            token_budget=config.training.tokens,
            batch_size=config.training.batch_size,
            seq_len=config.training.seq_len,
            learning_rate=config.training.lr,
            seed=config.training.seed,
            accumulation_factor=config.d0.batch.gradient_accumulation_steps,
        ),
        evaluation_summary=EvaluationSummary(
            eval_interval_tokens=config.evaluation.eval_interval_tokens,
            final_loss=None,
            final_perplexity=None,
            metrics_recorded=tuple(profile_definition["dependent_variables"]),
        ),
        hypothesis=profile_definition["hypothesis"],
        control=profile_definition["control"],
        fixed_constraints=tuple(profile_definition["fixed_constraints"]),
        independent_variable=profile_definition["independent_variable"],
        dependent_variables=tuple(profile_definition["dependent_variables"]),
        minimum_useful_effect=profile_definition["minimum_useful_effect"],
        failure_threshold=profile_definition["failure_threshold"],
        kill_criterion=profile_definition["kill_criterion"],
        checkpoint_evidence_required=profile_definition["checkpoint_evidence_required"],
        generated_output_evidence_required=profile_definition[
            "generated_output_evidence_required"
        ],
    )
    _require(manifest.evidence.missing == _EXPECTED_MISSING, f"{profile}: missing evidence changed")
    _require(manifest.configuration_artifact is None, f"{profile}: synthetic config artifact exists")
    _require(manifest.provenance_artifact is None, f"{profile}: synthetic provenance exists")
    _require(not manifest.telemetry_artifacts, f"{profile}: synthetic telemetry exists")
    _require(not manifest.checkpoint_artifacts, f"{profile}: synthetic checkpoint exists")
    _require(not manifest.generated_output_artifacts, f"{profile}: synthetic output exists")
    encoded = canonical_manifest_bytes(manifest)
    _require(parse_manifest_bytes(encoded) == manifest, f"{profile}: manifest round trip failed")
    return manifest


def validate_profile(
    definition: Mapping[str, Any],
    contract: Mapping[str, Any],
    profile: Profile,
) -> dict[str, Any]:
    profiles = definition["profiles"]
    profile_definition = profiles[profile]
    _require(isinstance(profile_definition, Mapping), f"{profile}: definition must be an object")

    fingerprint = _load_fingerprint(FINGERPRINT_PATHS[profile])
    expected = _expected_profile(contract, profile, fingerprint)
    _require(
        dict(profile_definition) == expected,
        f"{profile}: formal experiment definition disagrees with frozen contract",
    )

    manifest = build_validation_manifest(contract, profile, profile_definition, fingerprint)
    payload = canonical_manifest_bytes(manifest)
    return {
        "profile": profile,
        "classification": manifest.classification,
        "validation_fixture_status": manifest.status,
        "specification_fingerprint": fingerprint.digest_str,
        "manifest_canonical_sha256": hashlib.sha256(payload).hexdigest(),
        "manifest_bytes": len(payload),
        "missing_evidence": list(manifest.evidence.missing),
        "publication_authorized": False,
    }


def validate_definition(definition: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate one prospective definition against all authoritative D0 artifacts."""

    validate_config_bindings()
    contract = load_json_object(CONTRACT_PATH)
    _validate_definition_envelope(definition)

    for text in _walk_strings(definition):
        lowered = text.casefold()
        for placeholder in _FORBIDDEN_PLACEHOLDERS:
            _require(
                placeholder not in lowered,
                f"forbidden placeholder language found: {placeholder!r}",
            )

    reports = {
        "qualification": validate_profile(definition, contract, "qualification"),
        "canonical": validate_profile(definition, contract, "canonical"),
    }
    _require(
        reports["qualification"]["specification_fingerprint"]
        != reports["canonical"]["specification_fingerprint"],
        "profile fingerprints must differ",
    )
    _require(
        reports["qualification"]["manifest_canonical_sha256"]
        != reports["canonical"]["manifest_canonical_sha256"],
        "validation manifests must differ",
    )
    return reports


def validate_all() -> dict[str, dict[str, Any]]:
    return validate_definition(load_json_object(DEFINITION_PATH))


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Validate D0 formal experiment definition")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        report = validate_all()
    except (
        ContractValidationError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        sys.stderr.write(f"D0 FORMAL EXPERIMENT DEFINITION INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
