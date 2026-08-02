"""Validate D0 resolved configurations against the proposed execution contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from expertforge.config.models import ConfigRoot
from expertforge.config.resolve import canonical_bytes, resolve_config
from expertforge.identity.fingerprint import (
    ImmutableInput,
    SpecificationFingerprintRecord,
    verify_fingerprint,
)
from scripts.validate_d0_source_manifests import ContractValidationError, load_json_object

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments/d0/baseline-contract-v1.proposed.json"
CONFIG_PATHS = {
    "qualification": ROOT / "configs/d0/qualification.yaml",
    "canonical": ROOT / "configs/d0/canonical.yaml",
}
FINGERPRINT_PATHS = {
    "qualification": ROOT / "experiments/d0/qualification-fingerprint.json",
    "canonical": ROOT / "experiments/d0/canonical-fingerprint.json",
}
Profile = Literal["qualification", "canonical"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _expected_d0_payload(contract: Mapping[str, Any], profile: Profile) -> dict[str, Any]:
    tokenizer = contract["tokenizer"]
    dataset = contract["dataset"]
    model = contract["models"][profile]
    parameter_accounting = contract["parameter_accounting"]
    profile_thresholds = contract["thresholds"][profile]
    failure = contract["thresholds"]["failure"]
    kill = contract["thresholds"]["kill"]

    return {
        "profile": profile,
        "sources": {
            "contract_path": "experiments/d0/baseline-contract-v1.proposed.json",
            "dataset_repository": dataset["repository"],
            "dataset_revision": dataset["revision"],
            "dataset_configuration": dataset["configuration"],
            "dataset_manifest_path": dataset["source_manifest_path"],
            "dataset_manifest_sha256": dataset["source_manifest_sha256"],
            "tokenizer_repository": tokenizer["repository"],
            "tokenizer_revision": tokenizer["revision"],
            "tokenizer_manifest_path": tokenizer["source_manifest_path"],
            "tokenizer_manifest_sha256": tokenizer["source_manifest_sha256"],
        },
        "architecture": dict(contract["architecture"]),
        "sequence": dict(contract["sequence_semantics"]),
        "model": {
            "model_id": model["model_id"],
            "head_dimension": model["head_dimension"],
            "trainable_parameters": model["trainable_parameters"],
            "non_trainable_parameters": model["non_trainable_parameters"],
            "parameter_formula": parameter_accounting["formula"],
            "swiglu_rounding_rule": parameter_accounting["swiglu_rounding_rule"],
            "rope_trainable_parameters": parameter_accounting["rope_trainable_parameters"],
            "separate_output_head_parameters": parameter_accounting[
                "separate_output_head_parameters"
            ],
        },
        "batch": dict(contract["batch"]),
        "optimizer": dict(contract["optimizer"]),
        "initialization": dict(contract["initialization"]),
        "schedule": dict(contract["schedules"][profile]),
        "precision": dict(contract["precision_and_environment"]),
        "seeds": dict(contract["seeds"]),
        "evaluation": dict(contract["evaluation"]),
        "thresholds": {
            "minimum_final_validation_loss_improvement_nats": profile_thresholds[
                "minimum_final_validation_loss_improvement_nats"
            ],
            "maximum_final_loss_above_best_prior_nats": profile_thresholds.get(
                "maximum_final_loss_above_best_prior_nats"
            ),
            "maximum_consecutive_regressing_validation_boundaries": profile_thresholds.get(
                "maximum_consecutive_regressing_validation_boundaries"
            ),
            "regression_boundary_delta_nats": profile_thresholds.get(
                "regression_boundary_delta_nats"
            ),
            "minimum_final_to_initial_throughput_ratio": profile_thresholds.get(
                "minimum_final_to_initial_throughput_ratio"
            ),
            "maximum_skipped_updates": profile_thresholds["maximum_skipped_updates"],
            "maximum_peak_device_memory_fraction": profile_thresholds[
                "maximum_peak_device_memory_fraction"
            ],
            "requires_exact_checkpoint_round_trip": profile_thresholds.get(
                "requires_exact_checkpoint_round_trip"
            ),
            "requires_locked_environment_resume_equality": profile_thresholds.get(
                "requires_locked_environment_resume_equality"
            ),
            "maximum_checkpoint_write_seconds": failure["maximum_checkpoint_write_seconds"],
            "maximum_checkpoint_read_seconds": failure["maximum_checkpoint_read_seconds"],
            "maximum_peak_host_memory_fraction": failure[
                "maximum_peak_host_memory_fraction"
            ],
            "minimum_loss_improvement_at_quarter_budget_nats": failure[
                "minimum_loss_improvement_at_quarter_budget_nats"
            ],
            "maximum_rejected_recovery_attempts_before_kill": failure[
                "maximum_rejected_recovery_attempts_before_kill"
            ],
            "kill_on_any_non_finite_value": kill["any_non_finite_value"],
            "kill_on_any_skipped_optimizer_update": kill["any_skipped_optimizer_update"],
            "kill_on_any_checkpoint_or_resume_state_mismatch": kill[
                "any_checkpoint_or_resume_state_mismatch"
            ],
            "maximum_out_of_memory_failures_after_remediation": kill[
                "maximum_out_of_memory_failures_after_remediation"
            ],
            "maximum_failed_recovery_attempts": kill["maximum_failed_recovery_attempts"],
            "minimum_loss_improvement_at_half_budget_nats": kill[
                "minimum_loss_improvement_at_half_budget_nats"
            ],
        },
    }


def _immutable_inputs(config: ConfigRoot) -> tuple[ImmutableInput, ImmutableInput]:
    if config.d0 is None:
        raise ContractValidationError("D0 configuration section is missing")
    return (
        ImmutableInput(
            name="dataset.manifest",
            algorithm="sha256",
            digest=config.d0.sources.dataset_manifest_sha256,
        ),
        ImmutableInput(
            name="tokenizer.manifest",
            algorithm="sha256",
            digest=config.d0.sources.tokenizer_manifest_sha256,
        ),
    )


def _validate_core_binding(
    config: ConfigRoot,
    contract: Mapping[str, Any],
    profile: Profile,
) -> None:
    if config.d0 is None:
        raise ContractValidationError("D0 configuration section is missing")
    model = contract["models"][profile]
    schedule = contract["schedules"][profile]
    batch = contract["batch"]
    tokenizer = contract["tokenizer"]
    dataset = contract["dataset"]
    sequence = contract["sequence_semantics"]
    thresholds = contract["thresholds"][profile]

    _require(config.d0.profile == profile, f"{profile}: profile mismatch")
    _require(
        config.data.dataset_id
        == f"{dataset['repository']}@{dataset['revision']}:{dataset['configuration']}",
        f"{profile}: data.dataset_id mismatch",
    )
    _require(config.data.split == "train", f"{profile}: data.split mismatch")
    _require(
        config.data.seq_len == sequence["model_context_tokens"],
        f"{profile}: data.seq_len mismatch",
    )
    _require(
        config.tokenizer.tokenizer_id
        == f"{tokenizer['repository']}@{tokenizer['revision']}",
        f"{profile}: tokenizer.tokenizer_id mismatch",
    )
    _require(
        config.tokenizer.vocab_size == tokenizer["vocabulary_size"],
        f"{profile}: tokenizer.vocab_size mismatch",
    )
    _require(config.model.dim == model["model_width"], f"{profile}: model.dim mismatch")
    _require(
        config.model.n_layers == model["layers"],
        f"{profile}: model.n_layers mismatch",
    )
    _require(
        config.model.n_heads == model["attention_heads"],
        f"{profile}: model.n_heads mismatch",
    )
    _require(
        config.model.ffn_dim == model["swiglu_intermediate_width"],
        f"{profile}: model.ffn_dim mismatch",
    )
    _require(config.training.seed == contract["seeds"]["master_seed"], f"{profile}: seed mismatch")
    _require(
        config.training.tokens == schedule["training_target_tokens"],
        f"{profile}: training.tokens mismatch",
    )
    _require(
        config.training.batch_size == batch["global_sequences_per_update"],
        f"{profile}: training.batch_size mismatch",
    )
    _require(
        config.training.seq_len == sequence["model_context_tokens"],
        f"{profile}: training.seq_len mismatch",
    )
    _require(
        config.training.lr == schedule["peak_learning_rate"],
        f"{profile}: training.lr mismatch",
    )
    _require(
        config.evaluation.eval_interval_tokens
        == schedule["validation_interval_updates"] * batch["target_tokens_per_update"],
        f"{profile}: evaluation cadence mismatch",
    )
    _require(
        config.evaluation.loss_improvement_threshold
        == thresholds["minimum_final_validation_loss_improvement_nats"],
        f"{profile}: evaluation threshold mismatch",
    )
    _require(
        config.checkpointing.interval_tokens
        == schedule["checkpoint_interval_updates"] * batch["target_tokens_per_update"],
        f"{profile}: checkpoint cadence mismatch",
    )
    _require(
        config.hardware.device == "cuda" and config.hardware.dtype == "bf16",
        f"{profile}: hardware precision mismatch",
    )


def validate_profile(
    contract: Mapping[str, Any],
    profile: Profile,
    *,
    config_path: Path | None = None,
    fingerprint_path: Path | None = None,
) -> dict[str, str]:
    """Validate one D0 profile and its committed fingerprint record."""

    resolved_path = config_path or CONFIG_PATHS[profile]
    resolved_fingerprint_path = fingerprint_path or FINGERPRINT_PATHS[profile]
    envelope = resolve_config(resolved_path)
    if envelope.config.d0 is None:
        raise ContractValidationError(f"{profile}: D0 section is missing")

    expected = _expected_d0_payload(contract, profile)
    actual = envelope.config.d0.model_dump(mode="json")
    _require(actual == expected, f"{profile}: resolved D0 section disagrees with contract")
    _validate_core_binding(envelope.config, contract, profile)

    fingerprint_value = load_json_object(resolved_fingerprint_path)
    record = SpecificationFingerprintRecord.model_validate(fingerprint_value)
    inputs = _immutable_inputs(envelope.config)
    verify_fingerprint(record, canonical_bytes(envelope), inputs)

    return {
        "profile": profile,
        "config_path": resolved_path.relative_to(ROOT).as_posix()
        if resolved_path.is_relative_to(ROOT)
        else resolved_path.as_posix(),
        "source_sha256": envelope.content_hash,
        "canonical_config_sha256": record.canonical_config.digest,
        "specification_fingerprint": record.digest_str,
    }


def validate_all() -> dict[str, dict[str, str]]:
    """Validate both qualification and canonical D0 configurations."""

    contract = load_json_object(CONTRACT_PATH)
    reports = {
        "qualification": validate_profile(contract, "qualification"),
        "canonical": validate_profile(contract, "canonical"),
    }
    _require(
        reports["qualification"]["specification_fingerprint"]
        != reports["canonical"]["specification_fingerprint"],
        "qualification and canonical fingerprints must differ",
    )
    return reports


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Validate D0 configuration bindings")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        report = validate_all()
    except (
        ContractValidationError,
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        sys.stderr.write(f"D0 CONFIG BINDING INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
