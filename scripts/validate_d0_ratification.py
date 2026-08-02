"""Run every D0.0 ratification validator as one fail-closed command."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.validate_d0_config_binding import ROOT, validate_all as validate_config_bindings
from scripts.validate_d0_contract import validate_contract
from scripts.validate_d0_experiment_definition import validate_all as validate_experiment_definition
from scripts.validate_d0_generation_prompts import (
    PROMPT_MANIFEST_PATH,
    load_prompt_manifest,
    validate_generation_prompts,
)
from scripts.validate_d0_parameter_inventory import validate_all as validate_parameter_inventory
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    DATASET_MANIFEST_PATH,
    TOKENIZER_MANIFEST_PATH,
    ContractValidationError,
    load_json_object,
    validate_source_manifests,
)

_SCHEMA_VERSION = "expertforge-d0-ratification-validation/1"
_EXPECTED_VALIDATOR_ORDER = (
    "baseline_contract",
    "source_manifests",
    "configuration_bindings",
    "formal_experiment_definition",
    "parameter_inventory",
    "generation_prompts_and_contamination",
)
_EXPECTED_REMAINING_BLOCKERS = (
    "rendered_review_report",
    "PROJECT_STATE_synchronization",
)

ValidationReport = Mapping[str, Any]
Validator = Callable[[], ValidationReport]


@dataclass(frozen=True)
class ValidationStep:
    """One named validator in the permanent D0.0 gate."""

    name: str
    validator: Validator


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _contract_path() -> Path:
    return ROOT / CONTRACT_PATH


def _validate_baseline_contract() -> ValidationReport:
    return validate_contract(load_json_object(_contract_path()))


def _validate_source_manifests() -> ValidationReport:
    return validate_source_manifests(
        load_json_object(_contract_path()),
        load_json_object(ROOT / TOKENIZER_MANIFEST_PATH),
        load_json_object(ROOT / DATASET_MANIFEST_PATH),
    )


def _validate_configuration_bindings() -> ValidationReport:
    return validate_config_bindings()


def _validate_formal_experiment_definition() -> ValidationReport:
    return validate_experiment_definition()


def _validate_parameter_inventory() -> ValidationReport:
    return validate_parameter_inventory()


def _validate_generation_prompt_contract() -> ValidationReport:
    contract = load_json_object(_contract_path())
    manifest, raw = load_prompt_manifest(PROMPT_MANIFEST_PATH)
    return validate_generation_prompts(contract, manifest, raw)


VALIDATION_STEPS: tuple[ValidationStep, ...] = (
    ValidationStep("baseline_contract", _validate_baseline_contract),
    ValidationStep("source_manifests", _validate_source_manifests),
    ValidationStep("configuration_bindings", _validate_configuration_bindings),
    ValidationStep("formal_experiment_definition", _validate_formal_experiment_definition),
    ValidationStep("parameter_inventory", _validate_parameter_inventory),
    ValidationStep(
        "generation_prompts_and_contamination",
        _validate_generation_prompt_contract,
    ),
)


def _validate_step_registry(steps: Sequence[ValidationStep]) -> None:
    names = tuple(step.name for step in steps)
    _require(names == _EXPECTED_VALIDATOR_ORDER, "D0 ratification validator registry changed")
    _require(len(set(names)) == len(names), "D0 ratification validator names must be unique")
    for step in steps:
        _require(bool(step.name.strip()), "D0 ratification validator name is blank")
        _require(
            callable(step.validator), f"D0 ratification validator {step.name!r} is not callable"
        )


def _validate_remaining_blockers(contract: Mapping[str, Any]) -> tuple[str, ...]:
    blockers = contract.get("ratification_blockers")
    if not isinstance(blockers, list):
        raise ContractValidationError("ratification_blockers must be a list")
    _require(
        all(isinstance(blocker, str) for blocker in blockers),
        "ratification blockers must be strings",
    )
    resolved = tuple(blockers)
    _require(
        resolved == _EXPECTED_REMAINING_BLOCKERS,
        "D0 ratification blockers disagree with permanent gate state",
    )
    return resolved


def validate_all(
    steps: Sequence[ValidationStep] = VALIDATION_STEPS,
) -> dict[str, object]:
    """Execute the complete D0.0 validation bundle and return a stable report."""

    _validate_step_registry(steps)
    contract = load_json_object(_contract_path())
    blockers = _validate_remaining_blockers(contract)

    validator_reports: dict[str, dict[str, object]] = {}
    aggregate_payload: list[dict[str, str]] = []
    for step in steps:
        report = step.validator()
        _require(
            isinstance(report, Mapping),
            f"D0 ratification validator {step.name!r} returned a non-mapping report",
        )
        canonical_report = _canonical_bytes(report)
        report_sha256 = hashlib.sha256(canonical_report).hexdigest()
        validator_reports[step.name] = {
            "status": "passed",
            "report_sha256": report_sha256,
        }
        aggregate_payload.append({"name": step.name, "report_sha256": report_sha256})

    aggregate_sha256 = hashlib.sha256(_canonical_bytes(aggregate_payload)).hexdigest()
    return {
        "status": "valid_d0_ratification_bundle",
        "schema_version": _SCHEMA_VERSION,
        "validator_count": len(steps),
        "validator_order": list(_EXPECTED_VALIDATOR_ORDER),
        "validators": validator_reports,
        "aggregate_report_sha256": aggregate_sha256,
        "remaining_ratification_blockers": list(blockers),
        "remaining_ratification_blocker_count": len(blockers),
        "actual_corpus_scan_completed": False,
        "actual_corpus_scan_stage": "D0.1_preflight_before_packing_or_training",
        "material_execution_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Validate the complete D0.0 ratification bundle")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        report = validate_all()
    except (
        ContractValidationError,
        KeyError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        sys.stderr.write(f"D0 RATIFICATION BUNDLE INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
