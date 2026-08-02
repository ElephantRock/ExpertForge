from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from scripts.validate_d0_ratification import (
    VALIDATION_STEPS,
    ValidationStep,
    Validator,
    _validate_remaining_blockers,
    main,
    validate_all,
)
from scripts.validate_d0_source_manifests import ContractValidationError

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ORDER = (
    "baseline_contract",
    "source_manifests",
    "configuration_bindings",
    "formal_experiment_definition",
    "parameter_inventory",
    "generation_prompts_and_contamination",
)


def _passing_steps(recorded: list[str] | None = None) -> tuple[ValidationStep, ...]:
    steps: list[ValidationStep] = []
    for name in EXPECTED_ORDER:

        def validate(current_name: str = name) -> Mapping[str, object]:
            if recorded is not None:
                recorded.append(current_name)
            return {"validator": current_name, "status": "passed"}

        steps.append(ValidationStep(name, validate))
    return tuple(steps)


def test_committed_ratification_bundle_validates() -> None:
    report = validate_all()

    assert report["status"] == "valid_d0_ratification_bundle"
    assert report["validator_count"] == 6
    assert report["validator_order"] == list(EXPECTED_ORDER)
    assert report["remaining_ratification_blockers"] == [
        "rendered_review_report",
        "PROJECT_STATE_synchronization",
    ]
    assert report["remaining_ratification_blocker_count"] == 2
    assert report["actual_corpus_scan_completed"] is False
    assert report["actual_corpus_scan_stage"] == "D0.1_preflight_before_packing_or_training"
    assert report["material_execution_authorized"] is False


def test_every_registered_validator_runs_once_in_order() -> None:
    recorded: list[str] = []

    report = validate_all(_passing_steps(recorded))

    assert recorded == list(EXPECTED_ORDER)
    assert list(report["validators"]) == list(EXPECTED_ORDER)


def test_validator_registry_is_exact_and_immutable() -> None:
    assert tuple(step.name for step in VALIDATION_STEPS) == EXPECTED_ORDER

    with pytest.raises(ContractValidationError, match="registry changed"):
        validate_all(tuple(reversed(_passing_steps())))

    with pytest.raises(ContractValidationError, match="registry changed"):
        validate_all(_passing_steps()[:-1])


def test_validator_failure_stops_later_steps() -> None:
    recorded: list[str] = []
    steps = list(_passing_steps(recorded))

    def fail() -> Mapping[str, object]:
        recorded.append("formal_experiment_definition")
        raise ContractValidationError("synthetic validator failure")

    steps[3] = ValidationStep("formal_experiment_definition", fail)

    with pytest.raises(ContractValidationError, match="synthetic validator failure"):
        validate_all(tuple(steps))

    assert recorded == [
        "baseline_contract",
        "source_manifests",
        "configuration_bindings",
        "formal_experiment_definition",
    ]


def test_non_mapping_validator_report_is_rejected() -> None:
    steps = list(_passing_steps())
    invalid = cast(Validator, lambda: ["not", "a", "mapping"])
    steps[2] = ValidationStep("configuration_bindings", invalid)

    with pytest.raises(ContractValidationError, match="non-mapping report"):
        validate_all(tuple(steps))


def test_aggregate_report_digest_is_deterministic() -> None:
    first = validate_all(_passing_steps())
    second = validate_all(_passing_steps())

    assert first == second
    assert len(str(first["aggregate_report_sha256"])) == 64


def test_remaining_blocker_drift_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="permanent gate state"):
        _validate_remaining_blockers(
            {
                "ratification_blockers": [
                    "contract_validation_command_and_CI_gate",
                    "rendered_review_report",
                    "PROJECT_STATE_synchronization",
                ]
            }
        )


def test_permanent_command_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    captured = capsys.readouterr()

    assert "valid_d0_ratification_bundle" in captured.out
    assert captured.err == ""


def test_ci_invokes_exact_permanent_gate_command() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    command = "uv run --locked python scripts/validate_d0_ratification.py > /dev/null"

    assert "- name: Validate D0 ratification bundle" in workflow
    assert workflow.count(command) == 1
