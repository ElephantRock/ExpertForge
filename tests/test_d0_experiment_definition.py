from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from expertforge.identity.fingerprint import SpecificationFingerprintRecord
from scripts.validate_d0_config_binding import FINGERPRINT_PATHS
from scripts.validate_d0_experiment_definition import (
    DEFINITION_PATH,
    Profile,
    _EXPECTED_MISSING,
    build_validation_manifest,
    validate_all,
    validate_definition,
)
from scripts.validate_d0_source_manifests import ContractValidationError, load_json_object

ROOT = Path(__file__).resolve().parents[1]
_PROFILES: tuple[Profile, ...] = ("qualification", "canonical")


def _definition() -> dict[str, Any]:
    return copy.deepcopy(load_json_object(DEFINITION_PATH))


def _fingerprint(profile: Profile) -> SpecificationFingerprintRecord:
    record = SpecificationFingerprintRecord.model_validate_json(
        FINGERPRINT_PATHS[profile].read_text(encoding="utf-8")
    )
    record.verify_digest()
    return record


def test_formal_experiment_definition_validates() -> None:
    report = validate_all()

    assert set(report) == {"qualification", "canonical"}
    for profile in _PROFILES:
        assert report[profile]["classification"] == "formal_experiment"
        assert report[profile]["validation_fixture_status"] == "interrupted"
        assert report[profile]["missing_evidence"] == list(_EXPECTED_MISSING)
        assert report[profile]["publication_authorized"] is False
    assert (
        report["qualification"]["specification_fingerprint"]
        != report["canonical"]["specification_fingerprint"]
    )
    assert (
        report["qualification"]["manifest_canonical_sha256"]
        != report["canonical"]["manifest_canonical_sha256"]
    )


def test_validation_manifest_is_partial_interrupted_and_unpublished() -> None:
    definition = _definition()
    contract = load_json_object(ROOT / "experiments/d0/baseline-contract-v1.proposed.json")

    for profile in _PROFILES:
        manifest = build_validation_manifest(
            contract,
            profile,
            definition["profiles"][profile],
            _fingerprint(profile),
        )
        assert manifest.status == "interrupted"
        assert manifest.outcome_diagnostic == "handled_interruption"
        assert manifest.evidence.status == "partial"
        assert manifest.evidence.missing == _EXPECTED_MISSING
        assert manifest.configuration_artifact is None
        assert manifest.provenance_artifact is None
        assert manifest.telemetry_artifacts == ()
        assert manifest.checkpoint_artifacts == ()
        assert manifest.generated_output_artifacts == ()
        assert manifest.result is None
        assert manifest.decision is None


def test_definition_rejects_terminal_or_publication_claims() -> None:
    definition = _definition()
    definition["status"] = "completed"
    with pytest.raises(ContractValidationError, match="remain prospective"):
        validate_definition(definition)

    definition = _definition()
    definition["manifest_contract"]["publication_authorized"] = True
    with pytest.raises(ContractValidationError, match="compatibility boundary"):
        validate_definition(definition)


def test_definition_rejects_missing_formal_field() -> None:
    definition = _definition()
    del definition["profiles"]["qualification"]["hypothesis"]

    with pytest.raises(ContractValidationError, match="disagrees with frozen contract"):
        validate_definition(definition)


def test_definition_rejects_fingerprint_drift_and_profile_crossover() -> None:
    definition = _definition()
    definition["profiles"]["qualification"]["specification_fingerprint"] = (
        "spec-v1-sha256-" + "0" * 64
    )
    with pytest.raises(ContractValidationError, match="disagrees with frozen contract"):
        validate_definition(definition)

    definition = _definition()
    qualification = definition["profiles"]["qualification"]
    canonical = definition["profiles"]["canonical"]
    qualification["specification_fingerprint"], canonical["specification_fingerprint"] = (
        canonical["specification_fingerprint"],
        qualification["specification_fingerprint"],
    )
    with pytest.raises(ContractValidationError, match="disagrees with frozen contract"):
        validate_definition(definition)


def test_definition_rejects_threshold_and_constraint_drift() -> None:
    definition = _definition()
    definition["profiles"]["canonical"]["failure_threshold"] = "{}"
    with pytest.raises(ContractValidationError, match="disagrees with frozen contract"):
        validate_definition(definition)

    definition = _definition()
    definition["profiles"]["qualification"]["fixed_constraints"].append("untracked.constraint=true")
    with pytest.raises(ContractValidationError, match="disagrees with frozen contract"):
        validate_definition(definition)


def test_definition_rejects_placeholder_language() -> None:
    definition = _definition()
    definition["profiles"]["qualification"]["hypothesis"] = "TBD after implementation."

    with pytest.raises(ContractValidationError, match="placeholder"):
        validate_definition(definition)


def test_definition_file_is_deterministic_json() -> None:
    parsed = json.loads(DEFINITION_PATH.read_text(encoding="utf-8"))
    assert DEFINITION_PATH.read_text(encoding="utf-8") == json.dumps(parsed, indent=2) + "\n"


def test_validator_has_no_artifact_publication_path() -> None:
    source = (ROOT / "scripts/validate_d0_experiment_definition.py").read_text(encoding="utf-8")
    assert "ArtifactStore" not in source
    assert "ManifestGenerator" not in source
    assert ".publish(" not in source
