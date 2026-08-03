from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_d0_final_review_report import MANIFEST_PATH as FINAL_REVIEW_MANIFEST_PATH
from scripts.validate_d0_project_state import (
    MANIFEST_PATH,
    PROJECT_STATE_PATH,
    validate_all,
    validate_project_state,
)
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
)

ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> dict[str, object]:
    return dict(load_json_object(MANIFEST_PATH))


def _contract() -> dict[str, object]:
    return dict(load_json_object(ROOT / CONTRACT_PATH))


def _final_review_manifest() -> dict[str, object]:
    return dict(load_json_object(FINAL_REVIEW_MANIFEST_PATH))


def test_committed_project_state_validates() -> None:
    report = validate_all()

    assert report["status"] == "valid_d0_project_state"
    assert report["required_heading_count"] == 12
    assert report["remaining_ratification_blockers"] == []
    assert report["remaining_ratification_blocker_count"] == 0
    assert report["d0_1_authorized"] is False
    assert report["d0_2_authorized"] is False
    assert report["material_execution_authorized"] is False


def test_project_state_digest_drift_is_rejected() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["project_state_sha256"] = "0" * 64

    with pytest.raises(ContractValidationError, match="SHA-256 mismatch"):
        validate_project_state(
            manifest,
            PROJECT_STATE_PATH.read_bytes(),
            _contract(),
            _final_review_manifest(),
        )


def test_project_state_claim_drift_is_rejected() -> None:
    manifest = copy.deepcopy(_manifest())
    state_bytes = PROJECT_STATE_PATH.read_bytes() + b"\nmaterial execution authorized: true\n"
    manifest["project_state_sha256"] = hashlib.sha256(state_bytes).hexdigest()

    with pytest.raises(ContractValidationError, match="forbidden claim"):
        validate_project_state(
            manifest,
            state_bytes,
            _contract(),
            _final_review_manifest(),
        )


def test_nonterminal_contract_blocker_state_is_rejected() -> None:
    contract = copy.deepcopy(_contract())
    contract["ratification_blockers"] = ["PROJECT_STATE_synchronization"]

    with pytest.raises(ContractValidationError, match="not terminal"):
        validate_project_state(
            _manifest(),
            PROJECT_STATE_PATH.read_bytes(),
            contract,
            _final_review_manifest(),
        )


def test_project_state_manifest_is_deterministic_json() -> None:
    raw = MANIFEST_PATH.read_bytes()
    parsed = load_json_object(MANIFEST_PATH)

    assert raw == (json.dumps(parsed, indent=2, sort_keys=False) + "\n").encode("utf-8")
