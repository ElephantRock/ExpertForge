from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_d0_final_review_report import (
    MANIFEST_PATH,
    validate_all,
    validate_final_review_report,
)
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports/d0-baseline-contract-final-review.md"


def _manifest() -> dict[str, object]:
    return dict(load_json_object(MANIFEST_PATH))


def _contract() -> dict[str, object]:
    return dict(load_json_object(ROOT / CONTRACT_PATH))


def test_committed_final_review_report_validates() -> None:
    report = validate_all()

    assert report["status"] == "valid_d0_final_review_report"
    assert report["required_heading_count"] == 16
    assert report["remaining_ratification_blockers"] == ["PROJECT_STATE_synchronization"]
    assert report["material_execution_authorized"] is False


def test_report_digest_drift_is_rejected() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["report_sha256"] = "0" * 64

    with pytest.raises(ContractValidationError, match="SHA-256 mismatch"):
        validate_final_review_report(manifest, REPORT_PATH.read_bytes(), _contract())


def test_report_content_drift_is_rejected() -> None:
    manifest = copy.deepcopy(_manifest())
    report_bytes = REPORT_PATH.read_bytes() + b"\nmaterial execution authorized: true\n"
    manifest["report_sha256"] = hashlib.sha256(report_bytes).hexdigest()

    with pytest.raises(ContractValidationError, match="forbidden claim"):
        validate_final_review_report(manifest, report_bytes, _contract())


def test_stale_contract_blocker_state_is_rejected() -> None:
    contract = copy.deepcopy(_contract())
    contract["ratification_blockers"] = [
        "rendered_review_report",
        "PROJECT_STATE_synchronization",
    ]

    with pytest.raises(ContractValidationError, match="blocker state"):
        validate_final_review_report(_manifest(), REPORT_PATH.read_bytes(), contract)


def test_review_manifest_is_deterministic_json() -> None:
    raw = MANIFEST_PATH.read_bytes()
    parsed = load_json_object(MANIFEST_PATH)

    assert raw == (json.dumps(parsed, indent=2, sort_keys=False) + "\n").encode("utf-8")
