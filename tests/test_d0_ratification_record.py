from __future__ import annotations

import copy
import json

import pytest

from scripts.validate_d0_final_review_report import MANIFEST_PATH as FINAL_REVIEW_MANIFEST_PATH
from scripts.validate_d0_ratification_record import (
    RATIFICATION_PATH,
    validate_all,
    validate_ratification_record,
)
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
)


def _record() -> dict[str, object]:
    return dict(load_json_object(RATIFICATION_PATH))


def _contract() -> dict[str, object]:
    return dict(load_json_object(CONTRACT_PATH))


def _final_review() -> dict[str, object]:
    return dict(load_json_object(FINAL_REVIEW_MANIFEST_PATH))


def test_committed_ratification_record_validates() -> None:
    report = validate_all()

    assert report["status"] == "valid_d0_ratification_record"
    assert report["record_sha256"] == "d3cd5c43091d1b68bfb63ec8be896cfa663f735e74f273adbfc9765e4bab8124"
    assert report["accepted_ci_run_number"] == 349
    assert report["d0_1_authorized"] is True
    assert report["d0_2_authorized"] is True
    assert report["d0_3_authorized"] is False
    assert report["material_execution_authorized"] is False


def test_ratification_record_digest_drift_is_rejected() -> None:
    record = copy.deepcopy(_record())
    record["record_sha256"] = "0" * 64

    with pytest.raises(ContractValidationError, match="SHA-256 mismatch"):
        validate_ratification_record(record, _contract(), _final_review())


def test_premature_training_authorization_is_rejected() -> None:
    record = copy.deepcopy(_record())
    record["d0_3_authorized"] = True

    with pytest.raises(ContractValidationError, match="d0_3_authorized must remain false"):
        validate_ratification_record(record, _contract(), _final_review())


def test_unresolved_review_state_is_rejected() -> None:
    record = copy.deepcopy(_record())
    record["all_blocking_review_threads_resolved"] = False

    with pytest.raises(ContractValidationError, match="review remains unresolved"):
        validate_ratification_record(record, _contract(), _final_review())


def test_ratification_record_is_deterministic_json() -> None:
    raw = RATIFICATION_PATH.read_bytes()
    parsed = load_json_object(RATIFICATION_PATH)

    assert raw == (json.dumps(parsed, indent=2, sort_keys=False) + "\n").encode("utf-8")
