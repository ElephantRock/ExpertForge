from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_d0_primitive_semantics_evidence import (
    PROPOSAL_PATH,
    REVIEW_PATH,
    STATE_PATH,
    _validate_review,
    _validate_state,
    validate_all,
)
from scripts.validate_d0_source_manifests import ContractValidationError


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _refresh_digest(value: dict[str, object], field: str) -> None:
    payload = dict(value)
    payload.pop(field, None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    value[field] = hashlib.sha256(canonical).hexdigest()


def test_committed_amendment_identity_evidence_is_valid() -> None:
    report = validate_all()
    assert report["status"] == "valid_primitive_semantics_amendment_evidence"
    assert report["proposal_sha256"] == (
        "5bc196942697833c4dfe9f227b16e2b69c6cafd07b05396a70650ecb34cebab0"
    )
    assert report["review_sha256"] == (
        "48961efb67a15f7aae0f714c6120b982cf46154ec9d94170abaa6e1ceb109767"
    )
    assert report["d0_2_acceptance_authorized"] is False
    assert report["d0_3_authorized"] is False


def test_review_fingerprint_mutation_fails_closed_even_with_refreshed_digest() -> None:
    review = copy.deepcopy(_load(REVIEW_PATH))
    identities = review["superseding_identities"]
    assert isinstance(identities, dict)
    qualification = identities["qualification"]
    assert isinstance(qualification, dict)
    qualification["specification_fingerprint"] = "spec-v1-sha256-" + "0" * 64
    _refresh_digest(review, "review_sha256")
    with pytest.raises(ContractValidationError, match="review_sha256 identity changed"):
        _validate_review(review, _load(PROPOSAL_PATH))


def test_review_cannot_authorize_d0_2_acceptance() -> None:
    review = copy.deepcopy(_load(REVIEW_PATH))
    review["d0_2_acceptance_authorized"] = True
    _refresh_digest(review, "review_sha256")
    with pytest.raises(ContractValidationError, match="D0.2 acceptance"):
        _validate_review(review, _load(PROPOSAL_PATH))


def test_project_state_cannot_authorize_d0_3() -> None:
    state = copy.deepcopy(_load(STATE_PATH))
    state["d0_3_authorized"] = True
    _refresh_digest(state, "project_state_sha256")
    with pytest.raises(ContractValidationError, match="D0.3"):
        _validate_state(state)


def test_project_state_digest_mutation_fails_closed() -> None:
    state = copy.deepcopy(_load(STATE_PATH))
    state["d0_2_acceptance_authorized"] = True
    with pytest.raises(ContractValidationError, match="D0.2 acceptance"):
        _validate_state(state)
