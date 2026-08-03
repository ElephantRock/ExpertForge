from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_d0_primitive_semantics_amendment import (
    AMENDMENT_PATH,
    PROJECT_STATE_PATH,
    validate_all,
    validate_amendment,
)
from scripts.validate_d0_source_manifests import ContractValidationError


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _refresh_digest(amendment: dict[str, object]) -> None:
    payload = dict(amendment)
    payload.pop("amendment_sha256", None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    amendment["amendment_sha256"] = hashlib.sha256(canonical).hexdigest()


def test_committed_primitive_semantics_amendment_is_valid() -> None:
    report = validate_all()
    assert report["status"] == "valid_primitive_semantics_amendment_proposal"
    assert report["amendment_sha256"] == (
        "5bc196942697833c4dfe9f227b16e2b69c6cafd07b05396a70650ecb34cebab0"
    )
    assert report["d0_3_authorized"] is False
    assert report["material_execution_authorized"] is False


def test_rotation_pairing_mutation_fails_even_with_refreshed_digest() -> None:
    amendment = copy.deepcopy(_load(AMENDMENT_PATH))
    semantics = amendment["primitive_semantics"]
    assert isinstance(semantics, dict)
    rope = semantics["rope"]
    assert isinstance(rope, dict)
    rope["pairing"] = "split_half"
    _refresh_digest(amendment)
    with pytest.raises(ContractValidationError, match="primitive numerical semantics"):
        validate_amendment(amendment, _load(PROJECT_STATE_PATH))


def test_historical_mutation_policy_fails_closed() -> None:
    amendment = copy.deepcopy(_load(AMENDMENT_PATH))
    policy = amendment["identity_policy"]
    assert isinstance(policy, dict)
    policy["historical_records_mutated"] = True
    _refresh_digest(amendment)
    with pytest.raises(ContractValidationError, match="identity policy"):
        validate_amendment(amendment, _load(PROJECT_STATE_PATH))


def test_d0_3_authorization_mutation_fails_closed() -> None:
    state = copy.deepcopy(_load(PROJECT_STATE_PATH))
    state["d0_3_authorized"] = True
    with pytest.raises(ContractValidationError, match="D0.3"):
        validate_amendment(_load(AMENDMENT_PATH), state)


def test_amendment_digest_mutation_fails_closed() -> None:
    amendment = copy.deepcopy(_load(AMENDMENT_PATH))
    amendment["scope"] = "changed"
    with pytest.raises(ContractValidationError, match="scope changed"):
        validate_amendment(amendment, _load(PROJECT_STATE_PATH))
