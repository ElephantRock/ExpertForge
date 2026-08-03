"""Validate the non-self-referential D0 primitive-semantics evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from expertforge.config.resolve import canonical_bytes, resolve_config
from expertforge.identity.fingerprint import (
    ImmutableInput,
    SpecificationFingerprint,
    verify_fingerprint,
)
from scripts.materialize_d0_primitive_semantics_amendment import materialize
from scripts.validate_d0_source_manifests import ContractValidationError, load_json_object

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_PATH = ROOT / "experiments/d0/primitive-semantics-amendment-v1.proposed.json"
REVIEW_PATH = ROOT / "experiments/d0/primitive-semantics-review-v1.json"
STATE_PATH = ROOT / "experiments/d0/project-state-v3.proposed.json"
FORMAL_PATH = ROOT / "experiments/d0/formal-experiment-definition-v2.json"

_PROPOSAL_SHA256 = "5bc196942697833c4dfe9f227b16e2b69c6cafd07b05396a70650ecb34cebab0"
_REVIEW_SHA256 = "48961efb67a15f7aae0f714c6120b982cf46154ec9d94170abaa6e1ceb109767"
_STATE_SHA256 = "4f745fca068196dc55c025115db62ea03a23ca15b1e134106ac57294c2ea3c2c"
_DATASET_SHA256 = "85fab524b13c49da78264eb03124ce113395657d6403627e53008aade7eea7d7"
_TOKENIZER_SHA256 = "eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c"

_EXPECTED_PROFILES = {
    "qualification": {
        "config_path": "configs/d0/qualification-primitive-v2.yaml",
        "canonical_config_sha256": (
            "c1f0a4893a56bad1019352c6d534c556d1db0c63e4d7c43c4676c322cc244ea1"
        ),
        "fingerprint_path": "experiments/d0/qualification-fingerprint-v2.json",
        "specification_fingerprint": (
            "spec-v1-sha256-8f1390a73ddad0f22d2354a2b45a42c5563c448b08ee317dd80e7d0ac83dfe91"
        ),
    },
    "canonical": {
        "config_path": "configs/d0/canonical-primitive-v2.yaml",
        "canonical_config_sha256": (
            "49def2ad8500bde97c9aac792bf498f03aed062cc2a4120d1877491822544506"
        ),
        "fingerprint_path": "experiments/d0/canonical-fingerprint-v2.json",
        "specification_fingerprint": (
            "spec-v1-sha256-edd02b2b7b130caaeb05b653729f9952a53f64f149204a602a3f657a4f05edfb"
        ),
    },
}
_EXPECTED_ACCEPTANCE_BLOCKERS = [
    "amendment_review_not_yet_accepted",
    "exact_head_ci_not_yet_accepted",
]


def _canonical_json_bytes(value: object) -> bytes:
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


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"{field} must be an array")
    resolved = list(value)
    _require(all(isinstance(item, str) for item in resolved), f"{field} must contain strings")
    return resolved


def _validate_self_digest(
    value: Mapping[str, Any],
    *,
    field: str,
    policy: str,
    expected: str,
) -> None:
    _require(value.get("digest_policy") == policy, f"{field} digest policy changed")
    declared = value.get(field)
    _require(declared == expected, f"{field} identity changed")
    payload = dict(value)
    payload.pop(field, None)
    actual = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    _require(actual == expected, f"{field} digest mismatch")


def _immutable_inputs() -> tuple[ImmutableInput, ...]:
    return (
        ImmutableInput(name="dataset.manifest", algorithm="sha256", digest=_DATASET_SHA256),
        ImmutableInput(
            name="primitive.semantics.amendment",
            algorithm="sha256",
            digest=_PROPOSAL_SHA256,
        ),
        ImmutableInput(
            name="tokenizer.manifest",
            algorithm="sha256",
            digest=_TOKENIZER_SHA256,
        ),
    )


def _validate_profile(
    profile: str,
    expected: Mapping[str, str],
    proposal_semantics: Mapping[str, Any],
) -> dict[str, str]:
    config_path = ROOT / expected["config_path"]
    fingerprint_path = ROOT / expected["fingerprint_path"]
    envelope = resolve_config(config_path)
    semantics = envelope.effective.d0_primitive_semantics
    _require(semantics is not None, f"{profile} primitive semantics are missing")
    _require(
        semantics.amendment_path == "experiments/d0/primitive-semantics-amendment-v1.proposed.json",
        f"{profile} amendment path changed",
    )
    _require(
        semantics.amendment_sha256 == _PROPOSAL_SHA256,
        f"{profile} amendment digest changed",
    )
    resolved_semantics = {
        "rope": semantics.rope.model_dump(mode="json"),
        "attention": semantics.attention.model_dump(mode="json"),
        "rmsnorm": semantics.rmsnorm.model_dump(mode="json"),
        "swiglu": semantics.swiglu.model_dump(mode="json"),
    }
    _require(
        resolved_semantics == dict(proposal_semantics),
        f"{profile} semantics differ from proposal",
    )

    config_bytes = canonical_bytes(envelope)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    _require(
        config_sha256 == expected["canonical_config_sha256"],
        f"{profile} canonical configuration digest changed",
    )
    fingerprint = SpecificationFingerprint.model_validate(load_json_object(fingerprint_path))
    verify_fingerprint(fingerprint, config_bytes, _immutable_inputs())
    _require(
        fingerprint.digest_str == expected["specification_fingerprint"],
        f"{profile} specification fingerprint changed",
    )
    return {
        "canonical_config_sha256": config_sha256,
        "specification_fingerprint": fingerprint.digest_str,
    }


def _validate_review(
    review: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(
        review.get("schema_version") == "expertforge-d0-primitive-semantics-review/1",
        "primitive-semantics review schema changed",
    )
    _require(
        review.get("status") == "preparation_complete_pending_acceptance",
        "primitive-semantics review status changed",
    )
    _require(
        review.get("issue") == 48 and review.get("pull_request") == 50,
        "review binding changed",
    )
    _require(
        review.get("proposal_sha256") == _PROPOSAL_SHA256,
        "review proposal identity changed",
    )
    _require(
        review.get("materialization_command")
        == "uv run --locked python -m scripts.materialize_d0_primitive_semantics_amendment --check",
        "review materialization command changed",
    )
    _require(review.get("historical_records_mutated") is False, "historical records were mutated")
    _require(
        review.get("prior_d0_model_attempts_invalidated") is False,
        "review claims prior D0 attempts were invalidated",
    )
    _require(
        _string_list(review.get("preparation_blockers_resolved"), "preparation_blockers_resolved")
        == ["affected_profile_fingerprints_not_yet_regenerated"],
        "resolved amendment preparation blockers changed",
    )
    _require(
        _string_list(review.get("acceptance_blockers_remaining"), "acceptance_blockers_remaining")
        == _EXPECTED_ACCEPTANCE_BLOCKERS,
        "remaining amendment acceptance blockers changed",
    )
    _require(
        review.get("d0_2_acceptance_authorized") is False,
        "D0.2 acceptance was authorized",
    )
    _require(review.get("d0_3_authorized") is False, "D0.3 was authorized")
    _require(
        review.get("material_execution_authorized") is False,
        "material execution was authorized",
    )
    _validate_self_digest(
        review,
        field="review_sha256",
        policy="sha256(canonical_json_without_review_sha256)",
        expected=_REVIEW_SHA256,
    )
    proposal_blockers = _string_list(proposal.get("ratification_blockers"), "ratification_blockers")
    _require(
        "affected_profile_fingerprints_not_yet_regenerated" in proposal_blockers,
        "historical proposal preparation blocker was rewritten",
    )
    return _mapping(review.get("superseding_identities"), "superseding_identities")


def _validate_state(state: Mapping[str, Any]) -> None:
    _require(
        state.get("schema_version") == "expertforge-d0-project-state/3-proposal",
        "amended project-state schema changed",
    )
    _require(
        state.get("status") == "ratified_with_pending_primitive_semantics_amendment",
        "amended project-state status changed",
    )
    _require(
        state.get("primitive_semantics_amendment_sha256") == _PROPOSAL_SHA256,
        "state proposal identity changed",
    )
    _require(
        state.get("historical_records_mutated") is False,
        "state claims historical mutation",
    )
    _require(state.get("d0_1_authorized") is True, "D0.1 authorization changed")
    _require(
        state.get("d0_2_implementation_authorized") is True,
        "D0.2 implementation changed",
    )
    _require(
        state.get("d0_2_acceptance_authorized") is False,
        "D0.2 acceptance must remain blocked",
    )
    _require(state.get("d0_3_authorized") is False, "D0.3 must remain blocked")
    _require(
        state.get("material_execution_authorized") is False,
        "material execution must remain blocked",
    )
    _require(
        _string_list(state.get("remaining_amendment_blockers"), "remaining_amendment_blockers")
        == _EXPECTED_ACCEPTANCE_BLOCKERS,
        "state amendment blockers changed",
    )
    _validate_self_digest(
        state,
        field="project_state_sha256",
        policy="sha256(canonical_json_without_project_state_sha256)",
        expected=_STATE_SHA256,
    )


def _validate_formal(formal: Mapping[str, Any]) -> None:
    _require(
        formal.get("schema_version") == "expertforge-d0-formal-experiment-definition/2",
        "amended formal-definition schema changed",
    )
    _require(
        formal.get("issue") == 48 and formal.get("pull_request") == 50,
        "formal binding changed",
    )
    _require(
        formal.get("primitive_semantics_amendment_sha256") == _PROPOSAL_SHA256,
        "formal proposal identity changed",
    )
    _require(
        formal.get("prior_d0_model_attempts_invalidated") is False,
        "formal definition invalidates prior attempts",
    )
    profiles = _mapping(formal.get("profiles"), "formal.profiles")
    for profile, expected in _EXPECTED_PROFILES.items():
        entry = _mapping(profiles.get(profile), f"formal.profiles.{profile}")
        _require(
            entry.get("config_path") == expected["config_path"],
            f"{profile} formal config path changed",
        )
        _require(
            entry.get("fingerprint_path") == expected["fingerprint_path"],
            f"{profile} formal fingerprint path changed",
        )
        _require(
            entry.get("specification_fingerprint") == expected["specification_fingerprint"],
            f"{profile} formal fingerprint changed",
        )


def validate_all() -> dict[str, object]:
    """Validate proposal-derived identities without introducing a digest cycle."""

    proposal = load_json_object(PROPOSAL_PATH)
    review = load_json_object(REVIEW_PATH)
    state = load_json_object(STATE_PATH)
    formal = load_json_object(FORMAL_PATH)

    _require(
        proposal.get("amendment_sha256") == _PROPOSAL_SHA256,
        "proposal digest changed",
    )
    identities = _validate_review(review, proposal)
    _validate_state(state)
    _validate_formal(formal)
    proposal_semantics = _mapping(proposal.get("primitive_semantics"), "primitive_semantics")

    profile_reports: dict[str, dict[str, str]] = {}
    for profile, expected in _EXPECTED_PROFILES.items():
        _require(
            dict(_mapping(identities.get(profile), f"superseding_identities.{profile}"))
            == expected,
            f"{profile} review identities changed",
        )
        profile_reports[profile] = _validate_profile(profile, expected, proposal_semantics)

    _require(
        identities.get("formal_experiment_definition_path")
        == "experiments/d0/formal-experiment-definition-v2.json",
        "review formal-definition path changed",
    )
    _require(
        identities.get("project_state_path") == "experiments/d0/project-state-v3.proposed.json",
        "review project-state path changed",
    )
    _require(
        identities.get("project_state_sha256") == _STATE_SHA256,
        "review state identity changed",
    )

    materialize(check=True)
    return {
        "status": "valid_primitive_semantics_amendment_evidence",
        "proposal_sha256": _PROPOSAL_SHA256,
        "review_sha256": _REVIEW_SHA256,
        "project_state_sha256": _STATE_SHA256,
        "profiles": profile_reports,
        "remaining_acceptance_blockers": list(_EXPECTED_ACCEPTANCE_BLOCKERS),
        "remaining_acceptance_blocker_count": len(_EXPECTED_ACCEPTANCE_BLOCKERS),
        "historical_records_mutated": False,
        "d0_2_acceptance_authorized": False,
        "d0_3_authorized": False,
        "material_execution_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


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
        sys.stderr.write(f"D0 PRIMITIVE-SEMANTICS EVIDENCE INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
