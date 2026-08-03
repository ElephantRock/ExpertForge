"""Validate the immutable D0 primitive-semantics amendment proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.validate_d0_source_manifests import ContractValidationError, load_json_object

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = ROOT / "experiments/d0/primitive-semantics-amendment-v1.proposed.json"
PROJECT_STATE_PATH = ROOT / "experiments/d0/project-state-v2.json"

_EXPECTED_UNCHANGED_DOMAINS = [
    "dataset",
    "tokenizer",
    "model_dimensions",
    "parameter_counts",
    "batch",
    "optimizer",
    "initialization_distributions",
    "schedules",
    "precision_environment",
    "seeds",
    "evaluation",
    "thresholds",
    "authorization_boundary",
]
_EXPECTED_PRIMITIVE_SEMANTICS: dict[str, object] = {
    "rope": {
        "pairing": "adjacent_pairs",
        "pair_indices": "(0,1),(2,3),...",
        "head_dimension_requirement": "positive_even_integer",
        "position_indexing": "zero_based",
        "inverse_frequency_formula": "inv_freq[i]=rope_base**(-2*i/head_dimension)",
        "inverse_frequency_index_range": "i_in_0_through_head_dimension_over_2_minus_1",
        "angle_formula": "theta=position*inv_freq[i]",
        "rotation_formula": [
            "rotated_even=x_even*cos(theta)-x_odd*sin(theta)",
            "rotated_odd=x_even*sin(theta)+x_odd*cos(theta)",
        ],
        "application": ["query", "key"],
        "value_rotated": False,
        "tensor_order_before_rotation": "batch_heads_sequence_head_dimension",
        "operation_order": (
            "project_then_reshape_heads_then_apply_rope_then_compute_attention_scores"
        ),
        "scaling": "none",
        "cache_trainable": False,
        "cache_persistent": False,
        "state_dict_keys": [],
    },
    "attention": {
        "score_formula": "(Q@K_transpose)/sqrt(head_dimension)",
        "causal_permission": "key_position<=query_position",
        "masked_probability_mass": 0.0,
        "softmax_accumulation_dtype_under_reduced_precision": "float32",
        "softmax_output_conversion": (
            "convert_to_declared_compute_dtype_before_value_aggregation_when_required"
        ),
    },
    "rmsnorm": {
        "formula": "x*rsqrt(mean(x^2,axis=-1,keepdims=true)+epsilon)*weight",
        "statistics_accumulation_dtype_under_reduced_precision": "float32",
        "centering": False,
        "bias": False,
        "trainable_parameters": ["weight"],
        "weight_initial_value": 1.0,
    },
    "swiglu": {
        "formula": "down_proj(silu(gate_proj(x))*up_proj(x))",
        "gate_projection_bias": False,
        "up_projection_bias": False,
        "down_projection_bias": False,
    },
}
_EXPECTED_IDENTITY_POLICY = {
    "historical_records_mutated": False,
    "resolved_profile_configurations_superseded": True,
    "profile_specification_fingerprints_superseded": True,
    "formal_experiment_definition_superseded": True,
    "project_state_superseded": True,
    "prior_d0_model_attempts_invalidated": False,
    "reason_no_attempt_invalidated": "no_D0_model_attempt_exists",
}
_EXPECTED_ACCEPTANCE_REQUIREMENTS = {
    "machine_validator": True,
    "affected_fingerprints_regenerated": True,
    "permanent_gate_extended": True,
    "exact_review_head_ci_success": True,
    "blocking_review_findings_resolved": True,
}
_ALLOWED_PREPARATION_BLOCKERS = {
    "affected_profile_fingerprints_not_yet_regenerated",
    "amendment_review_not_yet_accepted",
    "exact_head_ci_not_yet_accepted",
}


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


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _string_sequence(value: object, field: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"{field} must be an array")
    resolved = list(value)
    _require(all(isinstance(item, str) for item in resolved), f"{field} must contain strings")
    return resolved


def _validate_digest(amendment: Mapping[str, Any]) -> str:
    _require(
        amendment.get("digest_policy") == "sha256(canonical_json_without_amendment_sha256)",
        "primitive-semantics amendment digest policy changed",
    )
    declared = amendment.get("amendment_sha256")
    _require(isinstance(declared, str) and len(declared) == 64, "amendment_sha256 is invalid")
    payload = dict(amendment)
    payload.pop("amendment_sha256", None)
    actual = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    _require(declared == actual, "primitive-semantics amendment digest mismatch")
    return declared


def validate_amendment(
    amendment: Mapping[str, Any],
    project_state: Mapping[str, Any],
) -> dict[str, object]:
    """Validate the immutable proposal and its current authorization boundary."""

    _require(
        amendment.get("schema_version") == "expertforge-d0-primitive-semantics-amendment/1",
        "primitive-semantics amendment schema changed",
    )
    _require(amendment.get("status") == "proposed_not_ratified", "amendment status changed")
    _require(amendment.get("issue") == 48, "amendment issue changed")
    _require(amendment.get("parent_issue") == 41, "amendment parent issue changed")
    _require(amendment.get("ratified_issue") == 42, "ratified D0.0 issue binding changed")
    _require(
        amendment.get("base_main_commit") == "ca2601735cda2737567ace5b08e9dda7c6e119e6",
        "amendment base-main identity changed",
    )
    _require(
        amendment.get("historical_contract_path")
        == "experiments/d0/baseline-contract-v1.proposed.json",
        "historical contract path changed",
    )
    _require(
        amendment.get("ratification_record_path") == "experiments/d0/ratification-v1.json",
        "ratification-record path changed",
    )
    _require(
        amendment.get("ratification_record_sha256")
        == "d3cd5c43091d1b68bfb63ec8be896cfa663f735e74f273adbfc9765e4bab8124",
        "ratification-record identity changed",
    )
    _require(
        amendment.get("scope")
        == "output_affecting_rope_attention_rmsnorm_and_swiglu_conventions_only",
        "amendment scope changed",
    )
    _require(
        _string_sequence(amendment.get("unchanged_contract_domains"), "unchanged_contract_domains")
        == _EXPECTED_UNCHANGED_DOMAINS,
        "unchanged contract domains changed",
    )
    _require(
        dict(_mapping(amendment.get("primitive_semantics"), "primitive_semantics"))
        == _EXPECTED_PRIMITIVE_SEMANTICS,
        "primitive numerical semantics changed",
    )
    _require(
        dict(_mapping(amendment.get("identity_policy"), "identity_policy"))
        == _EXPECTED_IDENTITY_POLICY,
        "amendment identity policy changed",
    )
    _require(
        dict(_mapping(amendment.get("acceptance_requirements"), "acceptance_requirements"))
        == _EXPECTED_ACCEPTANCE_REQUIREMENTS,
        "amendment acceptance requirements changed",
    )
    blockers = _string_sequence(amendment.get("ratification_blockers"), "ratification_blockers")
    _require(len(blockers) == len(set(blockers)), "amendment blockers must be unique")
    _require(set(blockers) <= _ALLOWED_PREPARATION_BLOCKERS, "unknown amendment blocker")

    _require(project_state.get("d0_0_state") == "ratified", "D0.0 is not ratified")
    _require(project_state.get("d0_1_authorized") is True, "D0.1 authorization changed")
    _require(project_state.get("d0_2_authorized") is True, "D0.2 authorization changed")
    _require(project_state.get("d0_3_authorized") is False, "D0.3 must remain blocked")
    _require(
        project_state.get("material_execution_authorized") is False,
        "material execution must remain blocked",
    )
    _require(
        project_state.get("qualification_attempt_authorized") is False,
        "qualification attempts must remain blocked",
    )
    _require(
        project_state.get("canonical_attempt_authorized") is False,
        "canonical attempts must remain blocked",
    )

    amendment_sha256 = _validate_digest(amendment)
    return {
        "status": "valid_primitive_semantics_amendment_proposal",
        "amendment_sha256": amendment_sha256,
        "preparation_blockers": blockers,
        "preparation_blocker_count": len(blockers),
        "historical_records_mutated": False,
        "prior_d0_model_attempts_invalidated": False,
        "d0_3_authorized": False,
        "material_execution_authorized": False,
    }


def validate_all() -> dict[str, object]:
    """Load and validate the committed proposal and current project state."""

    return validate_amendment(
        load_json_object(AMENDMENT_PATH),
        load_json_object(PROJECT_STATE_PATH),
    )


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Validate the D0 primitive-semantics amendment")


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
        sys.stderr.write(f"D0 PRIMITIVE-SEMANTICS AMENDMENT INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
