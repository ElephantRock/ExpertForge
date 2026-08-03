"""Materialize superseding D0 configs, fingerprints, and formal definition."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from expertforge.config.resolve import canonical_bytes, resolve_config
from expertforge.identity.fingerprint import ImmutableInput, specification_fingerprint

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_SHA256 = "5bc196942697833c4dfe9f227b16e2b69c6cafd07b05396a70650ecb34cebab0"
DATASET_SHA256 = "85fab524b13c49da78264eb03124ce113395657d6403627e53008aade7eea7d7"
TOKENIZER_SHA256 = "eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c"

_PROFILE_PATHS = {
    "qualification": (
        ROOT / "configs/d0/qualification.yaml",
        ROOT / "configs/d0/qualification-primitive-v2.yaml",
        ROOT / "experiments/d0/qualification-fingerprint-v2.json",
    ),
    "canonical": (
        ROOT / "configs/d0/canonical.yaml",
        ROOT / "configs/d0/canonical-primitive-v2.yaml",
        ROOT / "experiments/d0/canonical-fingerprint-v2.json",
    ),
}
_FORMAL_V1 = ROOT / "experiments/d0/formal-experiment-definition-v1.json"
_FORMAL_V2 = ROOT / "experiments/d0/formal-experiment-definition-v2.json"

_SEMANTICS_YAML = """
d0_primitive_semantics:
  amendment_path: experiments/d0/primitive-semantics-amendment-v1.proposed.json
  amendment_sha256: 5bc196942697833c4dfe9f227b16e2b69c6cafd07b05396a70650ecb34cebab0
  rope:
    pairing: adjacent_pairs
    pair_indices: "(0,1),(2,3),..."
    head_dimension_requirement: positive_even_integer
    position_indexing: zero_based
    inverse_frequency_formula: "inv_freq[i]=rope_base**(-2*i/head_dimension)"
    inverse_frequency_index_range: i_in_0_through_head_dimension_over_2_minus_1
    angle_formula: "theta=position*inv_freq[i]"
    rotation_formula:
    - "rotated_even=x_even*cos(theta)-x_odd*sin(theta)"
    - "rotated_odd=x_even*sin(theta)+x_odd*cos(theta)"
    application:
    - query
    - key
    value_rotated: false
    tensor_order_before_rotation: batch_heads_sequence_head_dimension
    operation_order: project_then_reshape_heads_then_apply_rope_then_compute_attention_scores
    scaling: none
    cache_trainable: false
    cache_persistent: false
    state_dict_keys: []
  attention:
    score_formula: "(Q@K_transpose)/sqrt(head_dimension)"
    causal_permission: "key_position<=query_position"
    masked_probability_mass: 0.0
    softmax_accumulation_dtype_under_reduced_precision: float32
    softmax_output_conversion: convert_to_declared_compute_dtype_before_value_aggregation_when_required
  rmsnorm:
    formula: "x*rsqrt(mean(x^2,axis=-1,keepdims=true)+epsilon)*weight"
    statistics_accumulation_dtype_under_reduced_precision: float32
    centering: false
    bias: false
    trainable_parameters:
    - weight
    weight_initial_value: 1.0
  swiglu:
    formula: "down_proj(silu(gate_proj(x))*up_proj(x))"
    gate_projection_bias: false
    up_projection_bias: false
    down_projection_bias: false
""".lstrip()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _config_bytes(source: Path) -> bytes:
    original = source.read_bytes()
    if not original.endswith(b"\n"):
        original += b"\n"
    return original + _SEMANTICS_YAML.encode("utf-8")


def _fingerprint_bytes(config_payload: bytes) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory(prefix="expertforge-d0-amendment-") as directory:
        config_path = Path(directory) / "config.yaml"
        config_path.write_bytes(config_payload)
        envelope = resolve_config(config_path)
        fingerprint = specification_fingerprint(
            canonical_bytes(envelope),
            (
                ImmutableInput(
                    name="dataset.manifest",
                    algorithm="sha256",
                    digest=DATASET_SHA256,
                ),
                ImmutableInput(
                    name="primitive.semantics.amendment",
                    algorithm="sha256",
                    digest=AMENDMENT_SHA256,
                ),
                ImmutableInput(
                    name="tokenizer.manifest",
                    algorithm="sha256",
                    digest=TOKENIZER_SHA256,
                ),
            ),
        )
    value = fingerprint.model_dump(mode="json", by_alias=True)
    return _json_bytes(value), fingerprint.digest_str


def _formal_definition_bytes(fingerprints: dict[str, str]) -> bytes:
    value = json.loads(_FORMAL_V1.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("historical formal experiment definition must be an object")
    value["schema_version"] = "expertforge-d0-formal-experiment-definition/2"
    value["issue"] = 48
    value["pull_request"] = 50
    value["ratified_issue"] = 42
    value["supersedes"] = "experiments/d0/formal-experiment-definition-v1.json"
    value["primitive_semantics_amendment_path"] = (
        "experiments/d0/primitive-semantics-amendment-v1.proposed.json"
    )
    value["primitive_semantics_amendment_sha256"] = AMENDMENT_SHA256
    value["prior_d0_model_attempts_invalidated"] = False
    value["reason_no_attempt_invalidated"] = "no_D0_model_attempt_exists"
    profiles = value.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("historical formal definition profiles must be an object")
    for profile, fingerprint in fingerprints.items():
        entry = profiles.get(profile)
        if not isinstance(entry, dict):
            raise ValueError(f"missing formal profile {profile}")
        entry["config_path"] = f"configs/d0/{profile}-primitive-v2.yaml"
        entry["fingerprint_path"] = f"experiments/d0/{profile}-fingerprint-v2.json"
        entry["specification_fingerprint"] = fingerprint
        constraints = entry.get("fixed_constraints")
        if not isinstance(constraints, list) or not all(
            isinstance(item, str) for item in constraints
        ):
            raise ValueError(f"{profile} fixed constraints must be strings")
        updated = [
            f"configuration.specification_fingerprint={fingerprint}"
            if item.startswith("configuration.specification_fingerprint=")
            else item
            for item in constraints
        ]
        updated.extend(
            [
                f"primitive_semantics.amendment.sha256={AMENDMENT_SHA256}",
                "primitive_semantics.rope.pairing=adjacent_pairs",
                "primitive_semantics.rope.position_indexing=zero_based",
                "primitive_semantics.attention.causal_permission=key_position<=query_position",
                "primitive_semantics.attention.softmax_accumulation_dtype=float32",
                "primitive_semantics.rmsnorm.statistics_accumulation_dtype=float32",
            ]
        )
        entry["fixed_constraints"] = sorted(updated)
    return _json_bytes(value)


def _expected_outputs() -> dict[Path, bytes]:
    outputs: dict[Path, bytes] = {}
    fingerprints: dict[str, str] = {}
    for profile, (source, amended, fingerprint_path) in _PROFILE_PATHS.items():
        config_payload = _config_bytes(source)
        outputs[amended] = config_payload
        fingerprint_bytes, fingerprint = _fingerprint_bytes(config_payload)
        outputs[fingerprint_path] = fingerprint_bytes
        fingerprints[profile] = fingerprint
    outputs[_FORMAL_V2] = _formal_definition_bytes(fingerprints)
    return outputs


def materialize(*, check: bool) -> None:
    outputs = _expected_outputs()
    mismatches: list[str] = []
    for path, expected in outputs.items():
        if check:
            actual = path.read_bytes() if path.exists() else None
            if actual != expected:
                mismatches.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)
    if mismatches:
        raise ValueError(f"amendment materialization is stale: {mismatches}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        materialize(check=args.check)
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"D0 AMENDMENT MATERIALIZATION FAILED: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
