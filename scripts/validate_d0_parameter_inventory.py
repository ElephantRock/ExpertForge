"""Validate the D0 declarative tensor inventory against an independent closed-form count."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from scripts.validate_d0_contract import validate_contract
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
)

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "experiments/d0/parameter-inventory-v1.json"

Profile = Literal["qualification", "canonical"]
_PROFILES: tuple[Profile, ...] = ("qualification", "canonical")

_EXPECTED_COUNTING_SEMANTICS = {
    "counted_quantity": "trainable_parameter_elements",
    "tensor_instance_definition": ("distinct_parameter_tensors_after_layer_multiplicity_expansion"),
    "shape_rule": "product_of_resolved_positive_integer_dimensions",
    "multiplicity_rule": "positive_integer_or_profile_dimension_symbol",
    "aliases_counted_once": True,
    "buffers_are_parameters": False,
}

_EXPECTED_TERMS: tuple[tuple[str, str, tuple[str, ...], int | str], ...] = (
    (
        "token_embedding.weight",
        "shared_token_embedding_and_output_projection",
        ("V", "d"),
        1,
    ),
    (
        "blocks.attention_norm.weight",
        "pre_attention_rmsnorm_scale",
        ("d",),
        "L",
    ),
    (
        "blocks.attention.q_proj.weight",
        "query_projection",
        ("d", "d"),
        "L",
    ),
    (
        "blocks.attention.k_proj.weight",
        "key_projection",
        ("d", "d"),
        "L",
    ),
    (
        "blocks.attention.v_proj.weight",
        "value_projection",
        ("d", "d"),
        "L",
    ),
    (
        "blocks.attention.out_proj.weight",
        "attention_output_projection",
        ("d", "d"),
        "L",
    ),
    (
        "blocks.ffn_norm.weight",
        "pre_feed_forward_rmsnorm_scale",
        ("d",),
        "L",
    ),
    (
        "blocks.ffn.gate_proj.weight",
        "swiglu_gate_projection",
        ("f", "d"),
        "L",
    ),
    (
        "blocks.ffn.up_proj.weight",
        "swiglu_up_projection",
        ("f", "d"),
        "L",
    ),
    (
        "blocks.ffn.down_proj.weight",
        "swiglu_down_projection",
        ("d", "f"),
        "L",
    ),
    (
        "final_norm.weight",
        "final_rmsnorm_scale",
        ("d",),
        1,
    ),
)

_EXPECTED_ZERO_COMPONENTS = (
    "attention_projection_biases",
    "feed_forward_projection_biases",
    "normalization_biases",
    "rope_trainable_parameters",
    "dropout_trainable_parameters",
    "separate_output_head_parameters",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _require_sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise ContractValidationError(f"{field} must be an array")
    return value


def _require_exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ContractValidationError(f"{field} must be an exact integer")
    if value < minimum:
        raise ContractValidationError(f"{field} must be >= {minimum}")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    field: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    _require(not missing, f"{field} missing keys: {missing}")
    _require(not unexpected, f"{field} has unexpected keys: {unexpected}")


def _resolve_factor(
    value: object,
    dimensions: Mapping[str, int],
    field: str,
) -> int:
    if type(value) is int:
        return _require_exact_int(value, field, minimum=1)
    if isinstance(value, str):
        _require(value in dimensions, f"{field} references unknown dimension {value!r}")
        return _require_exact_int(
            dimensions[value],
            f"{field} resolved dimension {value!r}",
            minimum=1,
        )
    raise ContractValidationError(f"{field} must be an integer or dimension symbol")


def count_parameter_term(
    term: Mapping[str, Any],
    dimensions: Mapping[str, int],
) -> tuple[int, int]:
    """Return ``(parameter_elements, tensor_instances)`` for one inventory term."""

    shape = _require_sequence(term.get("shape"), "term.shape")
    _require(bool(shape), "term.shape must not be empty")
    multiplicity = _resolve_factor(term.get("multiplicity"), dimensions, "term.multiplicity")

    elements_per_tensor = 1
    for index, factor in enumerate(shape):
        elements_per_tensor *= _resolve_factor(
            factor,
            dimensions,
            f"term.shape[{index}]",
        )
    return elements_per_tensor * multiplicity, multiplicity


def _validate_header(
    inventory: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        inventory,
        {
            "schema_version",
            "status",
            "issue",
            "parent_issue",
            "contract_path",
            "counting_semantics",
            "parameter_terms",
            "aliases",
            "zero_parameter_components",
            "profiles",
        },
        "inventory",
    )
    _require(
        inventory["schema_version"] == "expertforge-d0-parameter-inventory/1",
        "unexpected parameter-inventory schema version",
    )
    _require(inventory["status"] == "proposed_not_ratified", "unexpected inventory status")
    _require(inventory["issue"] == contract["issue"] == 42, "inventory issue binding changed")
    _require(
        inventory["parent_issue"] == contract["parent_issue"] == 41,
        "inventory parent-issue binding changed",
    )
    _require(
        inventory["contract_path"] == CONTRACT_PATH.as_posix(),
        "inventory contract path changed",
    )

    semantics = _require_mapping(inventory["counting_semantics"], "counting_semantics")
    _require(
        dict(semantics) == _EXPECTED_COUNTING_SEMANTICS,
        "parameter-counting semantics changed",
    )


def _validate_terms(inventory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_terms = _require_sequence(inventory["parameter_terms"], "parameter_terms")
    _require(
        len(raw_terms) == len(_EXPECTED_TERMS),
        "parameter inventory must enumerate eleven tensor families",
    )

    terms: list[Mapping[str, Any]] = []
    for index, (raw_term, expected) in enumerate(zip(raw_terms, _EXPECTED_TERMS, strict=True)):
        term = _require_mapping(raw_term, f"parameter_terms[{index}]")
        _require_exact_keys(
            term,
            {"name", "role", "parameter_kind", "trainable", "shape", "multiplicity"},
            f"parameter_terms[{index}]",
        )
        name, role, shape, multiplicity = expected
        _require(term["name"] == name, f"parameter term {index} name changed")
        _require(term["role"] == role, f"{name} role changed")
        _require(term["parameter_kind"] == "weight", f"{name} parameter kind changed")
        _require(term["trainable"] is True, f"{name} must remain trainable")
        _require(
            tuple(_require_sequence(term["shape"], f"{name}.shape")) == shape,
            f"{name} shape changed",
        )
        _require(term["multiplicity"] == multiplicity, f"{name} multiplicity changed")
        terms.append(term)
    return terms


def _validate_aliases(inventory: Mapping[str, Any]) -> None:
    aliases = _require_sequence(inventory["aliases"], "aliases")
    _require(len(aliases) == 1, "exactly one tied output-head alias is required")
    alias = _require_mapping(aliases[0], "aliases[0]")
    _require_exact_keys(
        alias,
        {"name", "alias_of", "storage", "additional_parameter_elements"},
        "aliases[0]",
    )
    _require(alias["name"] == "output_head.weight", "output-head alias name changed")
    _require(
        alias["alias_of"] == "token_embedding.weight",
        "output head must alias token_embedding.weight",
    )
    _require(alias["storage"] == "shared", "output-head alias storage changed")
    _require(
        _require_exact_int(
            alias["additional_parameter_elements"],
            "aliases[0].additional_parameter_elements",
        )
        == 0,
        "tied output head must add zero parameter elements",
    )


def _validate_zero_components(inventory: Mapping[str, Any]) -> None:
    raw_components = _require_sequence(
        inventory["zero_parameter_components"],
        "zero_parameter_components",
    )
    _require(
        len(raw_components) == len(_EXPECTED_ZERO_COMPONENTS),
        "zero-parameter component inventory changed",
    )
    for index, (raw_component, expected_name) in enumerate(
        zip(raw_components, _EXPECTED_ZERO_COMPONENTS, strict=True)
    ):
        component = _require_mapping(
            raw_component,
            f"zero_parameter_components[{index}]",
        )
        _require_exact_keys(
            component,
            {"name", "expected_parameter_elements", "reason"},
            f"zero_parameter_components[{index}]",
        )
        _require(
            component["name"] == expected_name,
            f"zero-parameter component {index} name changed",
        )
        _require(
            _require_exact_int(
                component["expected_parameter_elements"],
                f"{expected_name}.expected_parameter_elements",
            )
            == 0,
            f"{expected_name} must remain zero",
        )
        reason = component["reason"]
        _require(
            isinstance(reason, str) and bool(reason.strip()), f"{expected_name} reason is blank"
        )


def _profile_dimensions(
    profile: Profile,
    profile_inventory: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, int]:
    model = _require_mapping(contract["models"][profile], f"models.{profile}")
    tokenizer = _require_mapping(contract["tokenizer"], "tokenizer")
    expected = {
        "V": _require_exact_int(
            tokenizer["vocabulary_size"],
            "tokenizer.vocabulary_size",
            minimum=1,
        ),
        "L": _require_exact_int(model["layers"], f"models.{profile}.layers", minimum=1),
        "d": _require_exact_int(
            model["model_width"],
            f"models.{profile}.model_width",
            minimum=1,
        ),
        "f": _require_exact_int(
            model["swiglu_intermediate_width"],
            f"models.{profile}.swiglu_intermediate_width",
            minimum=1,
        ),
    }
    dimensions = _require_mapping(
        profile_inventory["dimensions"],
        f"profiles.{profile}.dimensions",
    )
    _require_exact_keys(dimensions, set(expected), f"profiles.{profile}.dimensions")
    for name, expected_value in expected.items():
        actual = _require_exact_int(
            dimensions[name],
            f"profiles.{profile}.dimensions.{name}",
            minimum=1,
        )
        _require(actual == expected_value, f"{profile} dimension {name} disagrees with contract")
    return expected


def _validate_profile(
    profile: Profile,
    profile_inventory: Mapping[str, Any],
    terms: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    formula_report: Mapping[str, Any],
) -> dict[str, int]:
    _require_exact_keys(
        profile_inventory,
        {
            "model_id",
            "dimensions",
            "expected_term_elements",
            "expected_parameter_tensor_instances",
            "expected_parameter_families",
            "expected_trainable_parameters",
            "expected_non_trainable_parameters",
        },
        f"profiles.{profile}",
    )
    contract_model = _require_mapping(contract["models"][profile], f"models.{profile}")
    _require(
        profile_inventory["model_id"] == contract_model["model_id"],
        f"{profile} model identity changed",
    )
    dimensions = _profile_dimensions(profile, profile_inventory, contract)

    expected_terms = _require_sequence(
        profile_inventory["expected_term_elements"],
        f"profiles.{profile}.expected_term_elements",
    )
    _require(
        len(expected_terms) == len(terms),
        f"{profile} expected term subtotal inventory changed",
    )

    inventory_total = 0
    tensor_instances = 0
    for index, (term, raw_expected) in enumerate(zip(terms, expected_terms, strict=True)):
        name = str(term["name"])
        expected = _require_mapping(
            raw_expected,
            f"profiles.{profile}.expected_term_elements[{index}]",
        )
        _require_exact_keys(
            expected,
            {"name", "parameter_elements"},
            f"profiles.{profile}.expected_term_elements[{index}]",
        )
        _require(expected["name"] == name, f"{profile} expected term {index} name changed")
        calculated, instances = count_parameter_term(term, dimensions)
        declared_subtotal = _require_exact_int(
            expected["parameter_elements"],
            f"profiles.{profile}.{name}.parameter_elements",
            minimum=1,
        )
        _require(
            calculated == declared_subtotal,
            f"{profile} {name} subtotal mismatch",
        )
        inventory_total += calculated
        tensor_instances += instances

    declared_instances = _require_exact_int(
        profile_inventory["expected_parameter_tensor_instances"],
        f"profiles.{profile}.expected_parameter_tensor_instances",
        minimum=1,
    )
    _require(
        tensor_instances == declared_instances,
        f"{profile} parameter tensor-instance count mismatch",
    )
    _require(
        _require_exact_int(
            profile_inventory["expected_parameter_families"],
            f"profiles.{profile}.expected_parameter_families",
            minimum=1,
        )
        == len(terms),
        f"{profile} parameter-family count mismatch",
    )
    declared_total = _require_exact_int(
        profile_inventory["expected_trainable_parameters"],
        f"profiles.{profile}.expected_trainable_parameters",
        minimum=1,
    )
    _require(inventory_total == declared_total, f"{profile} inventory total mismatch")
    _require(
        declared_total == contract_model["trainable_parameters"],
        f"{profile} inventory total disagrees with contract",
    )
    _require(
        _require_exact_int(
            profile_inventory["expected_non_trainable_parameters"],
            f"profiles.{profile}.expected_non_trainable_parameters",
        )
        == contract_model["non_trainable_parameters"]
        == 0,
        f"{profile} non-trainable parameter total changed",
    )

    formula_models = _require_mapping(formula_report["models"], "formula_report.models")
    formula_profile = _require_mapping(
        formula_models[profile],
        f"formula_report.models.{profile}",
    )
    closed_form_total = _require_exact_int(
        formula_profile["derived_parameters"],
        f"formula_report.models.{profile}.derived_parameters",
        minimum=1,
    )
    _require(
        inventory_total == closed_form_total,
        f"{profile} tensor inventory and closed-form accounting disagree",
    )

    return {
        "parameter_families": len(terms),
        "parameter_tensor_instances": tensor_instances,
        "inventory_total": inventory_total,
        "closed_form_total": closed_form_total,
        "contract_total": _require_exact_int(
            contract_model["trainable_parameters"],
            f"models.{profile}.trainable_parameters",
            minimum=1,
        ),
    }


def validate_parameter_inventory(
    inventory: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the declarative inventory and compare two independent counters."""

    formula_report = validate_contract(contract)
    _validate_header(inventory, contract)
    terms = _validate_terms(inventory)
    _validate_aliases(inventory)
    _validate_zero_components(inventory)

    profiles = _require_mapping(inventory["profiles"], "profiles")
    _require_exact_keys(profiles, set(_PROFILES), "profiles")
    profile_reports = {
        profile: _validate_profile(
            profile,
            _require_mapping(profiles[profile], f"profiles.{profile}"),
            terms,
            contract,
            formula_report,
        )
        for profile in _PROFILES
    }

    return {
        "status": "valid_parameter_inventory",
        "schema_version": inventory["schema_version"],
        "comparison": "declarative_tensor_inventory_equals_independent_closed_form",
        "profiles": profile_reports,
    }


def validate_all() -> dict[str, Any]:
    """Validate the committed D0 parameter inventory and baseline contract."""

    return validate_parameter_inventory(
        load_json_object(INVENTORY_PATH),
        load_json_object(ROOT / CONTRACT_PATH),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate D0 tensor parameter accounting")
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--contract", type=Path, default=ROOT / CONTRACT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_parameter_inventory(
            load_json_object(args.inventory),
            load_json_object(args.contract),
        )
    except (ContractValidationError, KeyError, OSError, json.JSONDecodeError) as error:
        sys.stderr.write(f"D0 PARAMETER INVENTORY INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
