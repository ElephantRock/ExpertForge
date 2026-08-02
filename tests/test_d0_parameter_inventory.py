from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

import scripts.validate_d0_parameter_inventory as inventory_module
from scripts.validate_d0_parameter_inventory import (
    INVENTORY_PATH,
    count_parameter_term,
    validate_all,
    validate_parameter_inventory,
)
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
)

ROOT = Path(__file__).resolve().parents[1]


def _inventory() -> dict[str, Any]:
    return copy.deepcopy(dict(load_json_object(INVENTORY_PATH)))


def _contract() -> dict[str, Any]:
    return copy.deepcopy(dict(load_json_object(ROOT / CONTRACT_PATH)))


def _term(inventory: dict[str, Any], name: str) -> dict[str, Any]:
    terms = inventory["parameter_terms"]
    assert isinstance(terms, list)
    for term in terms:
        assert isinstance(term, dict)
        if term["name"] == name:
            return term
    raise AssertionError(f"missing term {name}")


def test_parameter_inventory_validates() -> None:
    report = validate_all()

    assert report["status"] == "valid_parameter_inventory"
    assert report["comparison"] == "declarative_tensor_inventory_equals_independent_closed_form"
    assert report["profiles"]["qualification"] == {
        "parameter_families": 11,
        "parameter_tensor_instances": 74,
        "inventory_total": 19_685_888,
        "closed_form_total": 19_685_888,
        "contract_total": 19_685_888,
    }
    assert report["profiles"]["canonical"] == {
        "parameter_families": 11,
        "parameter_tensor_instances": 110,
        "inventory_total": 76_738_176,
        "closed_form_total": 76_738_176,
        "contract_total": 76_738_176,
    }


def test_inventory_counter_does_not_import_closed_form_helpers() -> None:
    assert not hasattr(inventory_module, "parameter_count")
    assert not hasattr(inventory_module, "swiglu_width")


def test_generic_term_counter_known_answers() -> None:
    dimensions = {"V": 50_257, "L": 8, "d": 256, "f": 768}

    embedding_elements, embedding_instances = count_parameter_term(
        {"shape": ["V", "d"], "multiplicity": 1},
        dimensions,
    )
    q_projection_elements, q_projection_instances = count_parameter_term(
        {"shape": ["d", "d"], "multiplicity": "L"},
        dimensions,
    )
    gate_projection_elements, gate_projection_instances = count_parameter_term(
        {"shape": ["f", "d"], "multiplicity": "L"},
        dimensions,
    )

    assert (embedding_elements, embedding_instances) == (12_865_792, 1)
    assert (q_projection_elements, q_projection_instances) == (524_288, 8)
    assert (gate_projection_elements, gate_projection_instances) == (1_572_864, 8)


def test_missing_tensor_family_is_rejected() -> None:
    inventory = _inventory()
    terms = inventory["parameter_terms"]
    assert isinstance(terms, list)
    terms.pop()

    with pytest.raises(
        ContractValidationError,
        match="must enumerate eleven tensor families",
    ):
        validate_parameter_inventory(inventory, _contract())


def test_tensor_shape_drift_is_rejected() -> None:
    inventory = _inventory()
    _term(inventory, "blocks.attention.q_proj.weight")["shape"] = ["d"]

    with pytest.raises(ContractValidationError, match="q_proj.weight shape changed"):
        validate_parameter_inventory(inventory, _contract())


def test_layer_multiplicity_drift_is_rejected() -> None:
    inventory = _inventory()
    _term(inventory, "blocks.ffn.down_proj.weight")["multiplicity"] = 1

    with pytest.raises(ContractValidationError, match="down_proj.weight multiplicity changed"):
        validate_parameter_inventory(inventory, _contract())


def test_declared_term_subtotal_drift_is_rejected() -> None:
    inventory = _inventory()
    profiles = inventory["profiles"]
    assert isinstance(profiles, dict)
    qualification = profiles["qualification"]
    assert isinstance(qualification, dict)
    expected_terms = qualification["expected_term_elements"]
    assert isinstance(expected_terms, list)
    first = expected_terms[0]
    assert isinstance(first, dict)
    first["parameter_elements"] = 12_865_793

    with pytest.raises(
        ContractValidationError,
        match="qualification token_embedding.weight subtotal mismatch",
    ):
        validate_parameter_inventory(inventory, _contract())


def test_tied_output_head_double_count_is_rejected() -> None:
    inventory = _inventory()
    aliases = inventory["aliases"]
    assert isinstance(aliases, list)
    alias = aliases[0]
    assert isinstance(alias, dict)
    alias["additional_parameter_elements"] = 50_257 * 256

    with pytest.raises(
        ContractValidationError,
        match="tied output head must add zero parameter elements",
    ):
        validate_parameter_inventory(inventory, _contract())


def test_non_trainable_parameter_drift_is_rejected() -> None:
    inventory = _inventory()
    profiles = inventory["profiles"]
    assert isinstance(profiles, dict)
    canonical = profiles["canonical"]
    assert isinstance(canonical, dict)
    canonical["expected_non_trainable_parameters"] = 1

    with pytest.raises(
        ContractValidationError,
        match="canonical non-trainable parameter total changed",
    ):
        validate_parameter_inventory(inventory, _contract())


def test_profile_identity_crossover_is_rejected() -> None:
    inventory = _inventory()
    profiles = inventory["profiles"]
    assert isinstance(profiles, dict)
    qualification = profiles["qualification"]
    canonical = profiles["canonical"]
    assert isinstance(qualification, dict)
    assert isinstance(canonical, dict)
    qualification["model_id"] = canonical["model_id"]

    with pytest.raises(ContractValidationError, match="qualification model identity changed"):
        validate_parameter_inventory(inventory, _contract())


def test_unknown_dimension_symbol_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="unknown dimension"):
        count_parameter_term(
            {"shape": ["d", "unknown"], "multiplicity": 1},
            {"d": 256},
        )


def test_boolean_dimension_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="must be an exact integer"):
        count_parameter_term(
            {"shape": ["d"], "multiplicity": 1},
            {"d": True},  # type: ignore[dict-item]
        )
