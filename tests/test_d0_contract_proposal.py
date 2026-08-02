from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.validate_d0_contract import (
    CONTRACT_PATH,
    ContractValidationError,
    load_contract,
    parameter_count,
    swiglu_width,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict[str, object]:
    return dict(load_contract(ROOT / CONTRACT_PATH))


def test_parameter_accounting_known_answers() -> None:
    assert swiglu_width(256) == 768
    assert swiglu_width(576) == 1536
    assert (
        parameter_count(
            vocabulary_size=50_257,
            layers=8,
            width=256,
            ffn_width=768,
        )
        == 19_685_888
    )
    assert (
        parameter_count(
            vocabulary_size=50_257,
            layers=12,
            width=576,
            ffn_width=1_536,
        )
        == 76_738_176
    )


def test_proposed_contract_validates() -> None:
    report = validate_contract(load_contract(ROOT / CONTRACT_PATH))

    assert report["status"] == "valid_proposal"
    assert report["target_tokens_per_update"] == 65_536
    assert report["models"]["qualification"]["derived_parameters"] == 19_685_888
    assert report["models"]["canonical"]["derived_parameters"] == 76_738_176
    assert report["schedules"]["qualification"]["derived_updates"] == 4_000
    assert report["schedules"]["canonical"]["derived_updates"] == 32_000
    assert report["source_manifests"]["tokenizer"]["file_count"] == 5
    assert report["source_manifests"]["dataset"]["file_count"] == 14
    assert report["ratification_blocker_count"] == 7


def test_parameter_drift_is_rejected() -> None:
    contract = copy.deepcopy(_contract())
    models = contract["models"]
    assert isinstance(models, dict)
    qualification = models["qualification"]
    assert isinstance(qualification, dict)
    qualification["trainable_parameters"] = 19_685_889

    with pytest.raises(ContractValidationError, match="qualification parameter count mismatch"):
        validate_contract(contract)


def test_batch_drift_is_rejected() -> None:
    contract = copy.deepcopy(_contract())
    batch = contract["batch"]
    assert isinstance(batch, dict)
    batch["target_tokens_per_update"] = 65_535

    with pytest.raises(ContractValidationError, match="global target-token batch mismatch"):
        validate_contract(contract)


def test_approximate_placeholder_is_rejected() -> None:
    contract = copy.deepcopy(_contract())
    contract["status"] = "approximately ready"

    with pytest.raises(ContractValidationError):
        validate_contract(contract)
