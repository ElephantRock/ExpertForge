from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts.validate_d0_generation_prompts import (
    PROMPT_MANIFEST_PATH,
    load_prompt_manifest,
    scan_document_for_prompt_contamination,
    validate_generation_prompts,
)
from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    ContractValidationError,
    load_json_object,
)

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict[str, Any]:
    return copy.deepcopy(dict(load_json_object(ROOT / CONTRACT_PATH)))


def _manifest() -> tuple[dict[str, Any], bytes]:
    manifest, raw = load_prompt_manifest(PROMPT_MANIFEST_PATH)
    return copy.deepcopy(dict(manifest)), raw


def _scan(manifest: dict[str, Any], text: str) -> list[dict[str, object]]:
    return scan_document_for_prompt_contamination(
        manifest,
        document_id="doc-fixture",
        source_file_path="fixture.parquet",
        physical_row_index=7,
        text=text,
    )


def test_generation_prompt_contract_validates() -> None:
    manifest, raw = _manifest()
    report = validate_generation_prompts(_contract(), manifest, raw)

    assert report == {
        "status": "valid_generation_prompt_contract",
        "prompt_set_id": "d0-generation-prompts-v1",
        "prompt_count": 8,
        "prompt_payload_sha256": (
            "c52ef9f4420ff8160fd5f212370f46cefde71033ac38431bdad2878640070e51"
        ),
        "manifest_sha256": ("851934e4a49210b515967fad51308038dac1e7bc449774ca603f6bd9f47ccf3e"),
        "contamination_checks": [
            "exact_normalized_prompt_substring",
            "exact_normalized_probe_substring",
            "any_exact_contiguous_64_codepoint_prompt_window",
        ],
        "maximum_allowed_hits": 0,
        "actual_corpus_scan_completed": False,
        "actual_corpus_scan_stage": "D0.1_preflight_before_packing_or_training",
    }


def test_clean_document_has_no_hits() -> None:
    manifest, _ = _manifest()
    assert _scan(manifest, "A short unrelated document about rainfall measurements.") == []


def test_exact_prompt_match_has_highest_precedence() -> None:
    manifest, _ = _manifest()
    prompt = manifest["prompts"][0]["text"]
    hits = _scan(manifest, f"prefix {prompt} suffix")

    assert len(hits) == 1
    assert hits[0]["prompt_id"] == "d0-gen-01"
    assert hits[0]["check"] == "exact_normalized_prompt_substring"


def test_probe_only_match_is_detected() -> None:
    manifest, _ = _manifest()
    probe = manifest["prompts"][3]["contamination_probe_text"]
    hits = _scan(manifest, f"prefix {probe} suffix")

    assert len(hits) == 1
    assert hits[0]["prompt_id"] == "d0-gen-04"
    assert hits[0]["check"] == "exact_normalized_probe_substring"


def test_partial_64_codepoint_window_is_detected() -> None:
    manifest, _ = _manifest()
    prompt = manifest["prompts"][1]["text"]
    window = prompt[:64]
    hits = _scan(manifest, f"prefix {window} suffix")

    assert len(hits) == 1
    assert hits[0]["prompt_id"] == "d0-gen-02"
    assert hits[0]["check"] == "any_exact_contiguous_64_codepoint_prompt_window"


def test_crlf_document_normalizes_before_matching() -> None:
    manifest, _ = _manifest()
    prompt = manifest["prompts"][5]["text"].replace("\n", "\r\n")
    hits = _scan(manifest, prompt)

    assert len(hits) == 1
    assert hits[0]["prompt_id"] == "d0-gen-06"
    assert hits[0]["check"] == "exact_normalized_prompt_substring"


def test_nul_document_is_rejected() -> None:
    manifest, _ = _manifest()
    with pytest.raises(ContractValidationError, match="forbidden NUL"):
        _scan(manifest, "clean\x00not-clean")


def test_boolean_row_index_is_rejected() -> None:
    manifest, _ = _manifest()
    with pytest.raises(ContractValidationError, match="exact integer"):
        scan_document_for_prompt_contamination(
            manifest,
            document_id="doc-fixture",
            source_file_path="fixture.parquet",
            physical_row_index=True,
            text="clean",
        )


def test_contract_manifest_digest_drift_is_rejected() -> None:
    manifest, raw = _manifest()
    contract = _contract()
    contract["evaluation"]["generation_prompt_manifest_sha256"] = "0" * 64

    with pytest.raises(ContractValidationError, match="manifest SHA-256 mismatch"):
        validate_generation_prompts(contract, manifest, raw)


def test_decoding_drift_is_rejected() -> None:
    manifest, raw = _manifest()
    manifest["decoding"]["generation_temperature"] = 0.7

    with pytest.raises(ContractValidationError, match="decoding protocol changed"):
        validate_generation_prompts(_contract(), manifest, raw)


def test_prompt_text_hash_drift_is_rejected() -> None:
    manifest, raw = _manifest()
    manifest["prompts"][0]["text_sha256"] = "0" * 64

    with pytest.raises(ContractValidationError, match="text SHA-256 mismatch"):
        validate_generation_prompts(_contract(), manifest, raw)


def test_short_probe_is_rejected() -> None:
    manifest, raw = _manifest()
    prompt = manifest["prompts"][0]
    prompt["contamination_probe_text"] = prompt["text"][:32]
    prompt["contamination_probe_sha256"] = hashlib.sha256(
        prompt["contamination_probe_text"].encode("utf-8")
    ).hexdigest()

    with pytest.raises(ContractValidationError, match="shorter than 64"):
        validate_generation_prompts(_contract(), manifest, raw)


def test_duplicate_prompt_text_is_rejected() -> None:
    manifest, raw = _manifest()
    first = manifest["prompts"][0]
    second = manifest["prompts"][1]
    second["text"] = first["text"]
    second["contamination_probe_text"] = first["contamination_probe_text"]
    second["text_sha256"] = first["text_sha256"]
    second["contamination_probe_sha256"] = first["contamination_probe_sha256"]

    with pytest.raises(ContractValidationError, match="duplicates prompt text"):
        validate_generation_prompts(_contract(), manifest, raw)


def test_canonical_payload_digest_drift_is_rejected() -> None:
    manifest, raw = _manifest()
    manifest["prompt_payload_sha256"] = "0" * 64
    contract = _contract()
    contract["evaluation"]["generation_prompt_payload_sha256"] = "0" * 64

    with pytest.raises(ContractValidationError, match="canonical payload SHA-256 mismatch"):
        validate_generation_prompts(contract, manifest, raw)
