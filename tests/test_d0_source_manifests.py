from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.validate_d0_source_manifests import (
    CONTRACT_PATH,
    DATASET_MANIFEST_PATH,
    TOKENIZER_MANIFEST_PATH,
    ContractValidationError,
    canonical_manifest_digest,
    load_json_object,
    validate_source_manifests,
)

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict[str, object]:
    return dict(load_json_object(ROOT / CONTRACT_PATH))


def _tokenizer_manifest() -> dict[str, object]:
    return dict(load_json_object(ROOT / TOKENIZER_MANIFEST_PATH))


def _dataset_manifest() -> dict[str, object]:
    return dict(load_json_object(ROOT / DATASET_MANIFEST_PATH))


def _refresh_digest(manifest: dict[str, object]) -> None:
    manifest["manifest_sha256"] = canonical_manifest_digest(manifest)


def test_source_manifests_validate() -> None:
    report = validate_source_manifests(
        _contract(),
        _tokenizer_manifest(),
        _dataset_manifest(),
    )

    assert report["tokenizer"]["file_count"] == 5
    assert report["tokenizer"]["total_size_bytes"] == 3_647_931
    assert report["dataset"]["file_count"] == 14
    assert report["dataset"]["total_size_bytes"] == 28_518_193_415


def test_tokenizer_size_drift_is_rejected() -> None:
    tokenizer_manifest = copy.deepcopy(_tokenizer_manifest())
    files = tokenizer_manifest["files"]
    assert isinstance(files, list)
    first = files[0]
    assert isinstance(first, dict)
    first["size_bytes"] = 456_584

    with pytest.raises(ContractValidationError, match="total_size_bytes mismatch"):
        validate_source_manifests(_contract(), tokenizer_manifest, _dataset_manifest())


def test_duplicate_dataset_path_is_rejected() -> None:
    dataset_manifest = copy.deepcopy(_dataset_manifest())
    files = dataset_manifest["files"]
    assert isinstance(files, list)
    first = files[0]
    second = files[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    second["path"] = first["path"]
    _refresh_digest(dataset_manifest)

    with pytest.raises(ContractValidationError, match="file paths must be unique"):
        validate_source_manifests(_contract(), _tokenizer_manifest(), dataset_manifest)


def test_dataset_manifest_binding_drift_is_rejected() -> None:
    contract = copy.deepcopy(_contract())
    dataset = contract["dataset"]
    assert isinstance(dataset, dict)
    dataset["source_manifest_sha256"] = "0" * 64

    with pytest.raises(ContractValidationError, match="dataset manifest binding mismatch"):
        validate_source_manifests(contract, _tokenizer_manifest(), _dataset_manifest())


def test_manifest_canonical_digest_rejects_semantic_edit() -> None:
    tokenizer_manifest = copy.deepcopy(_tokenizer_manifest())
    tokenizer = tokenizer_manifest["tokenizer"]
    assert isinstance(tokenizer, dict)
    tokenizer["normalizer"] = "nfc"

    with pytest.raises(ContractValidationError, match="canonical digest mismatch"):
        validate_source_manifests(_contract(), tokenizer_manifest, _dataset_manifest())


def test_dataset_revision_before_sample_upload_is_rejected() -> None:
    contract = copy.deepcopy(_contract())
    dataset = contract["dataset"]
    assert isinstance(dataset, dict)
    dataset["revision"] = "21974026070c0d94eb843d8eba56d02550f4c0b5"

    with pytest.raises(ContractValidationError, match="sample upload commit"):
        validate_source_manifests(contract, _tokenizer_manifest(), _dataset_manifest())
