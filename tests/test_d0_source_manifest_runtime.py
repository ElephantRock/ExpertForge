from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from expertforge.d0.errors import SourceManifestError
from expertforge.d0.source_manifest import (
    canonical_json_bytes,
    load_source_manifest,
    parse_source_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/manifests/d0-fineweb-edu-sample-10bt-source-v1.json"
TOKENIZER_PATH = ROOT / "tokenizers/manifests/d0-gpt-neox-v1.json"


def _load_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _refresh_digest(manifest: dict[str, object]) -> None:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def test_dataset_manifest_runtime_identity() -> None:
    manifest = load_source_manifest(
        DATASET_PATH,
        expected_manifest_sha256="85fab524b13c49da78264eb03124ce113395657d6403627e53008aade7eea7d7",
    )
    assert manifest.kind == "dataset"
    assert manifest.repository == "HuggingFaceFW/fineweb-edu"
    assert manifest.revision == "84e8104e779e409e2267ac60609138e3dda2cbd2"
    assert len(manifest.files) == 14
    assert manifest.total_size_bytes == 28_518_193_415


def test_tokenizer_manifest_runtime_identity() -> None:
    manifest = load_source_manifest(
        TOKENIZER_PATH,
        expected_manifest_sha256="eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c",
    )
    assert manifest.kind == "tokenizer"
    assert manifest.repository == "EleutherAI/gpt-neox-20b"
    assert len(manifest.files) == 5
    assert manifest.total_size_bytes == 3_647_931


def test_manifest_path_traversal_is_rejected() -> None:
    manifest = copy.deepcopy(_load_mapping(TOKENIZER_PATH))
    files = manifest["files"]
    assert isinstance(files, list)
    first = files[0]
    assert isinstance(first, dict)
    first["path"] = "../merges.txt"
    _refresh_digest(manifest)
    with pytest.raises(SourceManifestError, match="normalized relative POSIX path"):
        parse_source_manifest(manifest)


def test_manifest_inventory_reordering_is_rejected() -> None:
    manifest = copy.deepcopy(_load_mapping(TOKENIZER_PATH))
    files = manifest["files"]
    assert isinstance(files, list)
    files[0], files[1] = files[1], files[0]
    _refresh_digest(manifest)
    with pytest.raises(SourceManifestError, match="lexicographically sorted"):
        parse_source_manifest(manifest)


def test_manifest_contract_binding_is_rejected_on_mismatch() -> None:
    manifest = _load_mapping(TOKENIZER_PATH)
    with pytest.raises(SourceManifestError, match="contract binding"):
        parse_source_manifest(manifest, expected_manifest_sha256="0" * 64)
