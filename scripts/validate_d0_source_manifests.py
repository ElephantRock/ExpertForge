"""Validate immutable D0 tokenizer and dataset source manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = Path("experiments/d0/baseline-contract-v1.proposed.json")
TOKENIZER_MANIFEST_PATH = Path("tokenizers/manifests/d0-gpt-neox-v1.json")
DATASET_MANIFEST_PATH = Path("data/manifests/d0-fineweb-edu-sample-10bt-source-v1.json")


class ContractValidationError(ValueError):
    """D0 contract evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def load_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{path}: expected a JSON object")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    return value


def _files(manifest: Mapping[str, Any], field: str) -> list[Mapping[str, Any]]:
    value = manifest.get("files")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"{field}.files must be an array")
    return [_mapping(member, f"{field}.files[{index}]") for index, member in enumerate(value)]


def _validate_inventory(manifest: Mapping[str, Any], field: str) -> dict[str, int]:
    files = _files(manifest, field)
    paths: list[str] = []
    total = 0
    for index, entry in enumerate(files):
        path = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if not isinstance(path, str) or not path:
            raise ContractValidationError(f"{field}.files[{index}].path invalid")
        if not (
            isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        ):
            raise ContractValidationError(f"{field}.files[{index}].sha256 invalid")
        if type(size) is not int or size < 0:
            raise ContractValidationError(f"{field}.files[{index}].size_bytes invalid")
        paths.append(path)
        total += size
    _require(len(set(paths)) == len(paths), f"{field} file paths must be unique")
    _require(paths == sorted(paths), f"{field} file paths must be sorted")
    _require(manifest.get("file_count") == len(files), f"{field} file_count mismatch")
    _require(manifest.get("total_size_bytes") == total, f"{field} total_size_bytes mismatch")
    actual = canonical_manifest_digest(manifest)
    _require(
        manifest.get("manifest_sha256") == actual,
        f"{field} canonical digest mismatch: actual {actual}",
    )
    return {"file_count": len(files), "total_size_bytes": total}


def _validate_dataset_provenance(manifest: Mapping[str, Any]) -> None:
    provenance = _mapping(manifest.get("artifact_provenance"), "dataset.artifact_provenance")
    expected_keys = {
        "artifact_kind",
        "format_version",
        "source_identity",
        "parent_experiment",
        "generation_method",
        "generation_command",
        "content_hash",
        "retention_status",
        "durable_location",
        "reproduction_path",
    }
    _require(set(provenance) == expected_keys, "dataset artifact provenance keys changed")
    _require(
        provenance["artifact_kind"] == "dataset_source_identity_manifest",
        "dataset artifact kind changed",
    )
    _require(provenance["format_version"] == 1, "dataset provenance format changed")
    for field in (
        "source_identity",
        "parent_experiment",
        "generation_method",
        "generation_command",
        "durable_location",
        "reproduction_path",
    ):
        _require(
            isinstance(provenance[field], str) and bool(str(provenance[field]).strip()),
            f"dataset provenance {field} missing",
        )
    _require(
        provenance["retention_status"] == "permanent_contract_evidence",
        "dataset retention status changed",
    )
    content_hash = _mapping(provenance["content_hash"], "dataset provenance content_hash")
    _require(content_hash.get("algorithm") == "sha256", "dataset content hash algorithm changed")
    _require(
        content_hash.get("scope") == "canonical_json_of_ordered_files_array",
        "dataset content hash scope changed",
    )
    inventory_digest = hashlib.sha256(
        _canonical_bytes(list(_files(manifest, "dataset")))
    ).hexdigest()
    _require(
        content_hash.get("digest") == inventory_digest,
        "dataset provenance content hash mismatch",
    )


def validate_source_manifests(
    contract: Mapping[str, Any],
    tokenizer_manifest: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    tokenizer = _mapping(contract.get("tokenizer"), "tokenizer")
    dataset = _mapping(contract.get("dataset"), "dataset")
    token_upstream = _mapping(tokenizer_manifest.get("upstream"), "tokenizer upstream")
    data_upstream = _mapping(dataset_manifest.get("upstream"), "dataset upstream")
    _require(
        token_upstream.get("repository") == tokenizer.get("repository")
        and token_upstream.get("revision") == tokenizer.get("revision"),
        "tokenizer upstream identity mismatch",
    )
    _require(
        tokenizer_manifest.get("manifest_sha256") == tokenizer.get("source_manifest_sha256"),
        "tokenizer manifest binding mismatch",
    )
    _require(
        data_upstream.get("repository") == dataset.get("repository")
        and data_upstream.get("revision") == dataset.get("revision")
        and data_upstream.get("configuration") == dataset.get("configuration"),
        "dataset revision must be the exact sample upload commit",
    )
    _require(
        dataset_manifest.get("manifest_sha256") == dataset.get("source_manifest_sha256"),
        "dataset manifest binding mismatch",
    )
    _require(
        dataset_manifest.get("normalization") == dataset.get("normalization"),
        "dataset normalization mismatch",
    )
    _require(
        dataset_manifest.get("split_contract") == dataset.get("split"),
        "dataset split contract mismatch",
    )
    _require(
        dataset_manifest.get("ordering")
        == {
            "source_files": "path_lexicographic",
            "rows_within_file": "physical_row_index_ascending",
        }
        and dataset.get("source_order")
        == ["file_path_lexicographic", "physical_row_index_ascending"],
        "dataset ordering mismatch",
    )
    _require(
        dataset_manifest.get("deduplication")
        == {
            "duplicate_key": dataset.get("duplicate_key"),
            "keep_rule": dataset.get("duplicate_keep_rule"),
        },
        "dataset deduplication mismatch",
    )
    _validate_dataset_provenance(dataset_manifest)
    return {
        "tokenizer": _validate_inventory(tokenizer_manifest, "tokenizer"),
        "dataset": _validate_inventory(dataset_manifest, "dataset"),
    }


def main() -> int:
    report = validate_source_manifests(
        load_json_object(ROOT / CONTRACT_PATH),
        load_json_object(ROOT / TOKENIZER_MANIFEST_PATH),
        load_json_object(ROOT / DATASET_MANIFEST_PATH),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
