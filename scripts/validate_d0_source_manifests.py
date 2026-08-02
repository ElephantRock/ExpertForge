"""Validate the proposed D0 contract and immutable source identities.

The validator imports no model or training implementation. It verifies
content-addressed dataset and tokenizer manifests and their contract bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONTRACT_PATH = Path("experiments/d0/baseline-contract-v1.proposed.json")
TOKENIZER_MANIFEST_PATH = Path("tokenizers/manifests/d0-gpt-neox-v1.json")
DATASET_MANIFEST_PATH = Path("data/manifests/d0-fineweb-edu-sample-10bt-source-v1.json")
_SHA40_LENGTH = 40
_SHA256_LENGTH = 64
_EXPECTED_DATASET_REVISION = "84e8104e779e409e2267ac60609138e3dda2cbd2"


class ContractValidationError(ValueError):
    """Raised when the proposed D0 contract is internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _require_exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ContractValidationError(f"{field} must be an exact integer")
    if value < minimum:
        raise ContractValidationError(f"{field} must be >= {minimum}")
    return value


def _validate_hex_digest(value: object, field: str, length: int) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{field} must be a string")
    _require(len(value) == length, f"{field} must contain {length} hexadecimal characters")
    _require(
        all(character in "0123456789abcdef" for character in value),
        f"{field} must be lowercase hexadecimal",
    )
    return value


def _validate_revision(value: object, field: str) -> str:
    return _validate_hex_digest(value, field, _SHA40_LENGTH)


def canonical_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Hash canonical JSON after excluding the self-referential digest field."""

    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_file_inventory(
    manifest: Mapping[str, Any],
    *,
    manifest_name: str,
    expected_paths: Sequence[str],
) -> dict[str, int]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ContractValidationError(f"{manifest_name}.files must be a list")

    paths: list[str] = []
    total_size = 0
    for index, member in enumerate(files):
        if not isinstance(member, Mapping):
            raise ContractValidationError(f"{manifest_name}.files[{index}] must be an object")
        path = member.get("path")
        if not isinstance(path, str) or not path:
            raise ContractValidationError(f"{manifest_name}.files[{index}].path must be non-empty")
        paths.append(path)
        _validate_hex_digest(
            member.get("sha256"),
            f"{manifest_name}.files[{index}].sha256",
            _SHA256_LENGTH,
        )
        total_size += _require_exact_int(
            member.get("size_bytes"),
            f"{manifest_name}.files[{index}].size_bytes",
            minimum=1,
        )

    _require(paths == sorted(paths), f"{manifest_name} file paths must be lexicographically sorted")
    _require(len(paths) == len(set(paths)), f"{manifest_name} file paths must be unique")
    _require(paths == list(expected_paths), f"{manifest_name} file inventory changed")
    _require(manifest.get("file_count") == len(files), f"{manifest_name}.file_count mismatch")
    _require(manifest.get("total_size_bytes") == total_size, f"{manifest_name}.total_size_bytes mismatch")

    declared_digest = _validate_hex_digest(
        manifest.get("manifest_sha256"),
        f"{manifest_name}.manifest_sha256",
        _SHA256_LENGTH,
    )
    _require(
        declared_digest == canonical_manifest_digest(manifest),
        f"{manifest_name} canonical digest mismatch",
    )
    return {"file_count": len(files), "total_size_bytes": total_size}


def validate_source_manifests(
    contract: Mapping[str, Any],
    tokenizer_manifest: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
) -> dict[str, dict[str, int | str]]:
    """Validate source manifests and their binding to the proposed contract."""

    tokenizer = contract["tokenizer"]
    tokenizer_upstream = tokenizer_manifest["upstream"]
    tokenizer_semantics = tokenizer_manifest["tokenizer"]
    _require(
        tokenizer_manifest.get("schema_version") == "expertforge-tokenizer-source-manifest/1",
        "unexpected tokenizer manifest schema",
    )
    _require(tokenizer_upstream["repository"] == tokenizer["repository"], "tokenizer repository mismatch")
    _require(tokenizer_upstream["revision"] == tokenizer["revision"], "tokenizer revision mismatch")
    _require(tokenizer_semantics["family"] == tokenizer["family"], "tokenizer family mismatch")
    _require(
        tokenizer_semantics["vocabulary_size"] == tokenizer["vocabulary_size"],
        "tokenizer vocabulary mismatch",
    )
    _require(tokenizer_semantics["padding_token"] is None, "tokenizer manifest must not define padding")
    _require(
        tokenizer_semantics["training_document_boundary_token_id"] == tokenizer["eos_token_id"],
        "tokenizer document boundary mismatch",
    )
    expected_tokenizer_paths = sorted(tokenizer["files"])
    tokenizer_report = _validate_file_inventory(
        tokenizer_manifest,
        manifest_name="tokenizer_manifest",
        expected_paths=expected_tokenizer_paths,
    )
    _require(
        tokenizer["source_manifest_path"] == TOKENIZER_MANIFEST_PATH.as_posix(),
        "tokenizer manifest path mismatch",
    )
    _require(
        tokenizer["source_manifest_sha256"] == tokenizer_manifest["manifest_sha256"],
        "tokenizer manifest binding mismatch",
    )
    _require(
        tokenizer["ratification_requires_local_file_digests"] is False,
        "tokenizer identity blocker must be closed",
    )

    dataset = contract["dataset"]
    dataset_upstream = dataset_manifest["upstream"]
    _require(
        dataset_manifest.get("schema_version") == "expertforge-dataset-source-manifest/1",
        "unexpected dataset manifest schema",
    )
    _require(dataset["revision"] == _EXPECTED_DATASET_REVISION, "dataset revision is not the sample upload commit")
    _require(dataset_upstream["repository"] == dataset["repository"], "dataset repository mismatch")
    _require(dataset_upstream["revision"] == dataset["revision"], "dataset revision mismatch")
    _require(dataset_upstream["configuration"] == dataset["configuration"], "dataset configuration mismatch")
    _require(dataset_upstream["license"] == dataset["license_declaration"], "dataset license mismatch")
    _require(dataset_manifest["schema"]["consumed_fields"] == dataset["consumed_fields"], "dataset fields mismatch")
    _require(dataset_manifest["normalization"] == dataset["normalization"], "dataset normalization mismatch")
    _require(dataset_manifest["split_contract"] == dataset["split"], "dataset split mismatch")
    expected_dataset_paths = [f"sample/10BT/{index:03d}_00000.parquet" for index in range(14)]
    dataset_report = _validate_file_inventory(
        dataset_manifest,
        manifest_name="dataset_manifest",
        expected_paths=expected_dataset_paths,
    )
    _require(
        dataset["source_manifest_path"] == DATASET_MANIFEST_PATH.as_posix(),
        "dataset manifest path mismatch",
    )
    _require(
        dataset["source_manifest_sha256"] == dataset_manifest["manifest_sha256"],
        "dataset manifest binding mismatch",
    )
    _require(
        dataset["ratification_requires_local_source_manifest"] is False,
        "dataset identity blocker must be closed",
    )

    return {
        "tokenizer": {
            **tokenizer_report,
            "manifest_sha256": tokenizer_manifest["manifest_sha256"],
        },
        "dataset": {
            **dataset_report,
            "manifest_sha256": dataset_manifest["manifest_sha256"],
        },
    }


def load_json_object(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractValidationError(f"{path} root must be an object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate D0 immutable source manifests")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--tokenizer-manifest", type=Path, default=TOKENIZER_MANIFEST_PATH)
    parser.add_argument("--dataset-manifest", type=Path, default=DATASET_MANIFEST_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_source_manifests(
            load_json_object(args.contract),
            load_json_object(args.tokenizer_manifest),
            load_json_object(args.dataset_manifest),
        )
    except (ContractValidationError, KeyError, OSError, json.JSONDecodeError) as error:
        sys.stderr.write(f"D0 SOURCE MANIFESTS INVALID: {error}\n")
        return 1
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
