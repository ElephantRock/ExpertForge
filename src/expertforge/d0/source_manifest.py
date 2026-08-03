"""Parsing and fail-closed validation for frozen D0 source manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from expertforge.d0.errors import SourceManifestError

_HEX = frozenset("0123456789abcdef")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize canonical UTF-8 JSON using the repository digest policy."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SourceManifestError("value is not canonical-JSON serializable") from exc


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceManifestError(f"{field} must be a JSON object")
    return value


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SourceManifestError(f"{field} must be a JSON array")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceManifestError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SourceManifestError(f"{field} must be an integer >= {minimum}")
    return value


def _digest(value: object, field: str) -> str:
    digest = _string(value, field)
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise SourceManifestError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _inventory_path(value: object, field: str) -> str:
    path = _string(value, field)
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise SourceManifestError(f"{field} must be a normalized relative POSIX path")
    if parsed.as_posix() != path:
        raise SourceManifestError(f"{field} must use normalized POSIX separators")
    return path


@dataclass(frozen=True, slots=True)
class SourceFileIdentity:
    """One immutable upstream file identity."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """A verified dataset or tokenizer source inventory."""

    kind: Literal["dataset", "tokenizer"]
    schema_version: str
    manifest_id: str
    repository: str
    revision: str
    license: str
    configuration: str | None
    split: str | None
    source_root: str | None
    files: tuple[SourceFileIdentity, ...]
    total_size_bytes: int
    manifest_sha256: str

    def file(self, path: str) -> SourceFileIdentity:
        """Return one exact file identity or fail closed."""

        for identity in self.files:
            if identity.path == path:
                return identity
        raise SourceManifestError(f"source path is not in frozen manifest: {path}")


def _manifest_kind(schema_version: str) -> Literal["dataset", "tokenizer"]:
    if schema_version == "expertforge-dataset-source-manifest/1":
        return "dataset"
    if schema_version == "expertforge-tokenizer-source-manifest/1":
        return "tokenizer"
    raise SourceManifestError(f"unsupported source-manifest schema: {schema_version}")


def _validate_dataset_contract(manifest: Mapping[str, Any]) -> None:
    ordering = _mapping(manifest.get("ordering"), "ordering")
    if dict(ordering) != {
        "source_files": "path_lexicographic",
        "rows_within_file": "physical_row_index_ascending",
    }:
        raise SourceManifestError("dataset source ordering contract changed")
    normalization = _mapping(manifest.get("normalization"), "normalization")
    if dict(normalization) != {
        "utf8_required": True,
        "nul_rejected": True,
        "newline_mapping": "CRLF_and_CR_to_LF",
        "other_codepoints_and_whitespace": "preserve_exactly",
    }:
        raise SourceManifestError("dataset normalization contract changed")
    deduplication = _mapping(manifest.get("deduplication"), "deduplication")
    if dict(deduplication) != {
        "duplicate_key": "sha256(normalized_utf8_text)",
        "keep_rule": "first_in_source_order",
    }:
        raise SourceManifestError("dataset duplicate-retention contract changed")


def parse_source_manifest(
    value: Mapping[str, Any],
    *,
    expected_manifest_sha256: str | None = None,
) -> SourceManifest:
    """Validate and convert one frozen source manifest."""

    schema_version = _string(value.get("schema_version"), "schema_version")
    kind = _manifest_kind(schema_version)
    if value.get("digest_policy") != "sha256(canonical_json_without_manifest_sha256)":
        raise SourceManifestError("source-manifest digest policy changed")

    declared_digest = _digest(value.get("manifest_sha256"), "manifest_sha256")
    canonical_payload = dict(value)
    canonical_payload.pop("manifest_sha256", None)
    actual_digest = hashlib.sha256(canonical_json_bytes(canonical_payload)).hexdigest()
    if declared_digest != actual_digest:
        raise SourceManifestError(
            f"source-manifest canonical digest mismatch: expected {declared_digest}, "
            f"actual {actual_digest}"
        )
    if expected_manifest_sha256 is not None and declared_digest != _digest(
        expected_manifest_sha256, "expected_manifest_sha256"
    ):
        raise SourceManifestError("source-manifest identity does not match contract binding")

    upstream = _mapping(value.get("upstream"), "upstream")
    repository = _string(upstream.get("repository"), "upstream.repository")
    revision = _string(upstream.get("revision"), "upstream.revision")
    license_name = _string(upstream.get("license"), "upstream.license")
    configuration = _optional_string(upstream.get("configuration"), "upstream.configuration")
    split = _optional_string(upstream.get("split"), "upstream.split")
    source_root = _optional_string(upstream.get("source_root"), "upstream.source_root")

    raw_files = _sequence(value.get("files"), "files")
    files: list[SourceFileIdentity] = []
    paths: list[str] = []
    total_size_bytes = 0
    for index, raw_file in enumerate(raw_files):
        entry = _mapping(raw_file, f"files[{index}]")
        if set(entry) != {"path", "sha256", "size_bytes"}:
            raise SourceManifestError(f"files[{index}] fields changed")
        path = _inventory_path(entry.get("path"), f"files[{index}].path")
        identity = SourceFileIdentity(
            path=path,
            sha256=_digest(entry.get("sha256"), f"files[{index}].sha256"),
            size_bytes=_exact_int(entry.get("size_bytes"), f"files[{index}].size_bytes"),
        )
        files.append(identity)
        paths.append(path)
        total_size_bytes += identity.size_bytes

    if not files:
        raise SourceManifestError("source manifest must contain at least one file")
    if len(set(paths)) != len(paths):
        raise SourceManifestError("source-manifest paths must be unique")
    if paths != sorted(paths):
        raise SourceManifestError("source-manifest paths must be lexicographically sorted")
    if _exact_int(value.get("file_count"), "file_count") != len(files):
        raise SourceManifestError("source-manifest file_count mismatch")
    if _exact_int(value.get("total_size_bytes"), "total_size_bytes") != total_size_bytes:
        raise SourceManifestError("source-manifest total_size_bytes mismatch")

    if kind == "dataset":
        if configuration is None or split is None or source_root is None:
            raise SourceManifestError("dataset upstream identity is incomplete")
        _validate_dataset_contract(value)
    elif configuration is not None or split is not None or source_root is not None:
        raise SourceManifestError("tokenizer upstream identity contains dataset-only fields")

    return SourceManifest(
        kind=kind,
        schema_version=schema_version,
        manifest_id=_string(value.get("manifest_id"), "manifest_id"),
        repository=repository,
        revision=revision,
        license=license_name,
        configuration=configuration,
        split=split,
        source_root=source_root,
        files=tuple(files),
        total_size_bytes=total_size_bytes,
        manifest_sha256=declared_digest,
    )


def load_source_manifest(
    path: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> SourceManifest:
    """Read strict UTF-8 JSON and validate one source manifest."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SourceManifestError(f"cannot read source manifest {path}: {exc}") from exc
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceManifestError(f"invalid source manifest {path}: {exc}") from exc
    return parse_source_manifest(
        _mapping(value, str(path)),
        expected_manifest_sha256=expected_manifest_sha256,
    )
