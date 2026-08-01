"""Canonical experiment-manifest serialization and closed parse diagnostics."""

from __future__ import annotations

import json
from typing import Any, NoReturn

from pydantic import ValidationError

from expertforge.experiments.models import (
    EXPERIMENT_MANIFEST_FORMAT_VERSION,
    EXPERIMENT_MANIFEST_SCHEMA,
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    ExperimentManifest,
)

__all__ = [
    "MAX_MANIFEST_BYTES",
    "ManifestCorruptError",
    "ManifestError",
    "ManifestVersionError",
    "canonical_manifest_bytes",
    "parse_manifest_bytes",
]

MAX_MANIFEST_BYTES = 1024 * 1024


class ManifestError(Exception):
    """Base class for experiment-manifest failures."""


class ManifestVersionError(ManifestError):
    """A readable manifest declares an unsupported schema or format version."""


class ManifestCorruptError(ManifestError):
    """Manifest bytes are malformed, non-canonical, or semantically invalid."""


def canonical_manifest_bytes(manifest: ExperimentManifest) -> bytes:
    """Compact sorted-key UTF-8 JSON with non-finite numbers prohibited."""
    try:
        return json.dumps(
            manifest.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ManifestCorruptError(
            f"manifest could not be serialized canonically: {exc}"
        ) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestCorruptError(f"duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> NoReturn:
    raise ManifestCorruptError(f"non-finite JSON number {token!r} is forbidden.")


def _preflight_version(data: dict[str, Any]) -> None:
    schema = data.get("schema")
    schema_version = data.get("schema_version")
    format_version = data.get("format_version")
    if schema is None or schema_version is None or format_version is None:
        raise ManifestCorruptError(
            "manifest must declare schema, schema_version, and format_version."
        )
    if schema != EXPERIMENT_MANIFEST_SCHEMA:
        raise ManifestVersionError(
            f"unsupported manifest schema {schema!r}; expected {EXPERIMENT_MANIFEST_SCHEMA!r}."
        )
    if schema_version != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
        raise ManifestVersionError(
            f"unsupported manifest schema_version {schema_version!r}; "
            f"expected {EXPERIMENT_MANIFEST_SCHEMA_VERSION}."
        )
    if format_version != EXPERIMENT_MANIFEST_FORMAT_VERSION:
        raise ManifestVersionError(
            f"unsupported manifest format_version {format_version!r}; "
            f"expected {EXPERIMENT_MANIFEST_FORMAT_VERSION}."
        )


def parse_manifest_bytes(raw: bytes) -> ExperimentManifest:
    """Strictly parse canonical manifest bytes with version preflight."""
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ManifestCorruptError(
            f"manifest exceeds MAX_MANIFEST_BYTES={MAX_MANIFEST_BYTES}; got {len(raw)}."
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestCorruptError(f"manifest is not valid UTF-8: {exc}") from exc
    try:
        data = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except ManifestCorruptError:
        raise
    except json.JSONDecodeError as exc:
        raise ManifestCorruptError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestCorruptError("manifest root must be a JSON object.")
    _preflight_version(data)
    try:
        manifest = ExperimentManifest.model_validate_json(raw, strict=True)
    except ValidationError as exc:
        raise ManifestCorruptError(f"manifest failed strict validation: {exc}") from exc
    canonical = canonical_manifest_bytes(manifest)
    if canonical != raw:
        raise ManifestCorruptError("manifest bytes are not canonical compact sorted JSON.")
    return manifest
