"""Configuration resolution: the resolution envelope and canonical bytes
(Issue #5 decision §canonical-representation).

Two logically separate representations are produced:

1. :class:`ResolutionEnvelope` — provenance-rich: the source path, the SHA-256
   content hash of the raw source text, the raw + normalized override records,
   and the fully resolved :class:`~expertforge.config.models.ConfigRoot`.

2. :func:`canonical_bytes` — behavior-only: the effective configuration alone,
   serialized as deterministic UTF-8 JSON with sorted keys, compact separators,
   and no non-finite numbers. Two different sources or override sequences that
   resolve to the same effective configuration produce identical canonical
   bytes (the fingerprint input for Issue #6).

Overrides are applied to the raw mapping *before* Pydantic validation, so
type coercion and cross-field checks see the overridden values. Path validation
rejects unknown paths and non-leaf (section) targets.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from expertforge.config.models import ConfigRoot
from expertforge.config.overrides import OverrideError, OverrideRecord, parse_overrides
from expertforge.config.yaml_loader import (
    RestrictedYAMLError,
    load_restricted_yaml,
)

__all__ = [
    "ConfigResolutionError",
    "ResolutionEnvelope",
    "apply_overrides_to_mapping",
    "canonical_bytes",
    "resolve_config",
]


class ConfigResolutionError(Exception):
    """Raised when a configuration cannot be loaded, overridden, or validated."""


# --- override application with schema-driven path validation ---------------


def _set_leaf_schema_validated(
    mapping: dict[str, Any], path: str, value: Any, raw_token: str
) -> None:
    """Apply ``path=value`` to ``mapping``, validating the path against the
    :class:`ConfigRoot` schema.

    The override surface is defined by the schema, not by what the author wrote
    into YAML: omitted sections whose fields have schema defaults may still be
    overridden. Unknown paths and non-leaf (section) targets are rejected.
    Intermediate sections are materialized as empty dicts so the leaf can be set.
    """
    parts = path.split(".")
    # Walk the schema from ConfigRoot downward to confirm the path is valid and
    # reaches a leaf.
    current_model: type[BaseModel] = ConfigRoot
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        fields = current_model.model_fields
        if part not in fields:
            raise ConfigResolutionError(
                f"Override {raw_token!r} targets unknown path {'.'.join(parts[: i + 1])!r}."
            )
        field_info = fields[part]
        ann = field_info.annotation
        if is_last:
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                raise ConfigResolutionError(
                    f"Override {raw_token!r} targets non-leaf path {path!r}; "
                    "only leaf fields may be overridden."
                )
        else:
            # Must descend into a nested section model.
            if not (isinstance(ann, type) and issubclass(ann, BaseModel)):
                raise ConfigResolutionError(
                    f"Override {raw_token!r} cannot descend into {path!r}: "
                    f"{'.'.join(parts[: i + 1])!r} is a leaf, not a section."
                )
            current_model = ann

    # Materialize any omitted intermediate sections in the raw mapping, then set.
    node: dict[str, Any] = mapping
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):  # pragma: no cover - schema walk guards this
            raise ConfigResolutionError(
                f"Override {raw_token!r} path {path!r} is internally inconsistent."
            )
    last = parts[-1]
    # If the leaf exists and is itself a mapping, the path targeted a section
    # despite passing the schema check (shouldn't happen, but guard).
    existing = node.get(last)
    if isinstance(existing, dict):
        raise ConfigResolutionError(
            f"Override {raw_token!r} targets non-leaf path {path!r}; "
            "only leaf fields may be overridden."
        )
    node[last] = value


def apply_overrides_to_mapping(
    mapping: dict[str, Any], override_tokens: list[str]
) -> dict[str, Any]:
    """Return a deep copy of ``mapping`` with ``override_tokens`` applied.

    Validates each override path against the :class:`ConfigRoot` schema before
    applying; omitted defaulted sections may be overridden.
    """
    out = copy.deepcopy(mapping)
    records = parse_overrides(override_tokens)
    for rec in records:
        _set_leaf_schema_validated(out, rec.path, rec.value, rec.raw_token)
    return out


# --- resolution envelope ---------------------------------------------------


@dataclass(frozen=True)
class ResolutionEnvelope:
    """Provenance-rich resolution result. Deeply immutable.

    Attributes:
        source_path: resolved absolute path of the source YAML file.
        content_hash: SHA-256 hex of the raw source BYTES (pre-override).
        overrides: parsed override records actually applied (immutable tuple).
        config: the fully resolved, validated configuration model.
    """

    source_path: Path
    content_hash: str
    overrides: tuple[OverrideRecord, ...]
    config: ConfigRoot


def resolve_config(
    source: Path | str, override_tokens: list[str] | None = None
) -> ResolutionEnvelope:
    """Load, override, and validate ``source`` into a resolution envelope."""
    source_path = Path(source).resolve()
    # Hash the raw source BYTES so LF-vs-CRLF variants hash distinctly.
    raw_bytes = _read_bytes_or_raise(source_path)
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ConfigResolutionError(
            f"Configuration file {source_path} is not valid UTF-8: {e}"
        ) from e

    try:
        mapping = load_restricted_yaml(raw_text)
    except RestrictedYAMLError as e:
        raise ConfigResolutionError(f"Could not parse {source_path}: {e}") from e
    except yaml.YAMLError as e:
        # Malformed YAML (parser/scanner errors) — surface as a clean diagnostic
        # rather than leaking a PyYAML traceback.
        raise ConfigResolutionError(f"Could not parse {source_path}: {e}") from e

    try:
        overrides = parse_overrides(override_tokens or [])
        mapping = apply_overrides_to_mapping(mapping, [r.raw_token for r in overrides])
    except OverrideError as e:
        raise ConfigResolutionError(f"Invalid override: {e}") from e

    try:
        config = ConfigRoot.model_validate(mapping)
    except ValidationError as e:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise ConfigResolutionError(
            f"Configuration validation failed in {source_path}: {details}"
        ) from e

    return ResolutionEnvelope(
        source_path=source_path,
        content_hash=content_hash,
        overrides=tuple(overrides),
        config=config,
    )


def _read_bytes_or_raise(path: Path) -> bytes:
    """Read ``path`` as bytes, raising :class:`ConfigResolutionError` on I/O or
    decode failure so callers stay within the error boundary."""
    try:
        return path.read_bytes()
    except FileNotFoundError as e:
        raise ConfigResolutionError(f"Configuration file not found: {path}") from e
    except OSError as e:
        raise ConfigResolutionError(f"Could not read configuration file {path}: {e}") from e


# --- behavioral canonical bytes -------------------------------------------


def canonical_bytes(envelope: ResolutionEnvelope) -> bytes:
    """Deterministic UTF-8 JSON of the *effective* configuration only.

    Sorted keys, compact separators, non-finite numbers prohibited. The
    envelope's provenance (source path, hash, overrides) is deliberately
    excluded — only behavioral config contributes to the fingerprint.
    """
    effective = envelope.config.model_dump(mode="json")
    return json.dumps(
        effective,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
