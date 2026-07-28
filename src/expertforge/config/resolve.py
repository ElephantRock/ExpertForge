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

from pydantic import ValidationError

from expertforge.config.models import ConfigRoot
from expertforge.config.overrides import OverrideRecord, parse_overrides
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


# --- override application with path validation -----------------------------


def _set_leaf(mapping: dict[str, Any], path: str, value: Any, raw_token: str) -> None:
    """Walk ``path`` (dotted) within ``mapping`` and set the leaf to ``value``.

    Rejects unknown paths (key absent) and non-leaf targets (the path does not
    reach a scalar — overwriting a whole section is forbidden).
    """
    parts = path.split(".")
    node: Any = mapping
    walked: list[str] = []
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        if not isinstance(node, dict) or part not in node:
            raise ConfigResolutionError(
                f"Override {raw_token!r} targets unknown path {'.'.join(walked + [part])!r}."
            )
        walked.append(part)
        if is_last:
            current = node[part]
            if isinstance(current, dict):
                raise ConfigResolutionError(
                    f"Override {raw_token!r} targets non-leaf path {path!r}; "
                    "only existing leaf fields may be overridden."
                )
            node[part] = value
        else:
            if not isinstance(node[part], dict):
                # An intermediate part points at a scalar, not a section.
                raise ConfigResolutionError(
                    f"Override {raw_token!r} cannot descend into {path!r}: "
                    f"{'.'.join(walked)!r} is a leaf, not a section."
                )
            node = node[part]


def apply_overrides_to_mapping(
    mapping: dict[str, Any], override_tokens: list[str]
) -> dict[str, Any]:
    """Return a deep copy of ``mapping`` with ``override_tokens`` applied.

    Validates each override path is an existing leaf before applying.
    """
    out = copy.deepcopy(mapping)
    records = parse_overrides(override_tokens)
    for rec in records:
        _set_leaf(out, rec.path, rec.value, rec.raw_token)
    return out


# --- resolution envelope ---------------------------------------------------


@dataclass(frozen=True)
class ResolutionEnvelope:
    """Provenance-rich resolution result.

    Attributes:
        source_path: resolved absolute path of the source YAML file.
        content_hash: SHA-256 hex of the raw source bytes (pre-override).
        overrides: parsed override records actually applied.
        config: the fully resolved, validated configuration model.
    """

    source_path: Path
    content_hash: str
    overrides: list[OverrideRecord]
    config: ConfigRoot


def resolve_config(
    source: Path | str, override_tokens: list[str] | None = None
) -> ResolutionEnvelope:
    """Load, override, and validate ``source`` into a resolution envelope."""
    source_path = Path(source).resolve()
    raw_text = source_path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    try:
        mapping = load_restricted_yaml(raw_text)
    except RestrictedYAMLError as e:
        raise ConfigResolutionError(f"Could not parse {source_path}: {e}") from e

    overrides = parse_overrides(override_tokens or [])
    mapping = apply_overrides_to_mapping(mapping, [r.raw_token for r in overrides])

    try:
        config = ConfigRoot.model_validate(mapping)
    except ValidationError as e:
        # Surface the offending field path(s) and their messages — both the loc
        # and the human-readable msg, so diagnostics name the exact field.
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise ConfigResolutionError(
            f"Configuration validation failed in {source_path}: {details}"
        ) from e

    return ResolutionEnvelope(
        source_path=source_path,
        content_hash=content_hash,
        overrides=list(overrides),
        config=config,
    )


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
