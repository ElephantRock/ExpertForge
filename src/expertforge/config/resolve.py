"""Configuration resolution and canonical behavioral bytes."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import BaseModel, ValidationError

from expertforge.config.models import ConfigRoot
from expertforge.config.overrides import OverrideError, OverrideRecord, parse_overrides
from expertforge.config.yaml_loader import RestrictedYAMLError, load_restricted_yaml

__all__ = [
    "ConfigResolutionError",
    "ResolutionEnvelope",
    "apply_overrides_to_mapping",
    "canonical_bytes",
    "resolve_config",
]


class ConfigResolutionError(Exception):
    """Raised when a configuration cannot be loaded, overridden, or validated."""


def _nested_model_type(annotation: Any) -> type[BaseModel] | None:
    """Return the nested model class for direct or Optional model annotations."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    args = get_args(annotation)
    non_none = [argument for argument in args if argument is not type(None)]
    if len(args) == 2 and len(non_none) == 1:
        candidate = non_none[0]
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _set_leaf_schema_validated(
    mapping: dict[str, Any], path: str, value: Any, raw_token: str
) -> None:
    """Apply a leaf override after validating the complete schema path."""
    parts = path.split(".")
    current_model: type[BaseModel] = ConfigRoot
    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        fields = current_model.model_fields
        if part not in fields:
            raise ConfigResolutionError(
                f"Override {raw_token!r} targets unknown path "
                f"{'.'.join(parts[: index + 1])!r}."
            )
        nested_model = _nested_model_type(fields[part].annotation)
        if is_last:
            if nested_model is not None:
                raise ConfigResolutionError(
                    f"Override {raw_token!r} targets non-leaf path {path!r}; "
                    "only leaf fields may be overridden."
                )
        else:
            if nested_model is None:
                raise ConfigResolutionError(
                    f"Override {raw_token!r} cannot descend into {path!r}: "
                    f"{'.'.join(parts[: index + 1])!r} is a leaf, not a section."
                )
            current_model = nested_model

    node: dict[str, Any] = mapping
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):  # pragma: no cover
            raise ConfigResolutionError(
                f"Override {raw_token!r} path {path!r} is internally inconsistent."
            )
    last = parts[-1]
    if isinstance(node.get(last), dict):
        raise ConfigResolutionError(
            f"Override {raw_token!r} targets non-leaf path {path!r}; "
            "only leaf fields may be overridden."
        )
    node[last] = value


def apply_overrides_to_mapping(
    mapping: dict[str, Any], override_tokens: list[str]
) -> dict[str, Any]:
    """Return a deep copy with schema-validated overrides applied."""
    output = copy.deepcopy(mapping)
    for record in parse_overrides(override_tokens):
        _set_leaf_schema_validated(output, record.path, record.value, record.raw_token)
    return output


@dataclass(frozen=True)
class ResolutionEnvelope:
    source_path: Path
    content_hash: str
    overrides: tuple[OverrideRecord, ...]
    config: ConfigRoot


def resolve_config(
    source: Path | str, override_tokens: list[str] | None = None
) -> ResolutionEnvelope:
    """Load, override, and validate a configuration source."""
    source_path = Path(source).resolve()
    raw_bytes = _read_bytes_or_raise(source_path)
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigResolutionError(
            f"Configuration file {source_path} is not valid UTF-8: {error}"
        ) from error

    try:
        mapping = load_restricted_yaml(raw_text)
    except RestrictedYAMLError as error:
        raise ConfigResolutionError(f"Could not parse {source_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigResolutionError(f"Could not parse {source_path}: {error}") from error

    try:
        overrides = parse_overrides(override_tokens or [])
        mapping = apply_overrides_to_mapping(mapping, [record.raw_token for record in overrides])
    except OverrideError as error:
        raise ConfigResolutionError(f"Invalid override: {error}") from error

    try:
        config = ConfigRoot.model_validate(mapping)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()
        )
        raise ConfigResolutionError(
            f"Configuration validation failed in {source_path}: {details}"
        ) from error

    return ResolutionEnvelope(
        source_path=source_path,
        content_hash=content_hash,
        overrides=tuple(overrides),
        config=config,
    )


def _read_bytes_or_raise(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as error:
        raise ConfigResolutionError(f"Configuration file not found: {path}") from error
    except OSError as error:
        raise ConfigResolutionError(f"Could not read configuration file {path}: {error}") from error


def canonical_bytes(envelope: ResolutionEnvelope) -> bytes:
    """Return compact sorted UTF-8 JSON of the effective configuration."""
    try:
        effective = envelope.config.model_dump(mode="json")
        return json.dumps(
            effective,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (UnicodeEncodeError, ValueError, TypeError) as error:
        raise ConfigResolutionError(
            f"Configuration could not be serialized to canonical bytes: {error}"
        ) from error
