"""Dotted-path ``--set`` overrides (Issue #5 decision §4).

Parsing is purely syntactic:

- split at the first ``=``;
- the value is parsed as JSON if it is valid JSON, otherwise kept as a literal
  string;
- duplicate paths are rejected (not last-write-wins).

Semantic validation (the path must be an existing *leaf* in the resolved schema)
is performed when overrides are applied to a configuration mapping — see
:mod:`expertforge.config.resolve`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

__all__ = ["OverrideError", "OverrideRecord", "parse_overrides"]


class OverrideError(Exception):
    """Raised when an override token is malformed or duplicates another."""


@dataclass(frozen=True)
class OverrideRecord:
    """A single parsed ``--set`` override, with both raw and typed forms."""

    raw_token: str
    path: str
    value: object

    @property
    def typed_repr(self) -> str:
        """The normalized typed value, serialized canonically.

        Non-finite floats are not representable in canonical form; JSON encoding
        of ``None``/bool/int/float/str is deterministic here.
        """
        if isinstance(self.value, float):
            # Reject non-finite floats explicitly — they must not silently
            # propagate into a configuration.
            if self.value != self.value or self.value in (float("inf"), float("-inf")):
                raise OverrideError(f"Override {self.raw_token!r} has a non-finite numeric value.")
        return json.dumps(self.value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _parse_value(raw: str) -> object:
    """Parse ``raw`` as JSON if it is valid JSON, else return it as a string.

    A bare unquoted word like ``smoke-a`` is not valid JSON, so it stays a
    literal string. ``7``, ``0.0003``, ``true``, ``null``, ``"DEBUG"`` are valid
    JSON and become typed Python values. The empty string ``""`` is not valid
    JSON, so an empty override value becomes the literal empty string.

    Non-finite float values (``Infinity``, ``-Infinity``, ``NaN``) are accepted
    by Python's JSON parser but are rejected here — they must never reach the
    configuration, where they would break canonicalization.
    """
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise OverrideError(
            f"Override value {raw!r} is a non-finite float (NaN/Infinity); "
            "non-finite numbers are not permitted in configuration overrides."
        )
    return value


def parse_overrides(tokens: list[str]) -> list[OverrideRecord]:
    """Parse a list of ``path=value`` override tokens into records.

    Raises :class:`OverrideError` for malformed tokens, duplicate paths, or
    non-finite numeric values.
    """
    records: list[OverrideRecord] = []
    seen_paths: set[str] = set()
    for token in tokens:
        if "=" not in token:
            raise OverrideError(f"Override {token!r} is missing '='; expected 'path=value'.")
        path, raw = token.split("=", 1)
        if not path:
            raise OverrideError(f"Override {token!r} has an empty path.")
        # An empty VALUE is permitted: it is not valid JSON, so it becomes the
        # literal empty string (the field still has to validate it).
        if path in seen_paths:
            raise OverrideError(
                f"Duplicate override for path {path!r}; "
                "overrides must not repeat a path (no last-write-wins)."
            )
        seen_paths.add(path)
        value = _parse_value(raw)
        records.append(OverrideRecord(raw_token=token, path=path, value=value))
    return records
