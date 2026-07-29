"""Run and attempt identifiers (Issue #6 decision §2, §3; review item 5).

Formats::

    run-YYYYMMDDtHHMMSSz-<spec-prefix-12>-<20-random-hex>
    attempt-YYYYMMDDtHHMMSSz-<20-random-hex>

The timestamp is for human inspection only. Uniqueness comes from 80
cryptographically secure random bits (20 hex chars = 10 bytes). Generation
supports injected clock and entropy providers, collision detection against an
optional ``exists`` predicate, bounded retries, and a typed exhaustion error.

Clocks must return timezone-aware datetimes; naive datetimes are rejected so
timestamps are never interpreted via the host timezone. ``spec_prefix`` must be
exactly 12 lowercase hex characters. Generated IDs match strict lowercase
path-safe regexes (exported below) enforced at every boundary.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from datetime import UTC, datetime

__all__ = [
    "ATTEMPT_ID_PATTERN",
    "DEFAULT_MAX_RETRIES",
    "EntropyProvider",
    "ClockProvider",
    "ExistsPredicate",
    "IdentityCollisionError",
    "RUN_ID_PATTERN",
    "SPEC_PREFIX_PATTERN",
    "attempt_id",
    "run_id",
    "validate_run_id",
    "validate_attempt_id",
]

DEFAULT_MAX_RETRIES = 64

# 80 cryptographically secure random bits = 10 bytes = 20 hex chars.
_RANDOM_BYTES = 10
_SPEC_PREFIX_LEN = 12

# Strict lowercase path-safe patterns. The timestamp is 8 digits (date) + 't'
# + 6 digits (time) + 'z'. Run IDs carry a 12-hex spec prefix and 20-hex random;
# attempt IDs carry only the 20-hex random.
_TS = r"\d{8}t\d{6}z"
_HEX20 = r"[0-9a-f]{20}"
HEX12 = r"[0-9a-f]{12}"
RUN_ID_PATTERN = re.compile(rf"^run-{_TS}-{HEX12}-{_HEX20}$")
ATTEMPT_ID_PATTERN = re.compile(rf"^attempt-{_TS}-{_HEX20}$")
SPEC_PREFIX_PATTERN = re.compile(rf"^{HEX12}$")

EntropyProvider = Callable[[int], bytes]
ClockProvider = Callable[[], datetime]
ExistsPredicate = Callable[[str], bool]


class IdentityCollisionError(Exception):
    """Raised when ID generation cannot find a non-colliding value within
    ``max_retries`` attempts."""


def validate_run_id(value: str) -> None:
    """Raise ValueError if ``value`` is not a valid run ID."""
    if not RUN_ID_PATTERN.match(value):
        raise ValueError(f"Invalid run_id {value!r}; must match {RUN_ID_PATTERN.pattern}.")


def validate_attempt_id(value: str) -> None:
    """Raise ValueError if ``value`` is not a valid attempt ID."""
    if not ATTEMPT_ID_PATTERN.match(value):
        raise ValueError(f"Invalid attempt_id {value!r}; must match {ATTEMPT_ID_PATTERN.pattern}.")


def _validate_spec_prefix(spec_prefix: str) -> None:
    if not SPEC_PREFIX_PATTERN.match(spec_prefix):
        raise ValueError(
            f"spec_prefix must be {_SPEC_PREFIX_LEN} lowercase hex chars; got {spec_prefix!r}."
        )


def _require_aware(dt: datetime) -> datetime:
    """Reject naive datetimes so timestamps are never host-timezone-interpreted."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(
            f"Clock returned a naive datetime {dt!r}; timezone-aware UTC input is required."
        )
    return dt


def _format_timestamp(dt: datetime) -> str:
    """Format a timezone-aware datetime as ``YYYYMMDDtHHMMSSz`` (UTC)."""
    utc = _require_aware(dt).astimezone(UTC)
    return utc.strftime("%Y%m%dt%H%M%Sz")


def _random_hex(entropy: EntropyProvider) -> str:
    raw = entropy(_RANDOM_BYTES)
    if len(raw) != _RANDOM_BYTES:
        raise ValueError(f"Entropy provider returned {len(raw)} bytes; expected {_RANDOM_BYTES}.")
    return raw.hex()


def run_id(
    spec_prefix: str,
    *,
    clock: ClockProvider | None = None,
    entropy: EntropyProvider | None = None,
    exists: ExistsPredicate | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    """Generate a run ID.

    Args:
        spec_prefix: the first 12 lowercase hex chars of the specification
            fingerprint digest (for human inspection of which spec a run belongs to).
        clock: injectable timezone-aware UTC clock (default: ``datetime.now(UTC)``).
        entropy: injectable cryptographically secure byte source
            (default: :mod:`secrets`).
        exists: optional collision predicate; if it returns True for a candidate,
            a new candidate is generated. ``None`` disables collision checking.
        max_retries: bounded retries before raising :class:`IdentityCollisionError`.
    """
    _validate_spec_prefix(spec_prefix)
    _clock = clock or (lambda: datetime.now(UTC))
    _entropy = entropy or (lambda n: secrets.token_bytes(n))
    ts = _format_timestamp(_clock())
    for _ in range(max_retries):
        candidate = f"run-{ts}-{spec_prefix}-{_random_hex(_entropy)}"
        if exists is None or not exists(candidate):
            return candidate
    raise IdentityCollisionError(
        f"Could not generate a non-colliding run ID after {max_retries} attempts."
    )


def attempt_id(
    *,
    clock: ClockProvider | None = None,
    entropy: EntropyProvider | None = None,
    exists: ExistsPredicate | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    """Generate an attempt ID (one process invocation within a run)."""
    _clock = clock or (lambda: datetime.now(UTC))
    _entropy = entropy or (lambda n: secrets.token_bytes(n))
    ts = _format_timestamp(_clock())
    for _ in range(max_retries):
        candidate = f"attempt-{ts}-{_random_hex(_entropy)}"
        if exists is None or not exists(candidate):
            return candidate
    raise IdentityCollisionError(
        f"Could not generate a non-colliding attempt ID after {max_retries} attempts."
    )
