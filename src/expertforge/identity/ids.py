"""Run and attempt identifiers (Issue #6 decision §2, §3).

Formats::

    run-YYYYMMDDtHHMMSSz-<spec-prefix-12>-<20-random-hex>
    attempt-YYYYMMDDtHHMMSSz-<20-random-hex>

The timestamp is for human inspection only. Uniqueness comes from 80
cryptographically secure random bits (20 hex chars = 10 bytes). Generation
supports injected clock and entropy providers, collision detection against an
optional ``exists`` predicate, bounded retries, and a typed exhaustion error.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from datetime import UTC, datetime

__all__ = [
    "DEFAULT_MAX_RETRIES",
    "IdentityCollisionError",
    "EntropyProvider",
    "ClockProvider",
    "ExistsPredicate",
    "attempt_id",
    "run_id",
]

DEFAULT_MAX_RETRIES = 64

# 80 cryptographically secure random bits = 10 bytes = 20 hex chars.
_RANDOM_BYTES = 10
_SPEC_PREFIX_LEN = 12


EntropyProvider = Callable[[int], bytes]
ClockProvider = Callable[[], datetime]
ExistsPredicate = Callable[[str], bool]


class IdentityCollisionError(Exception):
    """Raised when ID generation cannot find a non-colliding value within
    ``max_retries`` attempts."""


def _format_timestamp(dt: datetime) -> str:
    """Format a UTC datetime as ``YYYYMMDDtHHMMSSz`` (inspection-only)."""
    utc = dt.astimezone(UTC)
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
        spec_prefix: the first 12 hex chars of the specification fingerprint
            digest (for human inspection of which spec a run belongs to).
        clock: injectable UTC clock (default: ``datetime.now(timezone.utc)``).
        entropy: injectable cryptographically secure byte source
            (default: :mod:`secrets`).
        exists: optional collision predicate; if it returns True for a candidate,
            a new candidate is generated. ``None`` disables collision checking.
        max_retries: bounded retries before raising :class:`IdentityCollisionError`.
    """
    if len(spec_prefix) != _SPEC_PREFIX_LEN:
        raise ValueError(f"spec_prefix must be {_SPEC_PREFIX_LEN} chars; got {len(spec_prefix)}.")
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
