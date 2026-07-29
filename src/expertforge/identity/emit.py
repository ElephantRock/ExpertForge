"""Identity emit orchestrator (Issue #6 decision §5).

Wires the identity subsystem together so a caller can emit a complete, durable,
reconstructable per-attempt identity for a resolved configuration in one step::

    resolved configuration envelope
    → specification fingerprint
    → spec-prefix (first 12 hex)
    → run_id + attempt_id
    → AttemptIdentityRecord (+ optional resume lineage)
    → identity sidecar (exclusive, deterministic)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from expertforge.config.resolve import ResolutionEnvelope, canonical_bytes
from expertforge.identity.fingerprint import (
    ImmutableInput,
    specification_fingerprint,
)
from expertforge.identity.ids import (
    DEFAULT_MAX_RETRIES,
    ClockProvider,
    EntropyProvider,
    ExistsPredicate,
    IdentityCollisionError,
    attempt_id,
    run_id,
)
from expertforge.identity.lineage import ResumeLineage
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.identity.sidecar import IdentitySidecarError, write_identity_sidecar

__all__ = ["IdentityEmitError", "emit_attempt_identity"]

_SPEC_PREFIX_LEN = 12


class IdentityEmitError(Exception):
    """Raised when a complete identity cannot be emitted (collision exhaustion,
    sidecar failure, etc.)."""


def _spec_prefix(digest_str: str) -> str:
    """Extract the 12-hex spec prefix from a ``spec-v1-sha256-<hex>`` digest."""
    hexpart = digest_str.rsplit("-", 1)[-1]
    return hexpart[:_SPEC_PREFIX_LEN]


def emit_attempt_identity(
    *,
    artifact_root: Path,
    config_envelope: ResolutionEnvelope,
    lineage: ResumeLineage | None = None,
    immutable_inputs: list[ImmutableInput] | None = None,
    clock: ClockProvider | None = None,
    entropy: EntropyProvider | None = None,
    exists: ExistsPredicate | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[AttemptIdentityRecord, Path]:
    """Compute and persist a per-attempt identity for a resolved configuration.

    Returns the :class:`AttemptIdentityRecord` and the sidecar path. The sidecar
    is written before this returns, so the identity is durable by the time
    training would begin. The same ``clock`` is used for the record's
    ``created_at_utc`` and for the timestamps embedded in the run/attempt IDs,
    keeping them consistent.
    """
    resolved_clock: ClockProvider = clock or (lambda: datetime.now(UTC))
    cb = canonical_bytes(config_envelope)
    fp = specification_fingerprint(cb, immutable_inputs=immutable_inputs)
    prefix = _spec_prefix(fp.digest_str)
    now = resolved_clock()

    try:
        rid = run_id(
            prefix, clock=resolved_clock, entropy=entropy, exists=exists, max_retries=max_retries
        )
        aid = attempt_id(
            clock=resolved_clock, entropy=entropy, exists=exists, max_retries=max_retries
        )
    except IdentityCollisionError as e:
        raise IdentityEmitError(str(e)) from e

    record = AttemptIdentityRecord(
        specification_fingerprint=fp.digest_str,
        run_id=rid,
        attempt_id=aid,
        created_at_utc=now,
        lineage=lineage,
    )
    try:
        path = write_identity_sidecar(artifact_root, record)
    except IdentitySidecarError as e:
        raise IdentityEmitError(str(e)) from e
    return record, path
