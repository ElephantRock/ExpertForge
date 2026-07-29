"""Identity emit orchestrator (Issue #6 decision §5; review item 1).

Wires the identity subsystem together so a caller can emit a complete, durable,
reconstructable per-attempt identity for a resolved configuration.

Allocation is explicit by **mode** (not inferred from nullable lineage), so the
run-versus-attempt contract is unambiguous:

- **independent**: new run ID + new attempt ID, no lineage.
- **resume**: retain the parent run ID, allocate a new attempt ID, record full
  native lineage (all three parent fields). The computed specification
  fingerprint is verified against the parent run's specification fingerprint.
- **fork**: new run ID + new attempt ID, with explicit parent lineage (full
  native). Used when the specification is materially changed but parentage is
  recorded.
- **legacy**: like resume, but ``parent_attempt_id`` may be ``None`` for
  imported/legacy provenance where the parent attempt cannot be reconstructed.
  The specification fingerprint is still verified against the parent run.

The same timezone-aware clock instant is used for ``created_at_utc`` and the
timestamps embedded in the run/attempt IDs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from expertforge.config.resolve import ResolutionEnvelope, canonical_bytes
from expertforge.identity.fingerprint import (
    FingerprintMismatch,
    ImmutableInput,
    SpecificationFingerprintRecord,
    fingerprint_id_prefix,
    specification_fingerprint,
    validate_fingerprint_id,
)
from expertforge.identity.ids import (
    DEFAULT_MAX_RETRIES,
    ClockProvider,
    EntropyProvider,
    ExistsPredicate,
    IdentityCollisionError,
    attempt_id,
    run_id,
    run_id_spec_prefix,
)
from expertforge.identity.lineage import ResumeLineage
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.identity.sidecar import IdentitySidecarError, write_identity_sidecar

__all__ = ["AllocationMode", "IdentityEmitError", "emit_attempt_identity"]

_SPEC_PREFIX_LEN = 12


class AllocationMode(Enum):
    """How a run/attempt identity is allocated."""

    INDEPENDENT = "independent"
    RESUME = "resume"
    FORK = "fork"
    LEGACY = "legacy"


class IdentityEmitError(Exception):
    """Raised when a complete identity cannot be emitted (collision exhaustion,
    spec-fingerprint mismatch on resume/fork/legacy, sidecar failure, etc.)."""


def _spec_prefix(digest_str: str) -> str:
    """Extract the 12-hex spec prefix from a ``spec-v1-sha256-<hex>`` digest."""
    hexpart = digest_str.rsplit("-", 1)[-1]
    return hexpart[:_SPEC_PREFIX_LEN]


def _require_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise IdentityEmitError(
            f"Clock returned a naive datetime {dt!r}; timezone-aware UTC input is required."
        )
    return dt


def emit_attempt_identity(
    *,
    artifact_root: Path,
    config_envelope: ResolutionEnvelope,
    mode: AllocationMode = AllocationMode.INDEPENDENT,
    lineage: ResumeLineage | None = None,
    parent_specification_fingerprint: str | None = None,
    retained_run_id: str | None = None,
    immutable_inputs: list[ImmutableInput] | None = None,
    clock: ClockProvider | None = None,
    entropy: EntropyProvider | None = None,
    exists: ExistsPredicate | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[AttemptIdentityRecord, Path]:
    """Compute and persist a per-attempt identity for a resolved configuration.

    Args:
        mode: allocation mode (see :class:`AllocationMode`).
        lineage: required for RESUME/FORK/LEGACY; must be None for INDEPENDENT.
            Native RESUME/FORK require ``parent_attempt_id``; LEGACY permits None.
        parent_specification_fingerprint: the parent run's specification
            fingerprint digest, required for RESUME/FORK/LEGACY. The computed
            fingerprint must match it (a materially changed specification is a
            fork and must use FORK with a fresh parent reference).
        retained_run_id: for RESUME/LEGACY, the existing run ID to retain.
            Required for those modes; must be None for INDEPENDENT/FORK.
    """
    resolved_clock: ClockProvider = clock or (lambda: datetime.now(UTC))
    # Capture exactly ONE clock instant and reuse it everywhere (created_at_utc
    # and both ID timestamps). The ID generators receive a constant callable
    # returning this same instant, so an advancing clock cannot produce
    # inconsistent timestamps within one emit.
    now = _require_aware(resolved_clock()).astimezone(UTC)

    def frozen_clock() -> datetime:
        return now

    cb = canonical_bytes(config_envelope)
    fp: SpecificationFingerprintRecord = specification_fingerprint(
        cb, immutable_inputs=immutable_inputs
    )
    prefix = _spec_prefix(fp.digest_str)

    # Validate mode-specific requirements and compute the run_id + lineage.
    if mode is AllocationMode.INDEPENDENT:
        if (
            lineage is not None
            or retained_run_id is not None
            or parent_specification_fingerprint is not None
        ):
            raise IdentityEmitError(
                "INDEPENDENT mode takes no lineage, retained_run_id, or parent fingerprint."
            )
        rid = _generate_run_id(prefix, frozen_clock, entropy, exists, max_retries)
        record_lineage: ResumeLineage | None = None

    elif mode is AllocationMode.RESUME:
        if lineage is None:
            raise IdentityEmitError("RESUME mode requires lineage with all three parent fields.")
        if lineage.parent_attempt_id is None:
            raise IdentityEmitError(
                "RESUME (native) requires parent_attempt_id; use LEGACY for imported provenance."
            )
        if retained_run_id is None:
            raise IdentityEmitError("RESUME mode requires retained_run_id.")
        if parent_specification_fingerprint is None:
            raise IdentityEmitError("RESUME mode requires parent_specification_fingerprint.")
        _verify_parent_spec(fp, parent_specification_fingerprint, mode)
        if lineage.parent_run_id != retained_run_id:
            raise IdentityEmitError("RESUME lineage.parent_run_id must equal retained_run_id.")
        _require_retained_run_prefix(retained_run_id, prefix, mode)
        _require_parent_run_prefix(lineage.parent_run_id, parent_specification_fingerprint, mode)
        rid = retained_run_id
        record_lineage = lineage

    elif mode is AllocationMode.FORK:
        if lineage is None:
            raise IdentityEmitError("FORK mode requires lineage with all three parent fields.")
        if lineage.parent_attempt_id is None:
            raise IdentityEmitError(
                "FORK (native) requires parent_attempt_id; use LEGACY for imported provenance."
            )
        if retained_run_id is not None:
            raise IdentityEmitError(
                "FORK mode allocates a new run_id; do not pass retained_run_id."
            )
        if parent_specification_fingerprint is None:
            raise IdentityEmitError("FORK mode requires parent_specification_fingerprint.")
        # FORK does NOT verify the current spec matches the parent's (the
        # specification is materially changed), but the parent fingerprint ID
        # format must be valid and the parent run's embedded prefix must match
        # the parent fingerprint's prefix.
        try:
            validate_fingerprint_id(parent_specification_fingerprint)
        except ValueError as e:
            raise IdentityEmitError(str(e)) from e
        _require_parent_run_prefix(lineage.parent_run_id, parent_specification_fingerprint, mode)
        rid = _generate_run_id(prefix, frozen_clock, entropy, exists, max_retries)
        record_lineage = lineage

    elif mode is AllocationMode.LEGACY:
        if lineage is None:
            raise IdentityEmitError("LEGACY mode requires lineage.")
        if retained_run_id is None:
            raise IdentityEmitError("LEGACY mode requires retained_run_id.")
        if parent_specification_fingerprint is None:
            raise IdentityEmitError("LEGACY mode requires parent_specification_fingerprint.")
        _verify_parent_spec(fp, parent_specification_fingerprint, mode)
        if lineage.parent_run_id != retained_run_id:
            raise IdentityEmitError("LEGACY lineage.parent_run_id must equal retained_run_id.")
        _require_retained_run_prefix(retained_run_id, prefix, mode)
        _require_parent_run_prefix(lineage.parent_run_id, parent_specification_fingerprint, mode)
        rid = retained_run_id
        record_lineage = lineage

    else:  # pragma: no cover - exhaustive enum
        raise IdentityEmitError(f"Unsupported allocation mode {mode!r}.")

    aid = _generate_attempt_id(frozen_clock, entropy, exists, max_retries)

    record = AttemptIdentityRecord(
        specification_fingerprint=fp,
        run_id=rid,
        attempt_id=aid,
        created_at_utc=now,
        lineage=record_lineage,
    )
    try:
        path = write_identity_sidecar(artifact_root, record)
    except IdentitySidecarError as e:
        raise IdentityEmitError(str(e)) from e
    return record, path


def _verify_parent_spec(
    fp: SpecificationFingerprintRecord,
    parent_specification_fingerprint: str,
    mode: AllocationMode,
) -> None:
    """Verify the computed spec fingerprint matches the parent run's, for modes
    that retain the same logical run (RESUME, LEGACY)."""
    try:
        # Enforce the exact fingerprint-ID format at this boundary.
        validate_fingerprint_id(parent_specification_fingerprint)
        if fp.digest_str != parent_specification_fingerprint:
            raise IdentityEmitError(
                f"{mode.value.capitalize()} requires the computed specification fingerprint "
                f"to match the parent run's. Computed {fp.digest_str!r} != parent "
                f"{parent_specification_fingerprint!r}. A materially changed specification "
                "must use FORK mode."
            )
    except (FingerprintMismatch, ValueError) as e:
        raise IdentityEmitError(str(e)) from e


def _require_retained_run_prefix(
    retained_run_id: str, current_prefix: str, mode: AllocationMode
) -> None:
    """For RESUME/LEGACY: the retained run ID's embedded spec prefix must match
    the current (== parent, since the spec is unchanged) fingerprint prefix."""
    try:
        run_prefix = run_id_spec_prefix(retained_run_id)
    except ValueError as e:  # invalid run ID format
        raise IdentityEmitError(str(e)) from e
    if run_prefix != current_prefix:
        raise IdentityEmitError(
            f"{mode.value.capitalize()} retained_run_id's embedded spec prefix "
            f"{run_prefix!r} does not match the current specification fingerprint "
            f"prefix {current_prefix!r}."
        )


def _require_parent_run_prefix(
    parent_run_id: str, parent_specification_fingerprint: str, mode: AllocationMode
) -> None:
    """For RESUME/LEGACY/FORK: the parent run ID's embedded spec prefix must
    match the parent specification fingerprint's prefix (the parent run must
    belong to the parent specification)."""
    try:
        run_prefix = run_id_spec_prefix(parent_run_id)
        fp_prefix = fingerprint_id_prefix(parent_specification_fingerprint)
    except ValueError as e:  # invalid run-ID or fingerprint-ID format
        raise IdentityEmitError(str(e)) from e
    if run_prefix != fp_prefix:
        raise IdentityEmitError(
            f"{mode.value.capitalize()} lineage.parent_run_id's embedded spec prefix "
            f"{run_prefix!r} does not match the parent specification fingerprint "
            f"prefix {fp_prefix!r}."
        )


def _generate_run_id(
    prefix: str,
    clock: ClockProvider,
    entropy: EntropyProvider | None,
    exists: ExistsPredicate | None,
    max_retries: int,
) -> str:
    try:
        return run_id(prefix, clock=clock, entropy=entropy, exists=exists, max_retries=max_retries)
    except IdentityCollisionError as e:
        raise IdentityEmitError(str(e)) from e


def _generate_attempt_id(
    clock: ClockProvider,
    entropy: EntropyProvider | None,
    exists: ExistsPredicate | None,
    max_retries: int,
) -> str:
    try:
        return attempt_id(clock=clock, entropy=entropy, exists=exists, max_retries=max_retries)
    except IdentityCollisionError as e:
        raise IdentityEmitError(str(e)) from e
