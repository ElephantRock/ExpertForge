"""Append-only registry management (amendments D, E, K).

The registry is the authoritative per-attempt inventory and state-transition
index. Each line is one strict deterministic registry-entry envelope
(:class:`RegistryEntry`) governed by :data:`ARTIFACT_REGISTRY_FORMAT_VERSION`.

Mutation is **lock-serialized** (amendment E): every append acquires a
per-attempt advisory lock (:class:`AttemptLock`) for sequence allocation,
transition validation, append, and fsync. ``O_APPEND`` atomicity is not relied
upon for regular files; a full-write loop runs under the lock.

Loading enforces (amendment D): canonical RFC 3339 UTC timestamps with six
fractional digits, exact deterministic JSON, strict schemas, duplicate-key
rejection, valid UTF-8, standard JSON numbers only, bounded line size,
contiguous sequence numbers beginning at zero, constant run/attempt/fingerprint
binding, legal per-artifact revision/state transitions, and explicit
``complete``/``incomplete``/``corrupt`` scan outcomes.

Truncated-tail recovery retains the accepted prefix. Orphan detection finds
canonical bundles without registry entries; reconciliation appends a missing
publication entry only after strictly validating the complete bundle and
identity binding (amendment K).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from expertforge.artifacts.locks import AttemptLock, LockUnavailableError
from expertforge.artifacts.models import (
    ARTIFACT_REGISTRY_FORMAT_VERSION,
    ArtifactConflictError,
    ArtifactRecord,
    ExternalReference,
    RegistryEntry,
    RegistryStatus,
    canonical_timestamp,
    is_legal_retention_transition,
    is_legal_storage_transition,
    validate_canonical_timestamp,
)

__all__ = [
    "RegistryError",
    "RegistryScanResult",
    "allocate_and_append",
    "canonical_timestamp",
    "compute_next_sequence",
    "load_registry",
    "registry_path_for_attempt",
    "scan_registry",
]


class RegistryError(Exception):
    """Raised on registry load/scan/append failures (malformed, bad version,
    identity drift, illegal transition, oversized line)."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def registry_path_for_attempt(artifact_root: Path, run_id: str, attempt_id: str) -> Path:
    """Canonical ``<attempt>/registry.jsonl`` path."""
    from expertforge.artifacts.paths import validate_id_component

    validate_id_component(run_id, "run_id")
    validate_id_component(attempt_id, "attempt_id")
    return artifact_root / run_id / "attempts" / attempt_id / "registry.jsonl"


# ---------------------------------------------------------------------------
# Canonical JSON parsing (duplicate-key + non-finite rejection)
# ---------------------------------------------------------------------------


class _ParseFailure(Exception):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _ParseFailure(f"duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _ParseFailure(f"non-standard JSON numeric constant {value!r}.")


def _parse_envelope(line: bytes) -> dict[str, Any]:
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _ParseFailure("envelope is not valid UTF-8.") from e
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except _ParseFailure:
        raise
    except json.JSONDecodeError as e:
        raise _ParseFailure("envelope is not valid JSON.") from e
    if not isinstance(parsed, dict):
        raise _ParseFailure("envelope must be a JSON object.")
    return parsed


# ---------------------------------------------------------------------------
# Truncated-tail split
# ---------------------------------------------------------------------------


def _split_lines(raw: bytes) -> tuple[list[bytes], bool]:
    """Split newline-terminated lines; flag a truncated (non-newline-terminated) tail."""
    if not raw:
        return [], False
    if raw.endswith(b"\n"):
        body = raw[:-1]
        return ([] if not body else body.split(b"\n")), False
    boundary = raw.rfind(b"\n")
    if boundary < 0:
        return [], True
    body = raw[:boundary]
    return ([] if not body else body.split(b"\n")), True


# ---------------------------------------------------------------------------
# Envelope validation
# ---------------------------------------------------------------------------


def _validate_envelope(
    parsed: dict[str, Any],
    *,
    raw_line: bytes,
    expected_sequence: int,
    expected_run_id: str,
    expected_attempt_id: str,
    expected_fingerprint: str,
) -> RegistryEntry:
    from expertforge.artifacts.models import MAX_REGISTRY_ENTRY_BYTES

    required = {
        "registry_format_version",
        "sequence",
        "run_id",
        "attempt_id",
        "specification_fingerprint",
        "recorded_at_utc",
        "entry_kind",
        "payload",
    }
    missing = required - parsed.keys()
    if missing:
        raise _ParseFailure(f"envelope is missing required keys: {sorted(missing)}.")
    extra = parsed.keys() - required
    if extra:
        raise _ParseFailure(f"envelope has unknown keys: {sorted(extra)}.")

    if parsed["registry_format_version"] != ARTIFACT_REGISTRY_FORMAT_VERSION:
        raise _ParseFailure(
            f"registry_format_version {parsed['registry_format_version']!r} is not supported "
            f"(expected {ARTIFACT_REGISTRY_FORMAT_VERSION})."
        )

    seq = parsed["sequence"]
    if not isinstance(seq, int) or isinstance(seq, bool) or seq != expected_sequence:
        raise _ParseFailure(
            f"sequence {seq!r} is not the expected contiguous value {expected_sequence}."
        )

    if parsed["run_id"] != expected_run_id:
        raise _ParseFailure("run_id does not match expected identity binding.")
    if parsed["attempt_id"] != expected_attempt_id:
        raise _ParseFailure("attempt_id does not match expected identity binding.")
    if parsed["specification_fingerprint"] != expected_fingerprint:
        raise _ParseFailure("specification_fingerprint does not match expected identity binding.")

    recorded = parsed["recorded_at_utc"]
    if not isinstance(recorded, str):
        raise _ParseFailure("recorded_at_utc must be a canonical timestamp string.")
    try:
        validate_canonical_timestamp(recorded)
    except ValueError as e:
        raise _ParseFailure(str(e)) from e

    payload = parsed["payload"]
    if not isinstance(payload, dict):
        raise _ParseFailure("payload must be a JSON object.")

    # Re-validate the canonical serialization is exactly deterministic: render
    # the parsed envelope back and confirm it round-trips byte-for-byte. This
    # catches key ordering drift, whitespace injection, and number reformatting.
    entry_kind = parsed["entry_kind"]
    if entry_kind not in (
        "initial_publication",
        "external_registration",
        "retention_transition",
        "verification_transition",
    ):
        raise _ParseFailure(f"unknown entry_kind {entry_kind!r}.")

    recorded_dt = _parse_canonical_dt(recorded)
    entry = RegistryEntry(
        registry_format_version=parsed["registry_format_version"],
        sequence=seq,
        run_id=parsed["run_id"],
        attempt_id=parsed["attempt_id"],
        specification_fingerprint=parsed["specification_fingerprint"],
        recorded_at_utc=recorded_dt,
        entry_kind=entry_kind,
        payload=payload,
    )
    rendered = entry.to_deterministic_json()
    if len(rendered) > MAX_REGISTRY_ENTRY_BYTES:
        raise _ParseFailure(
            f"envelope size {len(rendered)} exceeds {MAX_REGISTRY_ENTRY_BYTES} bytes."
        )
    # Deterministic round-trip: the stored line (minus newline) must equal the
    # canonical rendering. Accept either exact match or a normalized match
    # where only whitespace differs is NOT allowed (whitespace is part of the
    # canonical form). This catches injected formatting/whitespace.
    if rendered != raw_line:
        raise _ParseFailure("envelope is not canonical deterministic JSON.")
    return entry


def _parse_canonical_dt(value: str) -> datetime:
    # Fixed-width canonical UTC: YYYY-MM-DDTHH:MM:SS.ffffffZ
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=_UTC)


from datetime import UTC as _UTC  # noqa: E402

# ---------------------------------------------------------------------------
# Transition validation across an artifact's history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArtifactHistory:
    artifact_id: str
    record: ArtifactRecord
    external: ExternalReference | None = None


def _reduce_history(entries: list[RegistryEntry]) -> dict[str, _ArtifactHistory]:
    """Reduce entries to the last-known state per artifact_id, validating transitions."""
    histories: dict[str, _ArtifactHistory] = {}
    for entry in entries:
        payload = entry.payload
        kind = entry.entry_kind
        if kind == "initial_publication":
            rec = _payload_record(payload)
            aid = rec.artifact_id
            if aid in histories:
                raise RegistryError(f"duplicate initial_publication for artifact {aid!r}.")
            histories[aid] = _ArtifactHistory(artifact_id=aid, record=rec)
        elif kind == "external_registration":
            rec = _payload_record(payload)
            ext = _payload_external(payload)
            aid = rec.artifact_id
            if ext.artifact_id != aid:
                raise RegistryError(
                    f"external_registration payload artifact_id mismatch "
                    f"({ext.artifact_id!r} vs {aid!r})."
                )
            if aid in histories:
                # External registration may augment an existing artifact only
                # if it is a fresh external reference record (not a duplicate
                # initial_publication).
                existing = histories[aid]
                histories[aid] = _ArtifactHistory(
                    artifact_id=aid, record=existing.record, external=ext
                )
            else:
                histories[aid] = _ArtifactHistory(artifact_id=aid, record=rec, external=ext)
        elif kind in ("retention_transition", "verification_transition"):
            raw_aid: Any = payload.get("artifact_id")
            if not isinstance(raw_aid, str):
                raise RegistryError(f"{kind} payload missing artifact_id.")
            aid = raw_aid
            if aid not in histories:
                raise RegistryError(f"{kind} for unknown artifact {aid!r} (no prior publication).")
            existing = histories[aid]
            new_storage = payload.get("storage_class", existing.record.storage_class)
            new_retention = payload.get("retention", existing.record.retention)
            if not is_legal_storage_transition(existing.record.storage_class, new_storage):
                raise RegistryError(
                    f"illegal storage transition {existing.record.storage_class!r} "
                    f"-> {new_storage!r} for artifact {aid!r}."
                )
            if not is_legal_retention_transition(existing.record.retention, new_retention):
                raise RegistryError(
                    f"illegal retention transition {existing.record.retention!r} "
                    f"-> {new_retention!r} for artifact {aid!r}."
                )
            new_rec = existing.record.model_copy(
                update={"storage_class": new_storage, "retention": new_retention}
            )
            histories[aid] = _ArtifactHistory(
                artifact_id=aid, record=new_rec, external=existing.external
            )
    return histories


def _payload_record(payload: Mapping[str, Any]) -> ArtifactRecord:
    # An external_registration payload carries both the record fields and a
    # nested ``external_reference`` object under a stable key. Strip that key
    # before validating the immutable record portion.
    record_payload = {k: v for k, v in payload.items() if k != "external_reference"}
    try:
        return ArtifactRecord.model_validate(record_payload, strict=False)
    except Exception as e:
        raise RegistryError(f"payload is not a valid ArtifactRecord: {e}") from e


def _payload_external(payload: Mapping[str, Any]) -> ExternalReference:
    ext = payload.get("external_reference")
    if not isinstance(ext, dict):
        raise RegistryError("external_registration payload missing external_reference.")
    try:
        return ExternalReference.model_validate(ext, strict=False)
    except Exception as e:
        raise RegistryError(f"payload external_reference is invalid: {e}") from e


# ---------------------------------------------------------------------------
# Scan result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryScanResult:
    """The accepted prefix and ``complete``/``incomplete``/``corrupt`` status."""

    status: RegistryStatus
    accepted: list[RegistryEntry] = field(default_factory=list)
    reason: str | None = None

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)


# ---------------------------------------------------------------------------
# load_registry
# ---------------------------------------------------------------------------


def load_registry(
    path: Path,
    *,
    expected_run_id: str,
    expected_attempt_id: str,
    expected_fingerprint: str,
) -> list[RegistryEntry]:
    """Load and strictly validate the registry; return the accepted entries.

    Raises :class:`RegistryError` on: missing file, invalid UTF-8, malformed or
    non-object JSON, unknown registry version, identity drift, non-contiguous
    sequences, illegal transitions, oversized lines, or a truncated tail
    (incomplete). Use :func:`scan_registry` for diagnostic outcomes.
    """
    result = scan_registry(path)
    if result.status == "incomplete":
        raise RegistryError(f"registry {path} has a truncated tail: {result.reason}")
    if result.status == "corrupt":
        raise RegistryError(f"registry {path} is corrupt: {result.reason}")
    # Re-bind identity: scan_registry already enforces binding, but assert again
    # defensively for the typed contract.
    for entry in result.accepted:
        if entry.run_id != expected_run_id:
            raise RegistryError("registry run_id does not match expected identity.")
        if entry.attempt_id != expected_attempt_id:
            raise RegistryError("registry attempt_id does not match expected identity.")
        if entry.specification_fingerprint != expected_fingerprint:
            raise RegistryError(
                "registry specification_fingerprint does not match expected identity."
            )
    # Validate transitions across the whole history.
    _reduce_history(result.accepted)
    return result.accepted


# ---------------------------------------------------------------------------
# scan_registry
# ---------------------------------------------------------------------------


def scan_registry(path: Path) -> RegistryScanResult:
    """Return the accepted prefix and a ``complete``/``incomplete``/``corrupt`` status.

    The first corrupt or out-of-order line stops acceptance. A truncated
    (non-newline-terminated) tail yields ``incomplete``; otherwise the accepted
    prefix is returned with its status.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        # An absent registry is a complete empty registry.
        return RegistryScanResult(status="complete", accepted=[])
    except OSError as e:
        return RegistryScanResult(status="corrupt", reason=f"read error: {e}")

    lines, truncated = _split_lines(raw)

    # The identity binding is established by the first line; subsequent lines
    # must match it. If no lines exist but bytes were present (e.g. a single
    # newline), the registry is empty and complete.
    if not lines:
        if truncated:
            return RegistryScanResult(status="incomplete", reason="empty non-terminated line")
        return RegistryScanResult(status="complete", accepted=[])

    run_id: str | None = None
    attempt_id: str | None = None
    fingerprint: str | None = None
    accepted: list[RegistryEntry] = []

    for idx, line in enumerate(lines):
        try:
            parsed = _parse_envelope(line)
        except _ParseFailure as e:
            return RegistryScanResult(
                status="corrupt",
                accepted=accepted,
                reason=f"line {idx + 1}: {e}",
            )
        if run_id is None:
            run_id = parsed["run_id"]
            attempt_id = parsed["attempt_id"]
            fingerprint = parsed["specification_fingerprint"]
        # After the first line these are bound; assert to narrow for mypy.
        assert run_id is not None
        assert attempt_id is not None
        assert fingerprint is not None
        try:
            entry = _validate_envelope(
                parsed,
                raw_line=line,
                expected_sequence=idx,
                expected_run_id=run_id,
                expected_attempt_id=attempt_id,
                expected_fingerprint=fingerprint,
            )
        except _ParseFailure as e:
            return RegistryScanResult(
                status="corrupt",
                accepted=accepted,
                reason=f"line {idx + 1}: {e}",
            )
        accepted.append(entry)

    if truncated:
        return RegistryScanResult(
            status="incomplete",
            accepted=accepted,
            reason="truncated trailing line",
        )
    return RegistryScanResult(status="complete", accepted=accepted)


# ---------------------------------------------------------------------------
# Sequence allocation + append (lock-serialized)
# ---------------------------------------------------------------------------


def compute_next_sequence(path: Path) -> int:
    """The next contiguous sequence number for ``path`` (length of the accepted prefix)."""
    return scan_registry(path).accepted_count


def allocate_and_append(
    registry_path: Path,
    *,
    lock_path: Path,
    run_id: str,
    attempt_id: str,
    specification_fingerprint: str,
    entry_kind: str,
    payload: dict[str, Any],
    recorded_at_utc: datetime,
    transition_validator: Callable[[dict[str, _ArtifactHistory]], None] | None = None,
) -> RegistryEntry:
    """Atomically allocate the next sequence and append one envelope under the lock.

    The full sequence allocation, transition validation, append, and fsync
    happen while holding :class:`AttemptLock`. The append uses a full-write loop
    even though each entry is bounded (amendment E). ``O_APPEND`` is opened but
    not relied upon for cross-process atomicity.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    entry: RegistryEntry | None = None
    try:
        with AttemptLock(lock_path):
            next_seq = scan_registry(registry_path).accepted_count
            entry = RegistryEntry(
                registry_format_version=ARTIFACT_REGISTRY_FORMAT_VERSION,
                sequence=next_seq,
                run_id=run_id,
                attempt_id=attempt_id,
                specification_fingerprint=specification_fingerprint,
                recorded_at_utc=recorded_at_utc,
                entry_kind=entry_kind,  # type: ignore[arg-type]
                payload=payload,
            )
            # Transition validation across the whole history including this new
            # entry, under the lock so concurrent appends cannot interleave.
            existing = _reduce_history(scan_registry(registry_path).accepted)
            trial = [*scan_registry(registry_path).accepted, entry]
            try:
                _reduce_history(trial)
            except RegistryError as e:
                raise ArtifactConflictError(str(e)) from e
            if transition_validator is not None:
                transition_validator(existing)
            line = entry.to_deterministic_json() + b"\n"
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
            fd = os.open(registry_path, flags, 0o600)
            try:
                _full_write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
    except LockUnavailableError as e:
        raise RegistryError(f"could not acquire registry lock: {e}") from e
    assert entry is not None
    return entry


def _full_write(fd: int, data: bytes) -> None:
    view = memoryview(data)
    total = 0
    while total < len(view):
        written = os.write(fd, view[total:])
        if written <= 0:  # pragma: no cover - defensive
            raise OSError("os.write returned non-positive byte count")
        total += written


# canonical_timestamp is already imported from models; re-export for callers.
__all__.append("canonical_timestamp")
