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

from pydantic import ValidationError

from expertforge.artifacts.locks import AttemptLock, LockUnavailableError
from expertforge.artifacts.models import (
    ARTIFACT_REGISTRY_FORMAT_VERSION,
    ArtifactConflictError,
    ArtifactDescriptor,
    ArtifactRecord,
    ExternalReference,
    RegistryEntry,
    RegistryStatus,
    RetentionStatus,
    StorageClass,
    _thaw,
    canonical_timestamp,
    descriptor_to_artifact_id,
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


def _payload_record(payload: Mapping[str, Any], *, strip_external: bool = False) -> ArtifactRecord:
    """Strictly validate a payload (or sub-payload) as an :class:`ArtifactRecord`.

    Validation is strict (``model_validate_json`` over re-canonicalized bytes) so
    payloads are not coerced: types must match exactly. JSON-native structural
    normalization (lists to tuples, datetimes) is performed by Pydantic's strict
    mode only when reading from JSON. ``strict=False`` is never used.

    ``strip_external`` removes the nested ``external_reference`` key that an
    ``external_registration`` payload carries alongside the record fields; that
    key is not part of the immutable record schema and must be validated
    separately via :func:`_payload_external`.
    """
    record_payload: Mapping[str, Any] = payload
    if strip_external and "external_reference" in payload:
        record_payload = {k: v for k, v in payload.items() if k != "external_reference"}
    try:
        return ArtifactRecord.model_validate_json(
            _canonical_json_bytes(record_payload), strict=True
        )
    except (ValidationError, ValueError, TypeError) as e:
        raise RegistryError(f"payload is not a valid ArtifactRecord: {e}") from e


def _payload_external(payload: Mapping[str, Any]) -> ExternalReference:
    """Strictly validate a payload's nested ``external_reference`` object."""
    ext = payload.get("external_reference")
    if not isinstance(ext, Mapping):
        raise RegistryError("external_registration payload missing external_reference.")
    try:
        return ExternalReference.model_validate_json(_canonical_json_bytes(ext), strict=True)
    except (ValidationError, ValueError, TypeError) as e:
        raise RegistryError(f"payload external_reference is invalid: {e}") from e


def _canonical_json_bytes(obj: Any) -> bytes:
    """Compact, sorted-key, UTF-8, non-finite-prohibiting JSON of a payload."""
    return json.dumps(
        _thaw(obj) if isinstance(obj, Mapping) else obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity_matches_envelope(record_or_ext: Any, entry: RegistryEntry) -> None:
    """Validate the payload's identity binding matches the registry envelope."""
    if record_or_ext.run_id != entry.run_id:
        raise RegistryError(
            f"payload run_id {record_or_ext.run_id!r} does not match envelope {entry.run_id!r}."
        )
    if record_or_ext.attempt_id != entry.attempt_id:
        raise RegistryError(
            f"payload attempt_id {record_or_ext.attempt_id!r} does not match "
            f"envelope {entry.attempt_id!r}."
        )
    if record_or_ext.specification_fingerprint != entry.specification_fingerprint:
        raise RegistryError("payload specification_fingerprint does not match envelope binding.")


def _record_internally_consistent(rec: ArtifactRecord) -> None:
    """Validate that an ArtifactRecord's immutable descriptor-derived fields are
    internally consistent (recompute ``artifact_id`` and ``relative_path``)."""
    descriptor = ArtifactDescriptor(
        schema_version=rec.schema_version,
        run_id=rec.run_id,
        attempt_id=rec.attempt_id,
        specification_fingerprint=rec.specification_fingerprint,
        category=rec.category,
        format=rec.format,
        format_version=rec.format_version,
        content_digest=rec.content_digest,
        byte_size=rec.byte_size,
        producing_component=rec.producing_component,
        parent=rec.parent,
    )
    expected_id = descriptor_to_artifact_id(descriptor)
    if rec.artifact_id != expected_id:
        raise RegistryError(
            f"payload artifact_id {rec.artifact_id!r} does not match the "
            f"descriptor-derived id {expected_id!r}."
        )
    expected_rel = f"artifacts/{rec.category}/{rec.artifact_id}"
    if rec.relative_path != expected_rel:
        raise RegistryError(
            f"payload relative_path {rec.relative_path!r} does not match the "
            f"canonical path {expected_rel!r}."
        )


def _reduce_history(entries: list[RegistryEntry]) -> dict[str, _ArtifactHistory]:
    """Reduce entries to the last-known state per artifact_id, validating transitions.

    Every payload's identity binding (``run_id``/``attempt_id``/
    ``specification_fingerprint``) must match the registry envelope that carries
    it; every ArtifactRecord's immutable descriptor fields must be internally
    consistent (amendment D strict binding).
    """
    histories: dict[str, _ArtifactHistory] = {}
    for entry in entries:
        payload = entry.payload
        kind = entry.entry_kind
        if kind == "initial_publication":
            rec = _payload_record(payload)
            _identity_matches_envelope(rec, entry)
            _record_internally_consistent(rec)
            aid = rec.artifact_id
            if aid in histories:
                raise RegistryError(f"duplicate initial_publication for artifact {aid!r}.")
            histories[aid] = _ArtifactHistory(artifact_id=aid, record=rec)
        elif kind == "external_registration":
            rec = _payload_record(payload, strip_external=True)
            _identity_matches_envelope(rec, entry)
            _record_internally_consistent(rec)
            ext = _payload_external(payload)
            _identity_matches_envelope(ext, entry)
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
            raw_storage = payload.get("storage_class", existing.record.storage_class)
            raw_retention = payload.get("retention", existing.record.retention)
            if not isinstance(raw_storage, str) or not isinstance(raw_retention, str):
                raise RegistryError(f"{kind} payload has non-string storage/retention.")
            new_storage: StorageClass = raw_storage  # type: ignore[assignment]
            new_retention: RetentionStatus = raw_retention  # type: ignore[assignment]
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
        else:
            raise RegistryError(f"unknown entry_kind {kind!r}.")
    return histories


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
    sequences, payload schema or identity-binding failures, illegal transitions,
    oversized lines, or a truncated tail (incomplete). The scan already performs
    every validation that ``load`` requires; this entry point simply maps the
    scan status to a raised error. Use :func:`scan_registry` for diagnostic
    outcomes.
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
    return result.accepted


# ---------------------------------------------------------------------------
# scan_registry
# ---------------------------------------------------------------------------


def scan_registry(path: Path) -> RegistryScanResult:
    """Return the accepted prefix and a ``complete``/``incomplete``/``corrupt`` status.

    The scanner is authoritative (amendment D): it validates every accepted line
    against the strict shared validator, which covers envelope framing, strict
    payload schemas (:class:`ArtifactRecord` / :class:`ExternalReference`),
    identity binding between payload and envelope, internal descriptor
    consistency, and per-artifact lifecycle transitions. The first line that
    fails any of these stops acceptance; the rest of the file is the
    corrupt/incomplete tail. All parsing is wrapped so a typed ``corrupt`` result
    is always produced (no ``KeyError``/``ValidationError`` leaks).
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
        # One try/except around the whole per-line acceptance so any failure
        # (framing, schema, identity binding, transition) yields a typed corrupt
        # result. The accepted prefix is retained (truncated-tail recovery).
        try:
            parsed = _parse_envelope(line)
            if run_id is None:
                run_id = parsed["run_id"]
                attempt_id = parsed["attempt_id"]
                fingerprint = parsed["specification_fingerprint"]
            # After the first line these are bound; assert to narrow for mypy.
            assert run_id is not None
            assert attempt_id is not None
            assert fingerprint is not None
            entry = _validate_envelope(
                parsed,
                raw_line=line,
                expected_sequence=idx,
                expected_run_id=run_id,
                expected_attempt_id=attempt_id,
                expected_fingerprint=fingerprint,
            )
            accepted.append(entry)
            # Re-run the strict shared validator over the accepted prefix so the
            # scanner is authoritative for payloads and transitions, not just
            # envelope framing.
            _reduce_history(accepted)
        except _ParseFailure as e:
            return RegistryScanResult(
                status="corrupt",
                accepted=accepted,
                reason=f"line {idx + 1}: {e}",
            )
        except RegistryError as e:
            return RegistryScanResult(
                status="corrupt",
                accepted=accepted,
                reason=f"line {idx + 1}: {e}",
            )

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

    Refuses to append when the current registry scan is not ``complete``: an
    incomplete (truncated tail) or corrupt registry is never extended (item #1).
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with AttemptLock(lock_path):
            return _append_under_lock(
                registry_path,
                run_id=run_id,
                attempt_id=attempt_id,
                specification_fingerprint=specification_fingerprint,
                entry_kind=entry_kind,
                payload=payload,
                recorded_at_utc=recorded_at_utc,
                transition_validator=transition_validator,
            )
    except LockUnavailableError as e:
        raise RegistryError(f"could not acquire registry lock: {e}") from e


def _append_under_lock(
    registry_path: Path,
    *,
    run_id: str,
    attempt_id: str,
    specification_fingerprint: str,
    entry_kind: str,
    payload: dict[str, Any],
    recorded_at_utc: datetime,
    transition_validator: Callable[[dict[str, _ArtifactHistory]], None] | None = None,
) -> RegistryEntry:
    """Append one envelope assuming the per-attempt lock is already held.

    This is the lock-free inner body of :func:`allocate_and_append`, exposed so a
    caller that already holds :class:`AttemptLock` (e.g. publish's critical
    section) can append without re-acquiring the non-reentrant lock (item #6).
    Refuses to append to a non-complete registry (item #1).
    """
    scan = scan_registry(registry_path)
    if scan.status != "complete":
        raise RegistryError(
            f"registry {registry_path} is {scan.status} ({scan.reason}); "
            "refusing to append to a non-complete registry."
        )
    accepted = scan.accepted
    next_seq = len(accepted)
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
    # Transition validation across the whole history including this new entry,
    # under the lock so concurrent appends cannot interleave.
    existing = _reduce_history(accepted)
    trial = [*accepted, entry]
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
