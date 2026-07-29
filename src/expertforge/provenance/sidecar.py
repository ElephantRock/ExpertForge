"""Provenance sidecar (Issue #7 decision: artifact ownership).

Owns the narrow, durable provenance sidecar::

    <artifact-root>/<run-id>/attempts/<attempt-id>/run-provenance.json

Deterministic serialization, typed load and verification, exclusive publication,
full-write handling, fsync, and failure cleanup — mirroring the #6 identity
sidecar discipline. Issue #10 later registers, hashes, retains, and externally
locates this sidecar; Issue #12 references it without duplicating capture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.record import ProvenanceRecord

__all__ = [
    "ProvenanceSidecarError",
    "load_provenance_sidecar",
    "provenance_sidecar_path",
    "write_provenance_sidecar",
]

_SIDECAR_FILENAME = "run-provenance.json"


class ProvenanceSidecarError(Exception):
    """Raised on provenance sidecar write/load failures."""


def _validate_id_component(value: str, kind: str) -> None:
    if not value:
        raise ProvenanceSidecarError(f"{kind} must be non-empty.")
    bad_chars = {os.sep, "/", "\\"}
    if any(ch in value for ch in bad_chars) or ".." in value or value in {".", ".."}:
        raise ProvenanceSidecarError(
            f"{kind} {value!r} contains a path separator or traversal component."
        )


def provenance_sidecar_path(artifact_root: Path, run_id: str, attempt_id: str) -> Path:
    """Return the canonical provenance sidecar path for a run/attempt pair.

    Reuses the exact Issue #6 run/attempt ID validators (not just generic
    traversal checks).
    """
    from expertforge.identity.ids import validate_attempt_id, validate_run_id

    try:
        validate_run_id(run_id)
        validate_attempt_id(attempt_id)
    except ValueError as e:
        raise ProvenanceSidecarError(str(e)) from e
    return artifact_root / run_id / "attempts" / attempt_id / _SIDECAR_FILENAME


def _write_all_exclusive(target: Path, payload: bytes) -> None:
    """Create ``target`` exclusively, write the full payload, fsync, close.

    Full-write loop (``os.write`` may return short); on any failure after
    creation, the partial file is removed so a retry is not permanently blocked
    by ``O_EXCL``.
    """
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        view = memoryview(payload)
        total = 0
        while total < len(view):
            written = os.write(fd, view[total:])
            if written <= 0:  # pragma: no cover - defensive
                raise OSError("os.write returned non-positive byte count")
            total += written
        os.fsync(fd)
        os.close(fd)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


def write_provenance_sidecar(artifact_root: Path, record: ProvenanceRecord) -> Path:
    """Write ``record``'s provenance sidecar deterministically and exclusively."""
    target = provenance_sidecar_path(artifact_root, record.run_id, record.attempt_id)
    try:
        rendered = record.to_deterministic_json()
        rendered.decode("utf-8")  # validate UTF-8 round-trip before touching fs
    except (UnicodeError, ValueError) as e:
        raise ProvenanceSidecarError(f"Could not serialize provenance record: {e}") from e

    try:
        # mkdir inside the typed write-error boundary so a directory-creation
        # failure surfaces as ProvenanceSidecarError, not a raw OSError.
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_all_exclusive(target, rendered)
    except FileExistsError as e:
        raise ProvenanceSidecarError(
            f"Provenance sidecar already exists at {target}; refusing to overwrite."
        ) from e
    except OSError as e:
        raise ProvenanceSidecarError(f"Could not write provenance sidecar {target}: {e}") from e
    return target


def load_provenance_sidecar(
    path: Path,
    *,
    expected_identity: AttemptIdentityRecord | None = None,
    expected_source_digest: str | None = None,
) -> ProvenanceRecord:
    """Load and validate a provenance sidecar.

    When ``expected_identity`` is supplied, verifies the loaded record's
    run_id, attempt_id, specification fingerprint, immutable inputs, and
    timestamp match the identity. When ``expected_source_digest`` is supplied,
    verifies the ``source.snapshot`` input digest matches it.

    Raises :class:`ProvenanceSidecarError` on any mismatch.

    Raises :class:`ProvenanceSidecarError` for missing files, invalid UTF-8,
    malformed or non-object JSON, unknown schema versions, or validation
    failures.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError as e:
        raise ProvenanceSidecarError(f"Provenance sidecar not found: {path}") from e
    except OSError as e:
        raise ProvenanceSidecarError(f"Could not read provenance sidecar {path}: {e}") from e

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ProvenanceSidecarError(f"Provenance sidecar {path} is not valid UTF-8: {e}") from e

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProvenanceSidecarError(f"Provenance sidecar {path} is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ProvenanceSidecarError(
            f"Provenance sidecar {path} must be a JSON object; got {type(data).__name__}."
        )

    try:
        record = ProvenanceRecord.from_mapping(data)
    except ValueError as e:
        raise ProvenanceSidecarError(f"Provenance sidecar {path} failed validation: {e}") from e

    # Verify against the expected identity when supplied.
    if expected_identity is not None:
        ident = expected_identity
        if record.run_id != ident.run_id:
            raise ProvenanceSidecarError(
                f"run_id mismatch: record {record.run_id!r} vs identity {ident.run_id!r}."
            )
        if record.attempt_id != ident.attempt_id:
            raise ProvenanceSidecarError(
                f"attempt_id mismatch: record {record.attempt_id!r} vs identity {ident.attempt_id!r}."
            )
        if record.start_time_utc != ident.created_at_utc:
            raise ProvenanceSidecarError(
                f"timestamp mismatch: record {record.start_time_utc!r} vs identity {ident.created_at_utc!r}."
            )
        if record.specification_fingerprint != ident.specification_fingerprint:
            raise ProvenanceSidecarError("specification_fingerprint mismatch.")
        if record.immutable_inputs != ident.specification_fingerprint.immutable_inputs:
            raise ProvenanceSidecarError("immutable_inputs mismatch.")

    # Verify the source.snapshot digest when supplied.
    if expected_source_digest is not None:
        snap_input = next(
            (ii for ii in record.immutable_inputs if ii.name == "source.snapshot"),
            None,
        )
        if snap_input is None or snap_input.digest != expected_source_digest:
            raise ProvenanceSidecarError(
                f"source.snapshot digest mismatch: expected {expected_source_digest!r}."
            )

    return record
