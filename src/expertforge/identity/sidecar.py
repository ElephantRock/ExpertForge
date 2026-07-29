"""Identity sidecar write/read (Issue #6 decision §5; review item 6).

Each per-attempt identity record is written as deterministic JSON to::

    <artifact-root>/<run-id>/attempts/<attempt-id>/run-identity.json

Writes are exclusive (no overwrite): the file is created with ``O_CREAT |
O_EXCL`` so a second write to the same path fails atomically rather than
silently clobbering prior provenance. The complete sidecar is rendered and
UTF-8-validated before any byte is written. The full byte payload is written in
a loop (``os.write`` may return short), ``fsync``'d, and the file removed on any
write/fsync/close failure so a retry is not permanently blocked by ``O_EXCL``
and success is never reported with truncated provenance.

Loads are typed: the file is read as bytes (invalid UTF-8 → boundary error),
parsed as JSON (non-object JSON rejected), the identity-schema version is
checked, the record is validated, and the embedded fingerprint's stored digest
is verified against its envelope. Unknown versions and validation failures raise
:class:`IdentitySidecarError`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from expertforge.identity.record import AttemptIdentityRecord

__all__ = [
    "IdentitySidecarError",
    "load_identity_sidecar",
    "sidecar_path",
    "write_identity_sidecar",
]

_SIDECAR_FILENAME = "run-identity.json"


class IdentitySidecarError(Exception):
    """Raised on sidecar write/load failures (overwrite, malformed, bad version,
    validation failure, path traversal, I/O, short write, bad UTF-8)."""


def _validate_id_component(value: str, kind: str) -> None:
    """Reject IDs that could traverse or escape the artifact-root directory."""
    if not value:
        raise IdentitySidecarError(f"{kind} must be non-empty.")
    bad_chars = {os.sep, "/", "\\"}
    if any(ch in value for ch in bad_chars) or ".." in value or value in {".", ".."}:
        raise IdentitySidecarError(
            f"{kind} {value!r} contains a path separator or traversal component."
        )


def sidecar_path(artifact_root: Path, run_id: str, attempt_id: str) -> Path:
    """Return the canonical sidecar path for a run/attempt pair."""
    _validate_id_component(run_id, "run_id")
    _validate_id_component(attempt_id, "attempt_id")
    return artifact_root / run_id / "attempts" / attempt_id / _SIDECAR_FILENAME


def _write_all_exclusive(target: Path, payload: bytes) -> None:
    """Create ``target`` exclusively and write the full payload, fsync, close.

    On any failure after creation, the partial file is removed so a retry is not
    permanently blocked by ``O_EXCL`` and success is never reported with a
    truncated file.
    """
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        # Full write loop: os.write may return fewer bytes than requested.
        view = memoryview(payload)
        total = 0
        while total < len(view):
            written = os.write(fd, view[total:])
            if written <= 0:  # pragma: no cover - defensive
                raise OSError("os.write returned non-positive byte count")
            total += written
        os.fsync(fd)
    except Exception:
        # Remove the partial file so O_EXCL does not permanently block retries.
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise
    else:
        os.close(fd)


def write_identity_sidecar(artifact_root: Path, record: AttemptIdentityRecord) -> Path:
    """Write ``record``'s sidecar deterministically and exclusively.

    Renders the complete JSON, UTF-8-validates it, creates parent directories,
    then creates the file with ``O_CREAT | O_EXCL`` and writes the full payload
    with fsync. An existing sidecar is never overwritten.
    """
    target = sidecar_path(artifact_root, record.run_id, record.attempt_id)
    try:
        rendered = record.to_deterministic_json()
        rendered.decode("utf-8")  # validate UTF-8 round-trip before touching fs
    except (UnicodeError, ValueError) as e:
        raise IdentitySidecarError(f"Could not serialize identity record: {e}") from e

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_all_exclusive(target, rendered)
    except FileExistsError as e:
        raise IdentitySidecarError(
            f"Identity sidecar already exists at {target}; refusing to overwrite."
        ) from e
    except OSError as e:
        raise IdentitySidecarError(f"Could not write identity sidecar {target}: {e}") from e
    return target


def load_identity_sidecar(
    path: Path,
    *,
    expected_fingerprint: str | None = None,
) -> AttemptIdentityRecord:
    """Load and validate an identity sidecar.

    Raises :class:`IdentitySidecarError` for missing files, invalid UTF-8,
    malformed or non-object JSON, unknown identity-schema versions, validation
    failures, or fingerprint mismatch (when ``expected_fingerprint`` is supplied).
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError as e:
        raise IdentitySidecarError(f"Identity sidecar not found: {path}") from e
    except OSError as e:
        raise IdentitySidecarError(f"Could not read identity sidecar {path}: {e}") from e

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise IdentitySidecarError(f"Identity sidecar {path} is not valid UTF-8: {e}") from e

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise IdentitySidecarError(f"Identity sidecar {path} is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise IdentitySidecarError(
            f"Identity sidecar {path} must be a JSON object; got {type(data).__name__}."
        )

    try:
        record = AttemptIdentityRecord.from_mapping(data)
    except ValueError as e:
        raise IdentitySidecarError(f"Identity sidecar {path} failed validation: {e}") from e

    if expected_fingerprint is not None and record.fingerprint_digest_str() != expected_fingerprint:
        raise IdentitySidecarError(
            f"Identity sidecar {path} fingerprint {record.fingerprint_digest_str()!r} "
            f"does not match expected {expected_fingerprint!r}."
        )
    return record
