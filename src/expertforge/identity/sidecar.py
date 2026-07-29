"""Identity sidecar write/read (Issue #6 decision §5).

Each per-attempt identity record is written as deterministic JSON to::

    <artifact-root>/<run-id>/attempts/<attempt-id>/run-identity.json

Writes are exclusive (no overwrite): the file is created with ``O_CREAT |
O_EXCL`` so a second write to the same path fails atomically rather than
silently clobbering prior provenance. The complete sidecar is rendered and
UTF-8-validated before any byte is written, so a serialization failure never
leaves a partial file.

Loads are typed: the file is parsed, the identity-schema version is checked, the
record is validated, and the recorded fingerprint is verified against the
provided expected value (when supplied). Unknown versions and validation
failures raise :class:`IdentitySidecarError`.
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
    validation failure, path traversal, I/O)."""


def _validate_id_component(value: str, kind: str) -> None:
    """Reject IDs that could traverse or escape the artifact-root directory."""
    if not value:
        raise IdentitySidecarError(f"{kind} must be non-empty.")
    # Reject path separators and any parent-directory reference. Legitimate IDs
    # never contain "..", so the substring check is safe defense against
    # segment-like traversal even without a separator.
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


def write_identity_sidecar(artifact_root: Path, record: AttemptIdentityRecord) -> Path:
    """Write ``record``'s sidecar deterministically and exclusively.

    Renders the complete JSON, UTF-8-validates it, creates parent directories,
    then opens with ``O_CREAT | O_EXCL`` so an existing sidecar is never
    overwritten. Returns the written path.
    """
    target = sidecar_path(artifact_root, record.run_id, record.attempt_id)
    # Render and UTF-8-validate fully BEFORE touching the filesystem, so a
    # serialization failure cannot leave a partial file.
    try:
        rendered = record.to_deterministic_json()
        rendered.decode("utf-8")  # validate UTF-8 round-trip
    except (UnicodeError, ValueError) as e:
        raise IdentitySidecarError(f"Could not serialize identity record: {e}") from e

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation: fails atomically if the file already exists.
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as e:
        raise IdentitySidecarError(
            f"Identity sidecar already exists at {target}; refusing to overwrite."
        ) from e
    except OSError as e:
        raise IdentitySidecarError(f"Could not create identity sidecar {target}: {e}") from e
    try:
        os.write(fd, rendered)
    finally:
        os.close(fd)
    return target


def load_identity_sidecar(
    path: Path,
    *,
    expected_fingerprint: str | None = None,
) -> AttemptIdentityRecord:
    """Load and validate an identity sidecar.

    Raises :class:`IdentitySidecarError` for missing files, malformed JSON,
    unknown identity-schema versions, validation failures, or fingerprint
    mismatch (when ``expected_fingerprint`` is supplied).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise IdentitySidecarError(f"Identity sidecar not found: {path}") from e
    except OSError as e:
        raise IdentitySidecarError(f"Could not read identity sidecar {path}: {e}") from e

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise IdentitySidecarError(f"Identity sidecar {path} is not valid JSON: {e}") from e

    try:
        record = AttemptIdentityRecord.from_mapping(data)
    except ValueError as e:
        # Unknown schema version (raised by from_mapping) or pydantic
        # ValidationError (a ValueError subclass).
        raise IdentitySidecarError(f"Identity sidecar {path} failed validation: {e}") from e

    if (
        expected_fingerprint is not None
        and record.specification_fingerprint != expected_fingerprint
    ):
        raise IdentitySidecarError(
            f"Identity sidecar {path} fingerprint {record.specification_fingerprint!r} "
            f"does not match expected {expected_fingerprint!r}."
        )
    return record
