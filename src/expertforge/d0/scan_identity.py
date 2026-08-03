"""Immutable identity sidecar for restartable D0 contamination scans."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from expertforge.d0.errors import SourceOrderError
from expertforge.d0.source_manifest import canonical_json_bytes

_HEX = frozenset("0123456789abcdef")


def _digest(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise SourceOrderError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _commit(value: str) -> str:
    if len(value) != 40 or any(character not in _HEX for character in value):
        raise SourceOrderError("scanner_source_commit must be a full lowercase Git SHA")
    return value


@dataclass(frozen=True, slots=True)
class ScanIdentity:
    """Every immutable input that determines one restartable scan state."""

    dataset_manifest_sha256: str
    tokenizer_manifest_sha256: str
    prompt_manifest_sha256: str
    prompt_payload_sha256: str
    scanner_algorithm_version: str
    scanner_source_commit: str

    def as_record(self) -> dict[str, object]:
        """Return the canonical self-digested identity record."""

        if not self.scanner_algorithm_version:
            raise SourceOrderError("scanner_algorithm_version must not be blank")
        record: dict[str, object] = {
            "schema_version": "expertforge-d0-contamination-scan-identity/1",
            "dataset_manifest_sha256": _digest(
                self.dataset_manifest_sha256, "dataset_manifest_sha256"
            ),
            "tokenizer_manifest_sha256": _digest(
                self.tokenizer_manifest_sha256, "tokenizer_manifest_sha256"
            ),
            "prompt_manifest_sha256": _digest(
                self.prompt_manifest_sha256, "prompt_manifest_sha256"
            ),
            "prompt_payload_sha256": _digest(
                self.prompt_payload_sha256, "prompt_payload_sha256"
            ),
            "scanner_algorithm_version": self.scanner_algorithm_version,
            "scanner_source_commit": _commit(self.scanner_source_commit),
            "digest_policy": "sha256(canonical_json_without_scan_identity_sha256)",
        }
        record["scan_identity_sha256"] = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
        return record


def _validate_record(value: Mapping[str, object]) -> bytes:
    declared = value.get("scan_identity_sha256")
    if not isinstance(declared, str):
        raise SourceOrderError("scan identity digest is missing")
    if value.get("digest_policy") != "sha256(canonical_json_without_scan_identity_sha256)":
        raise SourceOrderError("scan identity digest policy changed")
    payload = dict(value)
    payload.pop("scan_identity_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if declared != actual:
        raise SourceOrderError("scan identity digest mismatch")
    return canonical_json_bytes(dict(value))


def bind_scan_identity(path: Path, identity: ScanIdentity) -> str:
    """Create an immutable identity sidecar or verify exact reuse."""

    expected_record = identity.as_record()
    expected_bytes = _validate_record(expected_record)
    expected_digest = str(expected_record["scan_identity_sha256"])
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise SourceOrderError("scan identity sidecar must not be a symlink")
        try:
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8", errors="strict"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceOrderError(f"cannot read existing scan identity: {exc}") from exc
        if not isinstance(value, Mapping):
            raise SourceOrderError("existing scan identity must be a JSON object")
        actual_bytes = _validate_record(value)
        if actual_bytes != expected_bytes:
            raise SourceOrderError("restart state is bound to a different scan identity")
        return expected_digest

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".part",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(expected_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        published = True
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise SourceOrderError(f"cannot publish scan identity sidecar: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return expected_digest
