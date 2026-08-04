"""Streaming, descriptor-bound verification of frozen D0 source objects."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from expertforge.d0.errors import SourceVerificationError
from expertforge.d0.source_manifest import SourceFileIdentity, SourceManifest

_DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class VerifiedSourceFile:
    """Observed identity of one successfully verified regular file."""

    manifest_path: str
    local_path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedSourceBytes:
    """A verified regular file plus its collected payload bytes.

    The payload is read through the same descriptor used for inspection and
    hashing (no path-based reopen after verification), so callers that need the
    bytes at runtime — e.g. a tokenizer loaded via ``from_buffer`` — inherit the
    descriptor-bound guarantee instead of re-reading the path.
    """

    identity: VerifiedSourceFile
    payload: bytes


def resolve_inventory_path(root: Path, manifest_path: str) -> Path:
    """Resolve a manifest path beneath ``root`` without permitting traversal."""

    resolved_root = root.resolve()
    candidate = (resolved_root / manifest_path).resolve(strict=False)
    if not candidate.is_relative_to(resolved_root):
        raise SourceVerificationError(f"manifest path escapes cache root: {manifest_path}")
    return candidate


def _verify_source_payload(
    path: Path,
    identity: SourceFileIdentity,
    *,
    chunk_size: int,
    collect_payload: bool,
) -> tuple[VerifiedSourceFile, bytes | None]:
    """Verify one source object using the same descriptor for inspection and hashing.

    When ``collect_payload`` is True the full byte payload is accumulated and
    returned alongside the verified identity, read through the protected
    descriptor. Otherwise the payload slot of the returned tuple is None and the
    file is only streamed through the hash.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    try:
        if path.is_symlink():
            raise SourceVerificationError(f"source object must not be a symlink: {path}")
    except OSError as exc:
        raise SourceVerificationError(f"cannot inspect source path {path}: {exc}") from exc

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceVerificationError(f"cannot open source object {path}: {exc}") from exc

    digest = hashlib.sha256()
    size_bytes = 0
    payload_chunks: list[bytes] | None = [] if collect_payload else None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SourceVerificationError(f"source object must be a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
                size_bytes += len(chunk)
                if payload_chunks is not None:
                    payload_chunks.append(chunk)
    except OSError as exc:
        raise SourceVerificationError(f"cannot read source object {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    actual_digest = digest.hexdigest()
    if size_bytes != identity.size_bytes:
        raise SourceVerificationError(
            f"source size mismatch for {identity.path}: expected {identity.size_bytes}, "
            f"actual {size_bytes}"
        )
    if actual_digest != identity.sha256:
        raise SourceVerificationError(
            f"source SHA-256 mismatch for {identity.path}: expected {identity.sha256}, "
            f"actual {actual_digest}"
        )
    verified = VerifiedSourceFile(
        manifest_path=identity.path,
        local_path=path.resolve(),
        size_bytes=size_bytes,
        sha256=actual_digest,
    )
    payload = b"".join(payload_chunks) if payload_chunks is not None else None
    return verified, payload


def verify_source_file(
    path: Path,
    identity: SourceFileIdentity,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> VerifiedSourceFile:
    """Verify one source object using the same descriptor for inspection and hashing."""

    verified, _payload = _verify_source_payload(
        path, identity, chunk_size=chunk_size, collect_payload=False
    )
    return verified


def verify_source_file_bytes(
    path: Path,
    identity: SourceFileIdentity,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> VerifiedSourceBytes:
    """Verify one source object and return its collected bytes.

    The bytes are read through the same protected descriptor used for the size
    and SHA-256 checks, so a caller that consumes the payload (e.g. a tokenizer
    constructed via ``from_buffer``) cannot be fooled by a path change between
    verification and use.
    """

    verified, payload = _verify_source_payload(
        path, identity, chunk_size=chunk_size, collect_payload=True
    )
    if payload is None:  # pragma: no cover - collect_payload=True always yields bytes
        raise SourceVerificationError(f"internal error: payload not collected for {identity.path}")
    return VerifiedSourceBytes(identity=verified, payload=payload)


def verify_source_inventory(
    root: Path,
    manifest: SourceManifest,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> tuple[VerifiedSourceFile, ...]:
    """Verify every manifest member in frozen lexicographic source order."""

    verified: list[VerifiedSourceFile] = []
    for identity in manifest.files:
        path = resolve_inventory_path(root, identity.path)
        verified.append(verify_source_file(path, identity, chunk_size=chunk_size))
    observed_total = sum(item.size_bytes for item in verified)
    if observed_total != manifest.total_size_bytes:
        raise SourceVerificationError(
            f"verified inventory byte total mismatch: expected {manifest.total_size_bytes}, "
            f"actual {observed_total}"
        )
    return tuple(verified)
