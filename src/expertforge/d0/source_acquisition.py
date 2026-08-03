"""Atomic acquisition and fail-closed cache reuse for D0 source objects."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

from expertforge.d0.errors import SourceAcquisitionError, SourceVerificationError
from expertforge.d0.source_manifest import SourceFileIdentity, SourceManifest
from expertforge.d0.source_verification import (
    VerifiedSourceFile,
    resolve_inventory_path,
    verify_source_file,
)

_DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


class DownloadResponse(Protocol):
    """Minimal streaming response contract used by acquisition and local fixtures."""

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> DownloadResponse: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


DownloadOpener = Callable[[str], DownloadResponse]


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Identity and provenance of one cache acquisition decision."""

    verified: VerifiedSourceFile
    source_url: str
    downloaded: bool


def default_download_opener(url: str) -> DownloadResponse:
    """Open one URL for streaming without exposing urllib types to callers."""

    return cast(DownloadResponse, urllib.request.urlopen(url, timeout=120))


def huggingface_resolve_url(manifest: SourceManifest, identity: SourceFileIdentity) -> str:
    """Construct the immutable Hugging Face resolve URL for one manifest member."""

    repository = urllib.parse.quote(manifest.repository, safe="/")
    revision = urllib.parse.quote(manifest.revision, safe="")
    path = urllib.parse.quote(identity.path, safe="/")
    namespace = "datasets/" if manifest.kind == "dataset" else ""
    return f"https://huggingface.co/{namespace}{repository}/resolve/{revision}/{path}?download=true"


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceAcquisitionError(f"cannot open cache directory for fsync: {path}: {exc}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise SourceAcquisitionError(f"cannot fsync cache directory {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def acquire_source_file(
    *,
    url: str,
    cache_root: Path,
    identity: SourceFileIdentity,
    opener: DownloadOpener = default_download_opener,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> AcquisitionResult:
    """Reuse a verified cache entry or atomically publish a verified download.

    An existing but mismatched cache entry is never silently replaced. Operators
    must preserve or remove it explicitly so corruption cannot be mistaken for a
    successful refresh.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    destination = resolve_inventory_path(cache_root, identity.path)
    if destination.exists() or destination.is_symlink():
        try:
            verified = verify_source_file(destination, identity, chunk_size=chunk_size)
        except SourceVerificationError as exc:
            raise SourceAcquisitionError(
                f"existing cache entry failed verification and was preserved: {destination}"
            ) from exc
        return AcquisitionResult(verified=verified, source_url=url, downloaded=False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    size_bytes = 0
    published = False
    try:
        try:
            with os.fdopen(temporary_descriptor, "wb", closefd=True) as output:
                temporary_descriptor = -1
                with opener(url) as response:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                        size_bytes += len(chunk)
                output.flush()
                os.fsync(output.fileno())
        except (OSError, urllib.error.URLError) as exc:
            raise SourceAcquisitionError(f"source download failed for {url}: {exc}") from exc

        actual_digest = digest.hexdigest()
        if size_bytes != identity.size_bytes:
            raise SourceAcquisitionError(
                f"downloaded size mismatch for {identity.path}: expected {identity.size_bytes}, "
                f"actual {size_bytes}"
            )
        if actual_digest != identity.sha256:
            raise SourceAcquisitionError(
                f"downloaded SHA-256 mismatch for {identity.path}: expected {identity.sha256}, "
                f"actual {actual_digest}"
            )
        try:
            os.replace(temporary_path, destination)
            published = True
            _fsync_directory(destination.parent)
        except OSError as exc:
            raise SourceAcquisitionError(
                f"cannot atomically publish verified source {destination}: {exc}"
            ) from exc
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if not published:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    verified = VerifiedSourceFile(
        manifest_path=identity.path,
        local_path=destination.resolve(),
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )
    return AcquisitionResult(verified=verified, source_url=url, downloaded=True)


def acquire_source_inventory(
    cache_root: Path,
    manifest: SourceManifest,
    *,
    opener: DownloadOpener = default_download_opener,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> tuple[AcquisitionResult, ...]:
    """Acquire every frozen manifest member in lexicographic source order."""

    results: list[AcquisitionResult] = []
    for identity in manifest.files:
        results.append(
            acquire_source_file(
                url=huggingface_resolve_url(manifest, identity),
                cache_root=cache_root,
                identity=identity,
                opener=opener,
                chunk_size=chunk_size,
            )
        )
    return tuple(results)
