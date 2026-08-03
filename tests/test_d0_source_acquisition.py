from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from expertforge.d0.errors import SourceAcquisitionError, SourceVerificationError
from expertforge.d0.source_acquisition import DownloadResponse, acquire_source_file
from expertforge.d0.source_manifest import SourceFileIdentity
from expertforge.d0.source_verification import verify_source_file


class _BytesResponse(io.BytesIO):
    pass


def _identity(path: str, payload: bytes) -> SourceFileIdentity:
    return SourceFileIdentity(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def test_verify_source_file_binds_size_and_digest(tmp_path: Path) -> None:
    payload = b"frozen source bytes"
    path = tmp_path / "source.bin"
    path.write_bytes(payload)
    verified = verify_source_file(path, _identity("source.bin", payload), chunk_size=3)
    assert verified.size_bytes == len(payload)
    assert verified.sha256 == hashlib.sha256(payload).hexdigest()


def test_verify_source_file_rejects_digest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(b"changed")
    with pytest.raises(SourceVerificationError, match="SHA-256 mismatch"):
        verify_source_file(path, _identity("source.bin", b"expected"))


def test_acquisition_publishes_verified_bytes_atomically(tmp_path: Path) -> None:
    payload = b"download fixture"
    calls: list[str] = []

    def opener(url: str) -> DownloadResponse:
        calls.append(url)
        return _BytesResponse(payload)

    identity = _identity("nested/source.bin", payload)
    result = acquire_source_file(
        url="https://example.invalid/source.bin",
        cache_root=tmp_path,
        identity=identity,
        opener=opener,
        chunk_size=4,
    )
    assert result.downloaded is True
    assert result.verified.local_path.read_bytes() == payload
    assert calls == ["https://example.invalid/source.bin"]
    assert not list((tmp_path / "nested").glob("*.part"))


def test_acquisition_reverifies_cache_without_network(tmp_path: Path) -> None:
    payload = b"cached fixture"
    destination = tmp_path / "source.bin"
    destination.write_bytes(payload)

    def opener(url: str) -> DownloadResponse:
        raise AssertionError(f"network opener should not be called: {url}")

    result = acquire_source_file(
        url="https://example.invalid/source.bin",
        cache_root=tmp_path,
        identity=_identity("source.bin", payload),
        opener=opener,
    )
    assert result.downloaded is False


def test_acquisition_preserves_corrupt_cache_entry(tmp_path: Path) -> None:
    destination = tmp_path / "source.bin"
    destination.write_bytes(b"corrupt")

    def opener(url: str) -> DownloadResponse:
        raise AssertionError(f"network opener should not be called: {url}")

    with pytest.raises(SourceAcquisitionError, match="preserved"):
        acquire_source_file(
            url="https://example.invalid/source.bin",
            cache_root=tmp_path,
            identity=_identity("source.bin", b"expected"),
            opener=opener,
        )
    assert destination.read_bytes() == b"corrupt"
