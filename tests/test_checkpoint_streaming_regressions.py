from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from expertforge.checkpoints.models import TensorMemberRef
from expertforge.checkpoints.restore import TensorComponentRef, _materialize
from expertforge.checkpoints.store import (
    CheckpointArchive,
    CheckpointCorruptError,
    CheckpointStore,
)
from expertforge.checkpoints.tar_reader import (
    ParsedMember,
    TarParseError,
    parse_ustar_archive_streaming,
)
from expertforge.checkpoints.tar_writer import TarMember, build_ustar_archive


def _redirect_mkstemp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Path]:
    real = tempfile.mkstemp
    created: list[Path] = []

    def redirected(*args: Any, **kwargs: Any) -> tuple[int, str]:
        kwargs["dir"] = tmp_path
        fd, name = real(*args, **kwargs)
        created.append(Path(name))
        return fd, name

    monkeypatch.setattr(tempfile, "mkstemp", redirected)
    return created


def _parse_file(path: Path) -> list[ParsedMember]:
    fd = os.open(path, os.O_RDONLY)
    try:
        return parse_ustar_archive_streaming(fd, path.stat().st_size, validate_member_order=False)
    finally:
        os.close(fd)


def test_stream_parser_cleans_in_progress_spill_on_padding_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _redirect_mkstemp(monkeypatch, tmp_path)
    payload = b"x" * (4 * 1024 * 1024 + 1)
    archive = bytearray(build_ustar_archive([TarMember("tensors/0.bin", payload)]))
    padding_start = 512 + len(payload)
    archive[padding_start] = 1
    path = tmp_path / "bad-padding.tar"
    path.write_bytes(archive)

    with pytest.raises(TarParseError, match="non-zero padding"):
        _parse_file(path)
    assert created
    assert all(not p.exists() for p in created)


def test_stream_parser_cleans_completed_spill_on_late_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _redirect_mkstemp(monkeypatch, tmp_path)
    payload = b"x" * (4 * 1024 * 1024 + 1)
    archive = bytearray(build_ustar_archive([TarMember("tensors/0.bin", payload)]))
    archive[-1] = 1
    path = tmp_path / "bad-terminator.tar"
    path.write_bytes(archive)

    with pytest.raises(TarParseError, match="second terminator"):
        _parse_file(path)
    assert created
    assert all(not p.exists() for p in created)


def test_archive_cleanup_closes_stream_and_retries_failed_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spill = tmp_path / "tensor.bin"
    spill.write_bytes(b"abcd")
    member = ParsedMember("tensors/0.bin", None, 4, spill)
    archive = object.__new__(CheckpointArchive)
    archive._members_by_name = {member.name: member}
    archive._temp_paths = [spill]
    archive._open_streams = []
    archive._mapped_buffers = {}
    archive._mapped_resources = {}

    stream = archive.member_stream(member.name)
    real_unlink = Path.unlink
    calls = 0

    def flaky_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("busy")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    archive.cleanup_temp_files()
    assert stream.closed
    assert archive._temp_paths == [spill]
    archive.cleanup_temp_files()
    assert archive._temp_paths == []
    assert not spill.exists()


def test_materialize_uses_mmap_backed_buffer_for_spill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spill = tmp_path / "tensor.bin"
    spill.write_bytes(b"abcd")
    member = ParsedMember("tensors/0.bin", None, 4, spill)
    archive = object.__new__(CheckpointArchive)
    archive._members_by_name = {member.name: member}
    archive._temp_paths = [spill]
    archive._open_streams = []
    archive._mapped_buffers = {}
    archive._mapped_resources = {}

    def forbidden_read_bytes(self: Path) -> bytes:
        raise AssertionError("Path.read_bytes must not be used")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    manifest_ref = TensorMemberRef(
        member_name=member.name,
        member_sha256=hashlib.sha256(b"abcd").hexdigest(),
        member_byte_size=4,
        logical_names=("weight",),
        dtype="uint8",
        shape=(4,),
    )
    component_ref = TensorComponentRef(
        member_name=member.name,
        logical_names=("weight",),
        dtype="uint8",
        shape=(4,),
    )
    tensor = _materialize(archive, component_ref, {member.name: manifest_ref})
    assert isinstance(tensor.raw_bytes, memoryview)
    assert bytes(tensor.raw_bytes) == b"abcd"
    archive.cleanup_temp_files()
    assert not spill.exists()


def test_load_cleans_spills_when_manifest_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _redirect_mkstemp(monkeypatch, tmp_path)
    tensor = b"x" * (4 * 1024 * 1024 + 1)
    members = [TarMember("manifest.json", b"{}")]
    for role in (
        "identity",
        "configuration",
        "provenance",
        "rng",
        "data_cursor",
        "counters",
        "model",
        "optimizer",
        "scheduler",
    ):
        members.append(TarMember(f"state/{role}.json", b"{}"))
    members.append(TarMember("tensors/0.bin", tensor))
    archive_bytes = build_ustar_archive(members)
    content = tmp_path / "content.tar"
    content.write_bytes(archive_bytes)
    digest = f"sha256:{hashlib.sha256(archive_bytes).hexdigest()}"
    record = SimpleNamespace(
        artifact_id="checkpoint-1", byte_size=len(archive_bytes), content_digest=digest
    )

    class FakeArtifactStore:
        def open_verified_content(self, artifact_id: str) -> tuple[int, Any]:
            return os.open(content, os.O_RDONLY), record

    class TestStore(CheckpointStore):
        def _inspect_record(self, artifact_id: str) -> Any:
            return record

        def _assert_record_loadable(self, record: Any) -> None:
            return None

        def _assert_identity_binding(self, record: Any, expected: Any) -> None:
            return None

        def _assert_resume_lineage(self, record: Any, expected: Any) -> None:
            return None

        def _resolve_resume_parent(self, record: Any, expected: Any) -> Any:
            return record

        def _decode_manifest(self, members: list[ParsedMember]) -> Any:
            raise CheckpointCorruptError("injected manifest failure")

    from expertforge.checkpoints import store as store_module

    monkeypatch.setattr(store_module, "MAX_INMEMORY_ARCHIVE_BYTES", 0)
    store = TestStore(FakeArtifactStore())  # type: ignore[arg-type]
    with pytest.raises(CheckpointCorruptError, match="injected manifest failure"):
        store.load("checkpoint-1", expected_identity=object())  # type: ignore[arg-type]
    assert created
    assert all(not p.exists() for p in created)
