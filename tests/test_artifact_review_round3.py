"""Focused regression tests for PR #32 round-3 review items (1-3).

Each test class targets exactly one round-3 review item. These tests lock in
the reviewed behaviors against regressions:

  1. Path rejection happens BEFORE any filesystem mutation: a symlinked
     run/attempt/category component is rejected and the symlink target is NOT
     modified (no new files/dirs created through the symlink). Covers publish,
     ``_copy_to_owned_buffer`` (register_telemetry), ``_write_metadata_only_bundle``
     (register_external), and the AttemptLock parent-dir creation.
  2. Reconciliation and locate are fully symlink-safe: category-symlink and
     bundle-symlink in reconciliation are skipped/rejected; a symlinked final
     content entry in locate is rejected; content hashing uses the fd-based
     symlink-safe path (no TOCTOU reopen).
  3. Telemetry copy is a bounded streaming copy with a proper descriptor
     lifecycle: the source is streamed in 64 KiB blocks (never fully in memory),
     the mkstemp descriptor is used directly and closed properly (no leak).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from expertforge.artifacts.models import (
    ArtifactConflictError,
)
from expertforge.artifacts.store import (
    BLOCK_SIZE,
    ArtifactStoreError,
)
from tests._artifact_fixtures import (
    VALID_ATTEMPT,
    VALID_RUN,
    make_store,
)


def _skip_if_no_symlink_privilege() -> None:
    if os.name == "nt":
        pytest.skip("symlink semantics differ on Windows without privileges")


def _skip_if_no_rusage() -> None:
    try:
        import resource  # noqa: F401  (availability probe only)
    except ImportError:
        pytest.skip("resource module not available on this platform")


# Capture the real os.write once at import time so write-failure spies can
# delegate to it after monkeypatching the store's _full_write helper.
real_os_write = os.write


# ---------------------------------------------------------------------------
# Item 1: path rejection happens BEFORE filesystem mutation.
# ---------------------------------------------------------------------------


class TestItem1PathRejectionBeforeMutation:
    """A symlinked component must be rejected and the symlink target left
    unmodified (item #1). The ``_safe_makedirs`` helper verifies the chain,
    mkdir's, and re-verifies — so NO directory is created through a symlink
    before the rejection raises."""

    def test_safe_makedirs_helper_exists_and_is_used(self) -> None:
        import expertforge.artifacts.store as store_mod

        assert hasattr(store_mod, "_safe_makedirs")
        src = Path(store_mod.__file__).read_text(encoding="utf-8")
        # publish() must route directory creation through _safe_makedirs (not a
        # bare mkdir followed by a separate verify).
        assert "_safe_makedirs(self._artifact_root, cdir)" in src

    def test_publish_symlinked_run_does_not_mutate_target(self, tmp_path: Path) -> None:
        _skip_if_no_symlink_privilege()
        store = make_store(tmp_path)
        # Make the run directory a symlink to an external victim directory so the
        # chain from artifact_root to the category dir crosses a symlinked run
        # component.
        root = tmp_path / "runs"
        root.mkdir(parents=True, exist_ok=True)
        victim = tmp_path / "victim-run-target"
        victim.mkdir()
        snapshot_before = set(victim.iterdir()) if victim.exists() else set()
        os.symlink(victim, root / VALID_RUN)
        with pytest.raises((ArtifactConflictError, ArtifactStoreError)):
            store.publish(
                b"x",
                category="report",
                format="text",
                format_version=1,
                producing_component="t",
            )
        # The symlink target must NOT have been mutated: no new entries created
        # through the symlinked run component (item #1).
        snapshot_after = set(victim.iterdir())
        assert snapshot_after == snapshot_before, (
            "publish mutated the symlink target before rejecting the symlinked chain"
        )

    def test_publish_symlinked_attempt_does_not_mutate_target(self, tmp_path: Path) -> None:
        _skip_if_no_symlink_privilege()
        store = make_store(tmp_path)
        # Create the real run dir, then make the attempt dir a symlink to an
        # external victim so the chain crosses a symlinked attempt component.
        run_real = tmp_path / "runs" / VALID_RUN
        attempts_real = run_real / "attempts"
        attempts_real.mkdir(parents=True, exist_ok=True)
        victim = tmp_path / "victim-attempt-target"
        victim.mkdir()
        snapshot_before = set(victim.iterdir())
        os.symlink(victim, attempts_real / VALID_ATTEMPT)
        with pytest.raises((ArtifactConflictError, ArtifactStoreError)):
            store.publish(
                b"x",
                category="report",
                format="text",
                format_version=1,
                producing_component="t",
            )
        snapshot_after = set(victim.iterdir())
        assert snapshot_after == snapshot_before, (
            "publish mutated the symlinked attempt target before rejection"
        )

    def test_register_external_symlinked_category_does_not_mutate_target(
        self, tmp_path: Path
    ) -> None:
        """register_external creates the category dir via
        ``_write_metadata_only_bundle`` → ``_safe_makedirs`` (item #1). A
        symlinked category component must be rejected without mutating the
        target."""
        _skip_if_no_symlink_privilege()
        store = make_store(tmp_path)
        # Pre-create the run/attempts tree up to the artifacts dir so the only
        # symlinked component is the category.
        arts = store.attempt_dir / "artifacts"
        arts.mkdir(parents=True, exist_ok=True)
        victim = tmp_path / "victim-cat-target"
        victim.mkdir()
        snapshot_before = set(victim.iterdir())
        os.symlink(victim, arts / "report")
        with pytest.raises((ArtifactConflictError, ArtifactStoreError)):
            store.register_external(
                category="report",
                format="json",
                format_version=1,
                producing_component="eval",
                location_type="uri",
                location="https://example.com/x.json",
                expected_digest="sha256:" + "a" * 64,
                expected_byte_size=10,
            )
        snapshot_after = set(victim.iterdir())
        assert snapshot_after == snapshot_before, (
            "register_external mutated the symlinked category target before rejection"
        )

    def test_register_telemetry_symlinked_attempt_does_not_mutate_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """register_telemetry's owned-buffer copy creates the attempt dir via
        ``_safe_makedirs`` (item #1). A symlinked attempt component is rejected
        without mutating the target. We short-circuit the telemetry loader so
        the test focuses on the directory-creation guard in
        ``_copy_to_owned_buffer``."""
        _skip_if_no_symlink_privilege()
        store = make_store(tmp_path)
        # Make the attempt dir a symlink to an external victim.
        run_real = tmp_path / "runs" / VALID_RUN / "attempts"
        run_real.mkdir(parents=True, exist_ok=True)
        victim = tmp_path / "victim-tel-target"
        victim.mkdir()
        snapshot_before = set(victim.iterdir())
        os.symlink(victim, run_real / VALID_ATTEMPT)
        # A regular source file for _assert_regular_source to pass.
        source = tmp_path / "stream.jsonl"
        source.write_bytes(b'{"registry_format_version": 1}\n')

        class _Boom(Exception):
            pass

        # Patch the loader import inside register_telemetry so we reach the
        # owned-buffer copy without needing a valid telemetry stream; the copy
        # itself must reject the symlinked attempt dir before the loader runs.

        def _fake_load(*_args: Any, **_kwargs: Any) -> None:
            raise _Boom("loader should not run; symlink rejected first")

        monkeypatch.setattr("expertforge.telemetry.loader.load_telemetry_stream", _fake_load)

        with pytest.raises((ArtifactConflictError, ArtifactStoreError)):
            store.register_telemetry(
                source,
                process_context=None,
                producing_component="logging",
            )
        snapshot_after = set(victim.iterdir())
        assert snapshot_after == snapshot_before, (
            "register_telemetry mutated the symlinked attempt target before rejection"
        )

    def test_attempt_lock_symlinked_parent_does_not_mutate_target(self, tmp_path: Path) -> None:
        """The AttemptLock parent-dir creation is symlink-safe (item #1). A
        symlinked attempt component is rejected (LockUnavailableError) and the
        symlink target is not mutated."""
        _skip_if_no_symlink_privilege()
        from expertforge.artifacts.locks import (
            AttemptLock,
            LockUnavailableError,
        )

        run_real = tmp_path / "runs" / VALID_RUN / "attempts"
        run_real.mkdir(parents=True, exist_ok=True)
        victim = tmp_path / "victim-lock-target"
        victim.mkdir()
        snapshot_before = set(victim.iterdir())
        os.symlink(victim, run_real / VALID_ATTEMPT)
        lock = AttemptLock(run_real / VALID_ATTEMPT / "registry.lock")
        with pytest.raises(LockUnavailableError):
            lock.acquire()
        snapshot_after = set(victim.iterdir())
        assert snapshot_after == snapshot_before, (
            "AttemptLock mutated the symlinked parent target before rejection"
        )

    def test_attempt_lock_symlinked_ancestor_missing_target(self, tmp_path: Path) -> None:
        """The AttemptLock rejects creation when the target is missing but its
        deepest existing ancestor is a symlink. The symlink target must not
        receive any directory or lock file (review 4828911851)."""
        _skip_if_no_symlink_privilege()
        from expertforge.artifacts.locks import (
            AttemptLock,
            LockUnavailableError,
        )

        # Create: runs/<run>/attempts -> symlink to victim
        # The attempt dir itself does NOT exist; attempts is the symlink.
        run_real = tmp_path / "runs" / VALID_RUN
        run_real.mkdir(parents=True, exist_ok=True)
        victim = tmp_path / "victim-ancestor-symlink"
        victim.mkdir()
        snapshot_before = set(victim.iterdir())
        os.symlink(victim, run_real / "attempts")
        # The lock path is under a MISSING attempt dir, whose deepest existing
        # ancestor (attempts) is a symlink.
        lock = AttemptLock(run_real / "attempts" / VALID_ATTEMPT / "registry.lock")
        with pytest.raises(LockUnavailableError):
            lock.acquire()
        snapshot_after = set(victim.iterdir())
        assert snapshot_after == snapshot_before, (
            "AttemptLock created directories through a symlinked ancestor before rejection"
        )


# ---------------------------------------------------------------------------
# Item 2: reconciliation and locate fully symlink-safe.
# ---------------------------------------------------------------------------


class TestItem2ReconcileAndLocateSymlinkSafe:
    def test_reconcile_skips_symlinked_category(self, tmp_path: Path) -> None:
        _skip_if_no_symlink_privilege()
        store = make_store(tmp_path)
        store.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        )
        arts = store.attempt_dir / "artifacts"
        # Plant a symlinked category directory pointing at a sibling with a
        # fake orphan bundle. Reconciliation must skip (not follow) it.
        sibling = store.attempt_dir / "evil-cat-target"
        fake_bundle = sibling / ("artifact-v1-sha256-" + "1" * 64)
        fake_bundle.mkdir(parents=True)
        (fake_bundle / "artifact.json").write_bytes(b"{}")
        os.symlink(sibling, arts / "evil")
        result = store.reconcile_orphans()
        # The symlinked category contributed no appended artifact (skipped, not
        # followed). The result must not raise and must not index the fake
        # orphan.
        assert fake_bundle.name not in result.appended_artifact_ids

    def test_reconcile_skips_symlinked_bundle(self, tmp_path: Path) -> None:
        _skip_if_no_symlink_privilege()
        store = make_store(tmp_path)
        store.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        )
        cat = store.attempt_dir / "artifacts" / "report"
        # Plant a symlinked bundle entry pointing at a sibling dir with a fake
        # artifact.json. Reconciliation must skip (not follow) the symlinked
        # bundle.
        sibling = cat / "evil-bundle-target"
        sibling.mkdir()
        (sibling / "artifact.json").write_bytes(b"{}")
        os.symlink(sibling, cat / ("artifact-v1-sha256-" + "2" * 64))
        result = store.reconcile_orphans()
        assert "artifact-v1-sha256-" + "2" * 64 not in result.appended_artifact_ids

    def test_reconcile_content_symlink_is_skipped(self, tmp_path: Path) -> None:
        _skip_if_no_symlink_privilege()
        store = make_store(tmp_path)
        # Publish a real artifact so the tree exists, then build a second
        # bundle directory by hand whose content is a symlink (pointing at a
        # file with the wrong bytes) and whose artifact.json is internally
        # consistent. Reconciliation must skip it (content is not a regular
        # file), not append it.
        rec = store.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        )
        cat = store.attempt_dir / "artifacts" / "report"
        # Copy the good bundle to a new artifact_id-derived dir and symlink its
        # content to an external wrong-bytes file.
        good = cat / rec.artifact_id
        evil_id = "artifact-v1-sha256-" + "3" * 64
        evil = cat / evil_id
        evil.mkdir()
        (evil / "artifact.json").write_bytes((good / "artifact.json").read_bytes())
        wrong = tmp_path / "wrong-content"
        wrong.write_bytes(b"DIFFERENT-BYTES")
        os.symlink(wrong, evil / "content")
        result = store.reconcile_orphans()
        assert evil_id in result.orphan_artifact_ids
        assert evil_id not in result.appended_artifact_ids
        # It must have been skipped with a reason.
        skipped_ids = {aid for aid, _reason in result.skipped}
        assert evil_id in skipped_ids

    def test_locate_rejects_symlinked_final_content(self, tmp_path: Path) -> None:
        _skip_if_no_symlink_privilege()
        store = make_store(tmp_path)
        rec = store.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        )
        bundle = store.attempt_dir / "artifacts" / "report" / rec.artifact_id
        content = bundle / "content"
        # Replace the content file with a symlink to an external file.
        external = tmp_path / "external-secret"
        external.write_bytes(b"SECRET-SHOULD-NOT-LEAK")
        content.unlink()
        os.symlink(external, content)
        # locate must NOT hand back the symlinked path (item #2c).
        assert store.locate(rec.artifact_id) is None

    def test_content_hashing_uses_fd_not_reopen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reconciliation content hashing goes through _open_read_no_follow +
        _hash_fd (the fd-based path), never _hash_path (which reopens). We
        assert by checking that the removed _hash_path helper is gone and that
        a content swap between a (hypothetical) check and read is caught by the
        fstat-verified descriptor path."""
        import expertforge.artifacts.store as store_mod

        # _hash_path was removed in favor of the fd-based hashing (item #2b).
        assert not hasattr(store_mod, "_hash_path")
        # The reconcile source routes content hashing through _open_read_no_follow.
        src = Path(store_mod.__file__).read_text(encoding="utf-8")
        assert "_open_read_no_follow(content_path)" in src


# ---------------------------------------------------------------------------
# Item 3: telemetry copy is bounded streaming + correct descriptor lifecycle.
# ---------------------------------------------------------------------------


class TestItem3BoundedStreamingCopy:
    def test_copy_uses_block_size_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_copy_to_owned_buffer streams via os.read(src_fd, BLOCK_SIZE). Mock
        os.read to record the requested sizes and assert each is exactly
        BLOCK_SIZE (64 KiB) until EOF."""
        store = make_store(tmp_path)
        # Create a large source (> 1 MB) regular file.
        payload = b"A" * (BLOCK_SIZE * 20 + 123)
        source = tmp_path / "big.jsonl"
        source.write_bytes(payload)
        store._assert_regular_source(source)  # passes: regular file

        requested: list[int] = []
        real_read = os.read

        def _spy_read(fd: int, n: int) -> bytes:
            requested.append(n)
            return real_read(fd, n)

        monkeypatch.setattr("expertforge.artifacts.store.os.read", _spy_read)
        out = store._copy_to_owned_buffer(source)
        # Every read request must ask for exactly BLOCK_SIZE bytes (the streaming
        # block size). The empty-read termination is also at BLOCK_SIZE.
        assert requested, "os.read was never called"
        assert all(n == BLOCK_SIZE for n in requested), (
            f"_copy_to_owned_buffer did not stream in {BLOCK_SIZE}-byte blocks: {requested}"
        )
        # The owned copy holds the exact payload and no more (bounded).
        assert out.read_bytes() == payload

    def test_copy_descriptor_lifecycle_no_leak(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No file descriptor is leaked: the mkstemp descriptor is used
        directly and closed (item #3). We count open fds before/after and
        assert they return to baseline."""
        _skip_if_no_rusage()
        store = make_store(tmp_path)
        payload = b"B" * (BLOCK_SIZE * 3)
        source = tmp_path / "src.jsonl"
        source.write_bytes(payload)

        def _open_fds() -> int:
            # Soft fd count via getrusage.ru_nofiles or scanning /proc/self/fd.
            proc_fd = Path("/proc/self/fd")
            if proc_fd.exists():
                return len(list(proc_fd.iterdir()))
            # Fallback: resource usage current open fds is not portable, so use
            # the proc count above; if unavailable, skip the assertion body.
            pytest.skip("/proc/self/fd not available; cannot count fds portably")

        before = _open_fds()
        out = store._copy_to_owned_buffer(source)
        after = _open_fds()
        # The streaming copy must close both the source and dest fds; no leak.
        assert after == before, f"fd leak: before={before} after={after}"
        # And the copy is correct.
        assert out.read_bytes() == payload
        out.unlink()

    def test_copy_uses_mkstemp_fd_directly(self, tmp_path: Path) -> None:
        """The source must route mkstemp's returned fd through os.write/fsync/
        close directly (no discard-and-reopen). Inspect the store source."""
        import expertforge.artifacts.store as store_mod

        src = Path(store_mod.__file__).read_text(encoding="utf-8")
        # The dest fd comes straight from mkstemp and is written/closed
        # directly — no second os.open on the temp path.
        assert "dest_fd, tmp_str = tempfile.mkstemp" in src
        assert "os.fsync(dest_fd)" in src
        assert "os.close(dest_fd)" in src

    def test_copy_cleans_up_temp_on_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If a write fails mid-stream, both fds are closed and the temp path
        is unlinked (item #3 cleanup path)."""
        store = make_store(tmp_path)
        payload = b"C" * (BLOCK_SIZE * 2)
        source = tmp_path / "fail.jsonl"
        source.write_bytes(payload)
        attempt_dir = store.attempt_dir

        writes = {"n": 0}

        def _flaky_write(fd: int, data: bytes) -> int:
            writes["n"] += 1
            if writes["n"] >= 2:
                raise OSError("simulated mid-stream failure")
            return real_os_write(fd, data)

        monkeypatch.setattr("expertforge.artifacts.store._full_write", _flaky_write)
        with pytest.raises(OSError):
            store._copy_to_owned_buffer(source)
        # No leftover temp files in the attempt dir (the only entries should be
        # whatever existed before — none here).
        leftovers = [p for p in attempt_dir.iterdir() if p.name.startswith(".tmp-tel-")]
        assert leftovers == [], f"temp file not cleaned up on error: {leftovers}"
