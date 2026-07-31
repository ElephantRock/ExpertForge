"""Issue #10 artifact store tests (amendments A, C, G, H, I, J).

Covers: publish (atomic bundle directories), register_existing (copies into
store ownership, no symlink following), register_external (identity-bound,
credential rejection), locate / list_artifacts / inspect / verify /
update_retention, path safety (traversal/absolute/symlink rejection), conflicts,
idempotency, full semantic artifact IDs, source-path mutation independence, and
the typed VerificationResult.

These are fast filesystem-backed tests (tmp_path); the portable integration
tier covers crash recovery, concurrency, and git check-ignore.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from expertforge.artifacts.models import (
    ArtifactConflictError,
    ArtifactNotFoundError,
    ArtifactRecord,
    ParentReference,
)
from expertforge.artifacts.store import ArtifactStore, sha256_stream
from tests._artifact_fixtures import make_store

_PAYLOAD = b'{"result":"hello"}'
_PAYLOAD_DIGEST = "sha256:" + hashlib.sha256(_PAYLOAD).hexdigest()


def _publish_report(store: ArtifactStore, content: bytes = _PAYLOAD) -> str:
    rec = store.publish(
        content,
        category="report",
        format="json",
        format_version=1,
        producing_component="training",
    )
    return rec.artifact_id


class TestPublishBundle:
    def test_publish_creates_bundle_directory(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        aid = _publish_report(store)
        bundle = store.attempt_dir / "artifacts" / "report" / aid
        assert bundle.is_dir()
        assert (bundle / "content").read_bytes() == _PAYLOAD
        meta = json.loads((bundle / "artifact.json").read_bytes())
        assert meta["schema"] == "expertforge.artifact-record"
        assert meta["artifact_id"] == aid
        assert meta["storage_class"] == "canonical_local"
        assert meta["retention"] == "retained"

    def test_publish_returns_full_semantic_id(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        aid = _publish_report(store)
        assert aid.startswith("artifact-v1-sha256-")
        assert len(aid) == len("artifact-v1-sha256-") + 64

    def test_publish_writes_registry_entry(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        aid = _publish_report(store)
        entries = store.list_artifacts()
        assert len(entries) == 1
        assert entries[0].artifact_id == aid

    def test_publish_idempotent_same_content(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        aid1 = _publish_report(store)
        aid2 = _publish_report(store)
        assert aid1 == aid2
        # Registry still has exactly one entry.
        assert len(store.list_artifacts()) == 1

    def test_publish_distinct_content_distinct_ids(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        a = _publish_report(store, b'{"a":1}')
        b = _publish_report(store, b'{"b":2}')
        assert a != b
        assert len(store.list_artifacts()) == 2

    def test_no_temp_bundle_left_behind(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        _publish_report(store)
        cdir = store.attempt_dir / "artifacts" / "report"
        leftovers = [p for p in cdir.iterdir() if p.name.startswith(".tmp-bundle-")]
        assert leftovers == []


class TestPublishPathSource:
    def test_publish_from_path_copies_bytes(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        src = tmp_path / "src.json"
        src.write_bytes(_PAYLOAD)
        aid = store.publish(
            src,
            category="report",
            format="json",
            format_version=1,
            producing_component="training",
        ).artifact_id
        content = store.locate(aid)
        assert content is not None and content.read_bytes() == _PAYLOAD


class TestRegisterExisting:
    def test_register_existing_copies_into_store(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        src = tmp_path / "src.bin"
        src.write_bytes(b"\x00\x01\x02" * 10)
        rec = store.register_existing(
            src,
            category="checkpoint",
            format="binary",
            format_version=1,
            producing_component="ckpt",
        )
        # Canonical content lives inside the bundle, not at the source path.
        content = store.locate(rec.artifact_id)
        assert content is not None
        assert content.read_bytes() == b"\x00\x01\x02" * 10
        # The canonical path is not the source path.
        assert content != src

    def test_source_path_mutation_does_not_change_canonical(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        src = tmp_path / "src.bin"
        src.write_bytes(b"original")
        rec = store.register_existing(
            src,
            category="checkpoint",
            format="binary",
            format_version=1,
            producing_component="ckpt",
        )
        # Mutate the source after registration.
        src.write_bytes(b"MUTATED-CANONICAL-MUST-NOT-CHANGE")
        content = store.locate(rec.artifact_id)
        assert content is not None
        assert content.read_bytes() == b"original"

    def test_register_existing_rejects_symlink(self, tmp_path: Path) -> None:
        if os.name == "nt":
            pytest.skip("symlink semantics differ on Windows without privileges")
        store = make_store(tmp_path)
        target = tmp_path / "target.bin"
        target.write_bytes(b"real")
        link = tmp_path / "link.bin"
        os.symlink(target, link)
        with pytest.raises(ArtifactConflictError):
            store.register_existing(
                link,
                category="checkpoint",
                format="binary",
                format_version=1,
                producing_component="ckpt",
            )

    def test_register_existing_rejects_directory(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        d = tmp_path / "adir"
        d.mkdir()
        with pytest.raises(ArtifactConflictError):
            store.register_existing(
                d,
                category="checkpoint",
                format="binary",
                format_version=1,
                producing_component="ckpt",
            )


class TestRegisterExternal:
    def test_register_external_identity_bound(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        ext = store.register_external(
            category="report",
            format="json",
            format_version=1,
            producing_component="eval",
            location_type="uri",
            location="https://example.com/report.json",
            expected_digest=_PAYLOAD_DIGEST,
            expected_byte_size=len(_PAYLOAD),
        )
        assert ext.artifact_id.startswith("artifact-v1-sha256-")
        assert ext.run_id == store.identity.run_id
        assert ext.availability == "unavailable"
        # inspect returns the external reference.
        got = store.inspect(ext.artifact_id)
        assert getattr(got, "location", None) == "https://example.com/report.json"

    def test_register_external_rejects_credentials(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ArtifactConflictError):
            store.register_external(
                category="report",
                format="json",
                format_version=1,
                producing_component="eval",
                location_type="uri",
                location="https://user:pw@example.com/r.json",
                expected_digest=_PAYLOAD_DIGEST,
                expected_byte_size=len(_PAYLOAD),
            )

    def test_register_external_rejects_query(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ArtifactConflictError):
            store.register_external(
                category="report",
                format="json",
                format_version=1,
                producing_component="eval",
                location_type="uri",
                location="https://example.com/r.json?token=x",
                expected_digest=_PAYLOAD_DIGEST,
                expected_byte_size=len(_PAYLOAD),
            )

    def test_register_external_filesystem_requires_root_id(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ArtifactConflictError):
            store.register_external(
                category="checkpoint",
                format="binary",
                format_version=1,
                producing_component="ckpt",
                location_type="filesystem_path",
                location="data/x.bin",
                expected_digest=_PAYLOAD_DIGEST,
                expected_byte_size=len(_PAYLOAD),
            )

    def test_register_external_filesystem_absolute_rejected(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ArtifactConflictError):
            store.register_external(
                category="checkpoint",
                format="binary",
                format_version=1,
                producing_component="ckpt",
                location_type="filesystem_path",
                location="/etc/x.bin",
                expected_digest=_PAYLOAD_DIGEST,
                expected_byte_size=len(_PAYLOAD),
                external_root_id="root-1",
            )


class TestLocateListInspect:
    def test_locate_returns_none_for_unknown(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        assert store.locate("artifact-v1-sha256-" + "0" * 64) is None

    def test_inspect_unknown_raises(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ArtifactNotFoundError):
            store.inspect("artifact-v1-sha256-" + "0" * 64)

    def test_list_filtered_by_category(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.publish(
            b'{"r":1}',
            category="report",
            format="json",
            format_version=1,
            producing_component="t",
        )
        store.publish(
            b"line\n",
            category="telemetry",
            format="jsonl",
            format_version=1,
            producing_component="t",
        )
        reports = store.list_artifacts(category="report")
        assert len(reports) == 1
        assert reports[0].category == "report"


class TestVerify:
    def test_verify_good_artifact(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        aid = _publish_report(store)
        result = store.verify(aid)
        assert result.status is True
        assert result.diagnostic_code == "verified"
        assert result.observed_digest == _PAYLOAD_DIGEST
        assert result.observed_size == len(_PAYLOAD)
        assert store.verify_bool(aid) is True

    def test_verify_missing_bundle(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        result = store.verify("artifact-v1-sha256-" + "0" * 64)
        assert result.status is False
        assert result.diagnostic_code == "missing_bundle"

    def test_verify_detects_tampered_content(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        aid = _publish_report(store)
        content = store.locate(aid)
        assert content is not None
        # Tamper with the canonical content.
        content.write_bytes(b"TAMPERED")
        result = store.verify(aid)
        assert result.status is False
        assert result.diagnostic_code == "digest_mismatch"


class TestUpdateRetention:
    def test_legal_retention_transition(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        aid = _publish_report(store)
        rec = store.update_retention(aid, "pending_transfer")
        assert rec.retention == "pending_transfer"
        # last-known state reflects the transition.
        inspected = store.inspect(aid)
        assert isinstance(inspected, ArtifactRecord)
        assert inspected.retention == "pending_transfer"

    def test_illegal_retention_transition_rejected(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        aid = _publish_report(store)
        # retained -> externally_retained is illegal.
        with pytest.raises(ArtifactConflictError):
            store.update_retention(aid, "externally_retained")

    def test_transition_to_missing_moves_storage_class(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        aid = _publish_report(store)
        rec = store.update_retention(aid, "missing")
        assert rec.retention == "missing"
        assert rec.storage_class == "metadata_only"

    def test_update_unknown_artifact_rejected(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        with pytest.raises(ArtifactNotFoundError):
            store.update_retention("artifact-v1-sha256-" + "0" * 64, "expired")


class TestPathSafetyAndFormat:
    def test_invalid_category_rejected(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        from expertforge.artifacts.store import ArtifactStoreError

        with pytest.raises((ArtifactStoreError, Exception)):
            store.publish(
                b"x",
                category="not_a_category",
                format="json",
                format_version=1,
                producing_component="t",
            )

    def test_format_not_allowed_for_category(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        from expertforge.artifacts.store import ArtifactStoreError

        with pytest.raises((ArtifactStoreError, Exception)):
            store.publish(
                b"x",
                category="telemetry",
                format="json",
                format_version=1,
                producing_component="t",
            )

    def test_category_with_separator_rejected(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        from expertforge.artifacts.paths import PathSafetyError

        with pytest.raises(PathSafetyError):
            store.publish(
                b"x",
                category="../x",
                format="json",
                format_version=1,
                producing_component="t",
            )


class TestCrossAttemptParent:
    def test_cross_attempt_parent_reference(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        parent = ParentReference(
            run_id="run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb",
            attempt_id="attempt-20260101t000000z-dddddddddddddddddddd",
            artifact_id="artifact-v1-sha256-" + "c" * 64,
        )
        rec = store.publish(
            b'{"c":3}',
            category="checkpoint",
            format="binary",
            format_version=1,
            producing_component="ckpt",
            parent=parent,
        )
        assert rec.parent is not None
        assert rec.parent.run_id == parent.run_id
        assert rec.parent.attempt_id == parent.attempt_id


class TestHashing:
    def test_sha256_stream(self) -> None:
        chunks = [b"abc", b"def", b"ghi"]
        idx = [0]

        def reader(n: int) -> bytes:
            if idx[0] >= len(chunks):
                return b""
            c = chunks[idx[0]]
            idx[0] += 1
            return c

        digest, size = sha256_stream(reader)
        assert digest == "sha256:" + hashlib.sha256(b"abcdefghi").hexdigest()
        assert size == 9

    def test_hashing_uses_64kib_blocks(self) -> None:
        from expertforge.artifacts.store import BLOCK_SIZE

        assert BLOCK_SIZE == 64 * 1024


class TestBundleIndependentlyVerifiable:
    def test_bundle_self_contained_metadata(self, tmp_path: Path) -> None:
        # Amendment K: the immutable artifact.json inside a canonical bundle is
        # self-contained metadata needed to verify that bundle.
        store = make_store(tmp_path)
        aid = _publish_report(store)
        bundle = store.attempt_dir / "artifacts" / "report" / aid
        meta = json.loads((bundle / "artifact.json").read_bytes())
        # All fields needed to verify the payload are present.
        for key in (
            "artifact_id",
            "content_digest",
            "byte_size",
            "category",
            "format",
            "run_id",
            "attempt_id",
            "specification_fingerprint",
        ):
            assert key in meta


class TestIdempotencyAfterRegistryGap:
    def test_publish_then_index_recovery(self, tmp_path: Path) -> None:
        # If a bundle exists but the registry entry is missing (crash between
        # rename and append), re-publishing must not duplicate and must index.
        store = make_store(tmp_path)
        aid = _publish_report(store)
        # Wipe the registry to simulate a crash after rename before append.
        store.registry_path.unlink()
        # Re-publish identical content: the bundle already exists; idempotent.
        aid2 = _publish_report(store)
        assert aid == aid2
        # Registry now indexes it (reconciliation-on-publish).
        assert any(r.artifact_id == aid for r in store.list_artifacts())
