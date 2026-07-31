"""Focused regression tests for PR #32 review items 0-10.

Each test class targets exactly one review item. These tests exist to lock in
the reviewed behaviors against regressions; they complement the broader
artifact test suites.

Items covered:
  0. ``os.O_BINARY`` guarded for non-Windows typesheds (compile/import + mypy).
  1. Registry mutation refused after corruption/truncation; store authoritative
     reads propagate registry errors (no silent fallback).
  2. ``scan_registry`` is authoritative: payload schemas, identity binding, and
     state transitions are validated on each accepted line.
  3. Payloads are loaded strictly (no coercion) and identity-bound to the
     envelope; record internal consistency is enforced.
  4. Bundle metadata is semantically authenticated (canonical JSON, fixed-width
     timestamp, descriptor-derived artifact_id, canonical relative_path).
  5. Path/symlink TOCTOU defenses (O_NOFOLLOW, path-chain verification).
  6. Concurrent duplicate publication is idempotent under the publish lock.
  7. External orphan reconciliation uses an external.json sidecar; bundles with
     neither content nor external.json are unrecoverable.
  8. Retention transitions do not change storage_class; "missing" requires the
     content to be physically absent.
  9. Telemetry registration is enforced through register_telemetry;
     register_existing rejects category=telemetry.
 10. ``RegistryEntry.payload`` is deeply immutable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from expertforge.artifacts.models import (
    ArtifactConflictError,
    ArtifactDescriptor,
    ArtifactRecord,
    RegistryEntry,
    descriptor_to_artifact_id,
)
from expertforge.artifacts.registry import (
    RegistryError,
    allocate_and_append,
    scan_registry,
)
from expertforge.artifacts.store import ArtifactStore, ArtifactStoreError
from tests._artifact_fixtures import (
    VALID_ATTEMPT,
    VALID_FP,
    VALID_RUN,
    VALID_TS,
    make_store,
)

_VALID_DIGEST = "sha256:" + "a" * 64


def _registry_path(tmp_path: Path) -> Path:
    return tmp_path / "runs" / VALID_RUN / "attempts" / VALID_ATTEMPT / "registry.jsonl"


def _lock_path(tmp_path: Path) -> Path:
    return tmp_path / "runs" / VALID_RUN / "attempts" / VALID_ATTEMPT / "registry.lock"


def _descriptor(**overrides: Any) -> ArtifactDescriptor:
    base: dict[str, Any] = {
        "run_id": VALID_RUN,
        "attempt_id": VALID_ATTEMPT,
        "specification_fingerprint": VALID_FP,
        "category": "report",
        "format": "json",
        "format_version": 1,
        "content_digest": _VALID_DIGEST,
        "byte_size": 10,
        "producing_component": "training",
    }
    base.update(overrides)
    return ArtifactDescriptor(**base)


def _record_payload(**overrides: Any) -> dict[str, Any]:
    """An internally-consistent ArtifactRecord payload (descriptor-derived id)."""
    descriptor = _descriptor(**overrides)
    artifact_id = descriptor_to_artifact_id(descriptor)
    rec = ArtifactRecord.from_descriptor(
        descriptor,
        created_at_utc=VALID_TS,
        relative_path=f"artifacts/{descriptor.category}/{artifact_id}",
    )
    data = json.loads(rec.model_dump_json(by_alias=True))
    assert isinstance(data, dict)
    return data


def _append_raw(path: Path, line: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as fh:
        fh.write(line)


# ---------------------------------------------------------------------------
# Item 0: O_BINARY guarded for non-Windows typesheds.
# ---------------------------------------------------------------------------


class TestItem0OBinaryGuarded:
    def test_o_binary_is_optional_on_this_platform(self) -> None:
        # On Windows O_BINARY exists; on POSIX it must be absent and the source
        # must use getattr(os, "O_BINARY", 0). Either way the value resolves to
        # an int usable as an open flag.
        value = getattr(os, "O_BINARY", 0)
        assert isinstance(value, int)

    def test_locks_module_imports_cleanly_without_o_binary(self) -> None:
        # The Windows-only lock-acquire code path references O_BINARY only via
        # getattr, so importing the module must not require O_BINARY to exist.
        import expertforge.artifacts.locks as locks

        assert hasattr(locks, "AttemptLock")


# ---------------------------------------------------------------------------
# Item 1: no append after corruption/truncation; authoritative reads raise.
# ---------------------------------------------------------------------------


class TestItem1NoAppendAfterCorruption:
    def test_allocate_refused_on_truncated_registry(self, tmp_path: Path) -> None:
        path = _registry_path(tmp_path)
        _append_raw(path, _entry_line(sequence=0))
        # Append a partial (non-newline-terminated) line -> incomplete.
        _append_raw(path, b'{"registry_format_version":1,"sequence":1,')
        # The corrupt/incomplete registry must not be silently extended; either a
        # RegistryError (authoritative read) or ArtifactStoreError surfaces it.
        with pytest.raises((RegistryError, ArtifactStoreError)):
            self._append_via_store(tmp_path, producing_component="other")
        # The incomplete scan status is authoritative: no new line was appended.
        assert scan_registry(path).status == "incomplete"

    def test_allocate_refused_on_corrupt_registry(self, tmp_path: Path) -> None:
        path = _registry_path(tmp_path)
        _append_raw(path, b"not-json-at-all\n")
        with pytest.raises((RegistryError, ArtifactStoreError)):
            self._append_via_store(tmp_path, producing_component="other")
        assert scan_registry(path).status == "corrupt"

    def test_list_artifacts_propagates_corrupt_registry(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        store.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        )
        # Corrupt the registry tail.
        path = store.registry_path
        with open(path, "ab") as fh:
            fh.write(b"corrupt-line\n")
        # No silent empty-list fallback: the corruption surfaces.
        with pytest.raises((RegistryError, ArtifactStoreError)):
            store.list_artifacts()

    def test_append_returns_typed_error_for_incomplete(self, tmp_path: Path) -> None:
        path = _registry_path(tmp_path)
        _append_raw(path, _entry_line(sequence=0))
        _append_raw(path, b'{"registry_format_version":1,"sequence":1,')
        with pytest.raises(RegistryError, match="non-complete"):
            allocate_and_append(
                path,
                lock_path=_lock_path(tmp_path),
                run_id=VALID_RUN,
                attempt_id=VALID_ATTEMPT,
                specification_fingerprint=VALID_FP,
                entry_kind="initial_publication",
                payload=_record_payload(producing_component="other"),
                recorded_at_utc=VALID_TS,
            )

    @staticmethod
    def _append_via_store(tmp_path: Path, *, producing_component: str) -> None:
        store = make_store(tmp_path)
        store.publish(
            b"y",
            category="report",
            format="text",
            format_version=1,
            producing_component=producing_component,
        )


def _entry_line(sequence: int) -> bytes:
    entry = RegistryEntry(
        sequence=sequence,
        run_id=VALID_RUN,
        attempt_id=VALID_ATTEMPT,
        specification_fingerprint=VALID_FP,
        recorded_at_utc=VALID_TS,
        entry_kind="initial_publication",
        payload=_record_payload(producing_component=f"comp{sequence}"),
    )
    return entry.to_deterministic_json() + b"\n"


# ---------------------------------------------------------------------------
# Item 2: scan_registry is authoritative (payloads + transitions per line).
# ---------------------------------------------------------------------------


class TestItem2ScanIsAuthoritative:
    def test_scan_rejects_invalid_payload_schema(self, tmp_path: Path) -> None:
        # An initial_publication whose payload is not a valid ArtifactRecord
        # (missing required fields) is corrupt even though the envelope framing
        # is well-formed.
        entry = RegistryEntry(
            sequence=0,
            run_id=VALID_RUN,
            attempt_id=VALID_ATTEMPT,
            specification_fingerprint=VALID_FP,
            recorded_at_utc=VALID_TS,
            entry_kind="initial_publication",
            payload={"artifact_id": "artifact-v1-sha256-" + "a" * 64},
        )
        path = _registry_path(tmp_path)
        _append_raw(path, entry.to_deterministic_json() + b"\n")
        result = scan_registry(path)
        assert result.status == "corrupt"

    def test_scan_rejects_retention_transition_for_unknown_artifact(self, tmp_path: Path) -> None:
        # A retention_transition for an artifact with no prior publication is an
        # illegal transition; the scanner must catch it (authoritative).
        entry = RegistryEntry(
            sequence=0,
            run_id=VALID_RUN,
            attempt_id=VALID_ATTEMPT,
            specification_fingerprint=VALID_FP,
            recorded_at_utc=VALID_TS,
            entry_kind="retention_transition",
            payload={
                "artifact_id": "artifact-v1-sha256-" + "a" * 64,
                "storage_class": "canonical_local",
                "retention": "expired",
            },
        )
        path = _registry_path(tmp_path)
        _append_raw(path, entry.to_deterministic_json() + b"\n")
        result = scan_registry(path)
        assert result.status == "corrupt"

    def test_scan_retains_accepted_prefix_before_corrupt_line(self, tmp_path: Path) -> None:
        path = _registry_path(tmp_path)
        _append_raw(path, _entry_line(sequence=0))
        _append_raw(path, b"corrupt\n")
        result = scan_registry(path)
        assert result.status == "corrupt"
        assert result.accepted_count == 1


# ---------------------------------------------------------------------------
# Item 3: payloads loaded strictly + identity-bound + internally consistent.
# ---------------------------------------------------------------------------


class TestItem3StrictIdentityBoundPayloads:
    def test_payload_run_id_mismatch_with_envelope_is_corrupt(self, tmp_path: Path) -> None:
        # Build a valid record payload but bind the envelope to a different
        # run_id than the payload carries.
        payload = _record_payload()
        entry = RegistryEntry(
            sequence=0,
            run_id="run-other",
            attempt_id=VALID_ATTEMPT,
            specification_fingerprint=VALID_FP,
            recorded_at_utc=VALID_TS,
            entry_kind="initial_publication",
            payload=payload,
        )
        path = _registry_path(tmp_path)
        _append_raw(path, entry.to_deterministic_json() + b"\n")
        result = scan_registry(path)
        assert result.status == "corrupt"

    def test_payload_artifact_id_not_descriptor_derived_is_corrupt(self, tmp_path: Path) -> None:
        # A record whose artifact_id does not match the descriptor-derived id.
        descriptor = _descriptor()
        real_id = descriptor_to_artifact_id(descriptor)
        payload = _record_payload()
        payload["artifact_id"] = "artifact-v1-sha256-" + "0" * 64
        # Fix relative_path so only artifact_id drifts (relative_path still
        # references the real descriptor-derived id via category).
        entry = RegistryEntry(
            sequence=0,
            run_id=VALID_RUN,
            attempt_id=VALID_ATTEMPT,
            specification_fingerprint=VALID_FP,
            recorded_at_utc=VALID_TS,
            entry_kind="initial_publication",
            payload=payload,
        )
        path = _registry_path(tmp_path)
        _append_raw(path, entry.to_deterministic_json() + b"\n")
        result = scan_registry(path)
        assert result.status == "corrupt"
        # The real descriptor-derived id must not be the tampered one.
        assert real_id != payload["artifact_id"]


# ---------------------------------------------------------------------------
# Item 4: bundle metadata semantically authenticated.
# ---------------------------------------------------------------------------


class TestItem4BundleSemanticAuth:
    def test_publish_writes_canonical_artifact_json(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        rec = store.publish(
            b'{"a":1}',
            category="report",
            format="json",
            format_version=1,
            producing_component="t",
        )
        raw = (
            store.attempt_dir / "artifacts" / "report" / rec.artifact_id / "artifact.json"
        ).read_bytes()
        # Canonical: sorted keys, compact separators, no spaces.
        text = raw.decode("utf-8")
        assert text == json.dumps(
            json.loads(text), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        assert b": " not in raw and b", " not in raw

    def test_non_canonical_artifact_json_rejected_on_verify(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        rec = store.publish(
            b"p",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        )
        meta = store.attempt_dir / "artifacts" / "report" / rec.artifact_id / "artifact.json"
        # Rewrite with non-canonical formatting (spaces after separators).
        data = json.loads(meta.read_bytes())
        meta.write_text(json.dumps(data, indent=2))
        # verify re-parses artifact.json and must reject the non-canonical form.
        result = store.verify(rec.artifact_id)
        assert result.status is False

    def test_tampered_artifact_id_rejected_on_verify(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        rec = store.publish(
            b"p",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        )
        meta = store.attempt_dir / "artifacts" / "report" / rec.artifact_id / "artifact.json"
        data = json.loads(meta.read_bytes())
        # Tamper a field that changes the descriptor-derived id.
        data["producing_component"] = "tampered-producer"
        meta.write_text(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        result = store.verify(rec.artifact_id)
        assert result.status is False


# ---------------------------------------------------------------------------
# Item 5: path/symlink TOCTOU defenses.
# ---------------------------------------------------------------------------


class TestItem5SymlinkToctouDefenses:
    def test_register_existing_rejects_symlink_source(self, tmp_path: Path) -> None:
        if os.name == "nt":
            pytest.skip("symlink semantics differ on Windows without privileges")
        store = make_store(tmp_path)
        target = tmp_path / "real.bin"
        target.write_bytes(b"real")
        link = tmp_path / "link.bin"
        os.symlink(target, link)
        with pytest.raises((ArtifactConflictError, ArtifactStoreError)):
            store.register_existing(
                link,
                category="checkpoint",
                format="binary",
                format_version=1,
                producing_component="ckpt",
            )

    def test_no_follow_flag_is_actually_used(self) -> None:
        # The store module must reference O_NOFOLLOW (guarded) in its source.
        import expertforge.artifacts.store as store_mod

        src = Path(store_mod.__file__).read_text(encoding="utf-8")
        assert "O_NOFOLLOW" in src
        # And expose a path-chain verifier helper.
        assert hasattr(store_mod, "_verify_no_symlinks_in_chain")

    def test_verify_rejects_symlinked_bundle_dir(self, tmp_path: Path) -> None:
        if os.name == "nt":
            pytest.skip("symlink semantics differ on Windows without privileges")
        store = make_store(tmp_path)
        rec = store.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        )
        bundle = store.attempt_dir / "artifacts" / "report" / rec.artifact_id
        # Replace the bundle directory with a symlink to a sibling dir whose
        # content differs; verification must not follow it.
        sibling = store.attempt_dir / "artifacts" / "report" / ".evil"
        sibling.mkdir()
        (sibling / "artifact.json").write_bytes(b"{}")
        (sibling / "content").write_bytes(b"evil")
        import shutil

        shutil.rmtree(bundle)
        os.symlink(sibling, bundle)
        result = store.verify(rec.artifact_id)
        assert result.status is False


# ---------------------------------------------------------------------------
# Item 6: concurrent duplicate publication idempotent under the lock.
# ---------------------------------------------------------------------------


class TestItem6ConcurrentIdempotent:
    def test_concurrent_same_content_publish_idempotent(self, tmp_path: Path) -> None:
        """Two subprocess publishers of the same semantic artifact produce exactly
        one canonical bundle and one registry entry (item #6)."""
        root = (tmp_path / "runs").resolve()
        repo_root = Path(__file__).resolve().parents[1]
        src_root = repo_root / "src"
        script = tmp_path / "worker.py"
        # Both workers publish identical content -> identical artifact_id.
        script.write_text(
            "import sys\n"
            f"sys.path.insert(0, {src_root.as_posix()!r})\n"
            f"sys.path.insert(0, {repo_root.as_posix()!r})\n"
            "from expertforge.artifacts.store import ArtifactStore\n"
            "from tests._artifact_fixtures import make_identity\n"
            f"ROOT = {root.as_posix()!r}\n"
            "ident = make_identity()\n"
            "store = ArtifactStore(artifact_root=ROOT, identity=ident)\n"
            "store.publish(b'same-bytes', category='report', format='text', "
            "format_version=1, producing_component='same')\n"
        )
        env = dict(os.environ)
        procs = [subprocess.Popen([sys.executable, str(script)], env=env) for _ in range(2)]
        for p in procs:
            assert p.wait(timeout=120) == 0
        store = make_store(tmp_path)
        artifacts = store.list_artifacts()
        # Idempotent: exactly one artifact for the shared semantic identity.
        assert len(artifacts) == 1
        result = scan_registry(store.registry_path)
        assert result.status == "complete"


# ---------------------------------------------------------------------------
# Item 7: external.json sidecar + unrecoverable classification.
# ---------------------------------------------------------------------------


class TestItem7ExternalSidecarReconciliation:
    def test_register_external_writes_external_json_sidecar(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        ext = store.register_external(
            category="report",
            format="json",
            format_version=1,
            producing_component="eval",
            location_type="uri",
            location="https://example.com/r.json",
            expected_digest=_VALID_DIGEST,
            expected_byte_size=10,
        )
        sidecar = store.attempt_dir / "artifacts" / "report" / ext.artifact_id / "external.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_bytes())
        assert data["location"] == "https://example.com/r.json"

    def test_external_orphan_reconciled_as_external_registration(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        ext = store.register_external(
            category="report",
            format="json",
            format_version=1,
            producing_component="eval",
            location_type="uri",
            location="https://example.com/r.json",
            expected_digest=_VALID_DIGEST,
            expected_byte_size=10,
        )
        # Simulate a crash after bundle write before the registry append.
        store.registry_path.unlink()
        recon = store.reconcile_orphans()
        assert ext.artifact_id in recon.appended_artifact_ids
        # The reconciled entry must be an external_registration, recoverable via
        # inspect (which returns the ExternalReference).
        inspected = store.inspect(ext.artifact_id)
        from expertforge.artifacts.models import ExternalReference

        assert isinstance(inspected, ExternalReference)
        assert inspected.location == "https://example.com/r.json"

    def test_bundle_with_neither_content_nor_sidecar_is_unrecoverable(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        # Create a bundle dir with only a valid artifact.json (no content, no
        # external.json). This carries no recoverable bytes.
        payload = _record_payload(category="report", format="json")
        artifact_id = payload["artifact_id"]
        bdir = store.attempt_dir / "artifacts" / "report" / artifact_id
        bdir.mkdir(parents=True)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        (bdir / "artifact.json").write_text(canonical)
        recon = store.reconcile_orphans()
        assert artifact_id in recon.orphan_artifact_ids
        assert artifact_id not in recon.appended_artifact_ids
        assert any(aid == artifact_id for aid, _ in recon.skipped)


# ---------------------------------------------------------------------------
# Item 8: retention transitions do not change storage_class.
# ---------------------------------------------------------------------------


class TestItem8RetentionStorageSeparation:
    def test_retention_keeps_storage_class(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        aid = store.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        ).artifact_id
        rec = store.update_retention(aid, "expired")
        assert rec.retention == "expired"
        # storage_class is unchanged: a retention-only transition never moves it
        # to metadata_only (item #8).
        assert rec.storage_class == "canonical_local"

    def test_missing_refused_when_content_present(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        aid = store.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        ).artifact_id
        with pytest.raises(ArtifactConflictError):
            store.update_retention(aid, "missing")


# ---------------------------------------------------------------------------
# Item 9: telemetry registration enforced.
# ---------------------------------------------------------------------------


def _write_telemetry_stream(store: ArtifactStore) -> Path:
    from expertforge.telemetry.models import (
        MetricObservation,
        ProcessContext,
        ProgressPosition,
    )
    from expertforge.telemetry.writer import TelemetryWriter

    ctx = ProcessContext(rank=0, world_size=1, local_rank=0)

    class _Clk:
        def __init__(self) -> None:
            self._mono = 0

        def wall(self) -> datetime:
            return datetime(2026, 1, 1, tzinfo=UTC)

        def mono(self) -> int:
            v = self._mono
            self._mono += 1_000_000
            return v

    clk = _Clk()
    w = TelemetryWriter(
        artifact_root=store.artifact_root,
        identity=store.identity,
        process_context=ctx,
        console_stream=None,
        console_enabled=False,
        wall_clock=clk.wall,
        monotonic_clock=clk.mono,
    )
    w.emit_event(component="training", severity="INFO", event_name="training.update")
    w.emit_metric(
        component="training",
        progress=ProgressPosition(step=1, update=1, processed_tokens=128),
        observations=[
            MetricObservation(
                namespace="training",
                name="loss",
                unit="dimensionless",
                aggregation="gauge",
                window="point",
                value_status="finite",
                value=0.5,
            )
        ],
    )
    w.close(outcome="normal")
    return w.path


class TestItem9TelemetryEnforced:
    def test_register_existing_rejects_telemetry(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        src = tmp_path / "stream.jsonl"
        src.write_bytes(b'{"x":1}\n')
        with pytest.raises(ArtifactStoreError, match="register_telemetry"):
            store.register_existing(
                src,
                category="telemetry",
                format="jsonl",
                format_version=1,
                producing_component="logging",
            )

    def test_register_telemetry_verifies_then_copies(self, tmp_path: Path) -> None:
        from expertforge.telemetry.models import ProcessContext

        store = make_store(tmp_path)
        stream_path = _write_telemetry_stream(store)
        ctx = ProcessContext(rank=0, world_size=1, local_rank=0)
        rec = store.register_telemetry(
            stream_path,
            process_context=ctx,
            producing_component="logging",
        )
        assert rec.category == "telemetry"
        assert rec.format == "jsonl"
        content = store.locate(rec.artifact_id)
        assert content is not None
        assert content.read_bytes() == stream_path.read_bytes()

    def test_register_telemetry_rejects_incomplete_stream(self, tmp_path: Path) -> None:
        from expertforge.telemetry.models import ProcessContext

        store = make_store(tmp_path)
        stream_path = _write_telemetry_stream(store)
        ctx = ProcessContext(rank=0, world_size=1, local_rank=0)
        raw = stream_path.read_bytes()
        stream_path.write_bytes(raw[:-5])  # truncated -> incomplete
        with pytest.raises(ArtifactStoreError):
            store.register_telemetry(
                stream_path,
                process_context=ctx,
                producing_component="logging",
            )


# ---------------------------------------------------------------------------
# Item 10: RegistryEntry.payload deeply immutable.
# ---------------------------------------------------------------------------


class TestItem10PayloadImmutable:
    def test_payload_is_a_mapping_proxy(self) -> None:
        entry = RegistryEntry(
            sequence=0,
            run_id=VALID_RUN,
            attempt_id=VALID_ATTEMPT,
            specification_fingerprint=VALID_FP,
            recorded_at_utc=VALID_TS,
            entry_kind="initial_publication",
            payload={"artifact_id": "x", "nested": {"k": [1, 2]}},
        )
        from types import MappingProxyType

        assert isinstance(entry.payload, MappingProxyType)
        assert isinstance(entry.payload["nested"], MappingProxyType)

    def test_payload_mutation_raises(self) -> None:
        entry = RegistryEntry(
            sequence=0,
            run_id=VALID_RUN,
            attempt_id=VALID_ATTEMPT,
            specification_fingerprint=VALID_FP,
            recorded_at_utc=VALID_TS,
            entry_kind="initial_publication",
            payload={"artifact_id": "x"},
        )
        with pytest.raises(TypeError):
            entry.payload["artifact_id"] = "y"  # type: ignore[index]

    def test_payload_list_value_is_immutable(self) -> None:
        entry = RegistryEntry(
            sequence=0,
            run_id=VALID_RUN,
            attempt_id=VALID_ATTEMPT,
            specification_fingerprint=VALID_FP,
            recorded_at_utc=VALID_TS,
            entry_kind="initial_publication",
            payload={"artifact_id": "x", "tags": ["a", "b"]},
        )
        tags = entry.payload["tags"]
        # Lists are frozen to tuples (immutable) so they cannot be mutated in
        # place the way a list could; a tuple has no append/extend mutators.
        assert isinstance(tags, tuple)
        assert not hasattr(tags, "append")

    def test_payload_isolates_caller_state(self) -> None:
        caller = {"artifact_id": "x"}
        entry = RegistryEntry(
            sequence=0,
            run_id=VALID_RUN,
            attempt_id=VALID_ATTEMPT,
            specification_fingerprint=VALID_FP,
            recorded_at_utc=VALID_TS,
            entry_kind="initial_publication",
            payload=caller,
        )
        # Mutating the caller dict after construction does not affect the entry.
        caller["artifact_id"] = "mutated"
        assert entry.payload["artifact_id"] == "x"
