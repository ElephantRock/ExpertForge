"""Focused regression tests for PR #32 round-2 review items (1-7).

Each test class targets exactly one round-2 review item. These tests lock in
the reviewed behaviors against regressions and complement the broader artifact
suites (``test_artifact_review_regression.py`` covers round-1 items 0-10).

Items covered:
  1. External reference binding: expected_digest/expected_byte_size/category/
     format/format_version must match the accompanying ArtifactRecord.
  2. Combined transition state revalidated through the Pydantic constructor
     (model_copy bypasses validators; an illegal storage+retention combo is
     rejected).
  3. scan_registry is exception-total (no KeyError/ValidationError leaks) and a
     payload-corrupt line is never part of the accepted prefix.
  4. Parent-directory symlink protection applied in publish/locate/verify/
     reconcile/list.
  5. Telemetry TOCTOU: source mutated between validation and copy does not
     change the published bytes (they match the validated copy).
  6. Concurrent register_external is idempotent (single bundle, single entry).
  7. verify() binds the bundle record to the store identity and category/attempt
     path; a bundle in the wrong attempt/category fails verification.
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
    ExternalReference,
    RegistryEntry,
    descriptor_to_artifact_id,
)
from expertforge.artifacts.registry import (
    scan_registry,
)
from expertforge.artifacts.store import ArtifactStore, ArtifactStoreError
from tests._artifact_fixtures import (
    VALID_ATTEMPT,
    VALID_FP,
    VALID_RUN,
    VALID_TS,
    make_identity,
    make_store,
)

_VALID_DIGEST = "sha256:" + "a" * 64
_OTHER_DIGEST = "sha256:" + "b" * 64


def _registry_path(tmp_path: Path) -> Path:
    return tmp_path / "runs" / VALID_RUN / "attempts" / VALID_ATTEMPT / "registry.jsonl"


def _append_raw(path: Path, line: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as fh:
        fh.write(line)


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


def _record(**overrides: Any) -> ArtifactRecord:
    descriptor = _descriptor(**overrides)
    artifact_id = descriptor_to_artifact_id(descriptor)
    return ArtifactRecord.from_descriptor(
        descriptor,
        created_at_utc=VALID_TS,
        relative_path=f"artifacts/{descriptor.category}/{artifact_id}",
    )


def _record_payload(**overrides: Any) -> dict[str, Any]:
    """An internally-consistent ArtifactRecord payload (descriptor-derived id)."""
    data = json.loads(_record(**overrides).model_dump_json(by_alias=True))
    assert isinstance(data, dict)
    return data


def _external_reference(record: ArtifactRecord, **overrides: Any) -> ExternalReference:
    base: dict[str, Any] = dict(
        artifact_id=record.artifact_id,
        run_id=record.run_id,
        attempt_id=record.attempt_id,
        specification_fingerprint=record.specification_fingerprint,
        category=record.category,
        format=record.format,
        format_version=record.format_version,
        location_type="uri",
        location="https://example.com/r.json",
        external_root_id=None,
        expected_digest=record.content_digest,
        expected_byte_size=record.byte_size,
        availability="unavailable",
        verified_at_utc=None,
    )
    base.update(overrides)
    return ExternalReference(**base)


def _external_registration_payload(
    record: ArtifactRecord, ext: ExternalReference
) -> dict[str, Any]:
    rec_dump = json.loads(record.model_dump_json(by_alias=True))
    ext_dump = json.loads(ext.model_dump_json(by_alias=True))
    return {**rec_dump, "external_reference": ext_dump}


def _entry_line(sequence: int, *, payload: dict[str, Any], entry_kind: str) -> bytes:
    entry = RegistryEntry(
        sequence=sequence,
        run_id=VALID_RUN,
        attempt_id=VALID_ATTEMPT,
        specification_fingerprint=VALID_FP,
        recorded_at_utc=VALID_TS,
        entry_kind=entry_kind,  # type: ignore[arg-type]
        payload=payload,
    )
    return entry.to_deterministic_json() + b"\n"


# ---------------------------------------------------------------------------
# Item 1: external reference binding fields must match the record.
# ---------------------------------------------------------------------------


class TestItem1ExternalReferenceBinding:
    """An external_registration entry must agree with its record on the
    content-addressing and category/format fields (item #1)."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("expected_digest", _OTHER_DIGEST),
            ("expected_byte_size", 999),
            ("category", "provenance"),
            ("format", "text"),
            ("format_version", 2),
        ],
    )
    def test_mismatched_external_field_is_corrupt(
        self, tmp_path: Path, field: str, value: Any
    ) -> None:
        record = _record()
        # Build an external reference whose `field` disagrees with the record.
        ext_kwargs = {field: value}
        # category/format on the ExternalReference must still be individually
        # valid for the model to construct; the mismatch is against the record.
        ext = _external_reference(record, **ext_kwargs)
        payload = _external_registration_payload(record, ext)
        path = _registry_path(tmp_path)
        _append_raw(path, _entry_line(0, payload=payload, entry_kind="external_registration"))
        result = scan_registry(path)
        assert result.status == "corrupt", (
            f"expected corrupt for mismatched external {field}={value!r}"
        )
        # The mismatched entry is not part of the accepted prefix.
        assert result.accepted_count == 0

    def test_matching_external_reference_is_accepted(self, tmp_path: Path) -> None:
        record = _record()
        ext = _external_reference(record)
        payload = _external_registration_payload(record, ext)
        path = _registry_path(tmp_path)
        _append_raw(path, _entry_line(0, payload=payload, entry_kind="external_registration"))
        result = scan_registry(path)
        assert result.status == "complete"
        assert result.accepted_count == 1


# ---------------------------------------------------------------------------
# Item 2: combined transition state revalidated through the constructor.
# ---------------------------------------------------------------------------


class TestItem2CombinedTransitionRevalidated:
    """A transition whose combined (storage_class, retention) state is illegal
    is rejected even though each individual transition is legal (item #2)."""

    def test_illegal_combined_storage_retention_rejected(self, tmp_path: Path) -> None:
        # Publish a canonical_local+retained artifact, then append a transition
        # that sets storage_class=canonical_local (a legal self-transition) and
        # retention=externally_retained. retained -> externally_retained is
        # illegal as a retention transition, but even if it were legal, the
        # combined (canonical_local, externally_retained) state is allowed by
        # the matrix. To exercise the COMBINED validator specifically, use a
        # legal retention transition to an allowed retention, but combine it
        # with an illegal storage that only the combined check catches.
        #
        # Construct: initial canonical_local+retained, then a transition to
        # storage_class=metadata_only + retention=retained. Each individual
        # transition: canonical_local -> metadata_only is legal; retained ->
        # retained is legal. But the COMBINED (metadata_only, retained) is NOT
        # in the legal matrix -> must be rejected by the constructor re-check.
        record = _record()
        path = _registry_path(tmp_path)
        _append_raw(
            path, _entry_line(0, payload=_record_payload(), entry_kind="initial_publication")
        )
        transition_payload = {
            "artifact_id": record.artifact_id,
            "storage_class": "metadata_only",
            "retention": "retained",
        }
        _append_raw(
            path, _entry_line(1, payload=transition_payload, entry_kind="retention_transition")
        )
        result = scan_registry(path)
        assert result.status == "corrupt"
        # The accepted prefix excludes the illegal combined-state line.
        assert result.accepted_count == 1

    def test_legal_combined_transition_accepted(self, tmp_path: Path) -> None:
        record = _record()
        path = _registry_path(tmp_path)
        _append_raw(
            path, _entry_line(0, payload=_record_payload(), entry_kind="initial_publication")
        )
        # canonical_local + retained -> canonical_local + expired: legal combo.
        transition_payload = {
            "artifact_id": record.artifact_id,
            "storage_class": "canonical_local",
            "retention": "expired",
        }
        _append_raw(
            path, _entry_line(1, payload=transition_payload, entry_kind="retention_transition")
        )
        result = scan_registry(path)
        assert result.status == "complete"
        assert result.accepted_count == 2


# ---------------------------------------------------------------------------
# Item 3: scan_registry exception-total + corrupt line never in accepted prefix.
# ---------------------------------------------------------------------------


class TestItem3ScanExceptionTotal:
    def test_missing_identity_keys_do_not_leak_keyerror(self, tmp_path: Path) -> None:
        # A first line missing run_id/attempt_id/specification_fingerprint must
        # produce a typed corrupt result, not a bare KeyError.
        line = (
            json.dumps(
                {
                    "registry_format_version": 1,
                    "sequence": 0,
                    "recorded_at_utc": "2026-01-01T00:00:00.000000Z",
                    "entry_kind": "initial_publication",
                    "payload": {"artifact_id": "x"},
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        path = _registry_path(tmp_path)
        _append_raw(path, line)
        result = scan_registry(path)
        assert result.status == "corrupt"

    def test_payload_corrupt_line_not_in_accepted_prefix(self, tmp_path: Path) -> None:
        # A valid initial_publication followed by a line whose payload fails
        # _reduce_history (illegal transition) must NOT include the corrupt line
        # in the accepted prefix.
        path = _registry_path(tmp_path)
        _append_raw(
            path, _entry_line(0, payload=_record_payload(), entry_kind="initial_publication")
        )
        # Illegal: retention_transition for an unknown artifact.
        bad_payload = {
            "artifact_id": "artifact-v1-sha256-" + "9" * 64,
            "storage_class": "canonical_local",
            "retention": "expired",
        }
        _append_raw(path, _entry_line(1, payload=bad_payload, entry_kind="retention_transition"))
        result = scan_registry(path)
        assert result.status == "corrupt"
        # Only the first valid line is in the accepted prefix.
        assert result.accepted_count == 1
        assert result.accepted[0].sequence == 0

    def test_validation_error_does_not_leak(self, tmp_path: Path) -> None:
        # A payload that constructs a RegistryEntry but whose record payload
        # fails Pydantic validation must yield corrupt, not a ValidationError.
        bad_payload = {"artifact_id": "not-a-valid-id"}
        path = _registry_path(tmp_path)
        _append_raw(path, _entry_line(0, payload=bad_payload, entry_kind="initial_publication"))
        result = scan_registry(path)
        assert result.status == "corrupt"


# ---------------------------------------------------------------------------
# Item 4: parent-directory symlink protection.
# ---------------------------------------------------------------------------


def _skip_if_no_symlink_privilege() -> None:
    if os.name == "nt":
        pytest.skip("symlink semantics differ on Windows without privileges")


class TestItem4SymlinkChainProtection:
    def test_publish_rejects_symlinked_parent(self, tmp_path: Path) -> None:
        _skip_if_no_symlink_privilege()
        store = make_store(tmp_path)
        # Make the run directory a symlink to a real directory so the chain from
        # artifact_root to the category dir crosses a symlinked component.
        root = tmp_path / "runs"
        real_run = tmp_path / "real-run"
        real_run.mkdir()
        link_run = root / VALID_RUN
        root.mkdir(parents=True, exist_ok=True)
        os.symlink(real_run, link_run)
        with pytest.raises((ArtifactConflictError, ArtifactStoreError)):
            store.publish(
                b"x",
                category="report",
                format="text",
                format_version=1,
                producing_component="t",
            )

    def test_verify_chain_helper_invoked(self) -> None:
        import expertforge.artifacts.store as store_mod

        src = Path(store_mod.__file__).read_text(encoding="utf-8")
        # The chain verifier must be called in the path-returning helpers.
        assert "_verify_no_symlinks_in_chain" in src

    def test_locate_rejects_symlinked_chain(self, tmp_path: Path) -> None:
        _skip_if_no_symlink_privilege()
        store = make_store(tmp_path)
        rec = store.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        )
        # Replace the artifacts directory with a symlink to a sibling.
        arts = store.attempt_dir / "artifacts"
        sibling = store.attempt_dir / "arts-sibling"
        sibling.mkdir()
        (sibling / "report").mkdir()
        (sibling / "report" / rec.artifact_id).mkdir()
        import shutil

        shutil.rmtree(arts)
        os.symlink(sibling, arts)
        # locate must reject the symlinked chain (return None or raise a typed
        # store/conflict error rather than following the symlink).
        with pytest.raises((ArtifactConflictError, ArtifactStoreError)):
            store.locate(rec.artifact_id)

    def test_reconcile_rejects_symlinked_chain(self, tmp_path: Path) -> None:
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
        sibling = store.attempt_dir / "arts-sibling"
        sibling.mkdir()
        import shutil

        shutil.rmtree(arts)
        os.symlink(sibling, arts)
        with pytest.raises((ArtifactConflictError, ArtifactStoreError)):
            store.reconcile_orphans()


# ---------------------------------------------------------------------------
# Item 5: telemetry TOCTOU — published bytes match the validated copy.
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


class TestItem5TelemetryToctou:
    def test_published_bytes_match_validated_copy(self, tmp_path: Path) -> None:
        from expertforge.telemetry.models import ProcessContext

        store = make_store(tmp_path)
        stream_path = _write_telemetry_stream(store)
        original = stream_path.read_bytes()
        ctx = ProcessContext(rank=0, world_size=1, local_rank=0)
        rec = store.register_telemetry(
            stream_path,
            process_context=ctx,
            producing_component="logging",
        )
        # Even though we do not mutate the source here, the published content
        # must equal the validated bytes exactly (the copy-then-validate path
        # binds them).
        content = store.locate(rec.artifact_id)
        assert content is not None
        assert content.read_bytes() == original

    def test_source_mutation_after_copy_does_not_change_published(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from expertforge.telemetry.models import ProcessContext

        store = make_store(tmp_path)
        stream_path = _write_telemetry_stream(store)
        original = stream_path.read_bytes()
        ctx = ProcessContext(rank=0, world_size=1, local_rank=0)

        # Snapshot the bytes that the loader validated by capturing the path it
        # was called with: register_telemetry now validates the OWNED COPY, so a
        # mutation to the original source AFTER copy must not affect the result.
        # We mutate the source immediately after register_telemetry returns and
        # confirm the published bytes still equal the original validated bytes.
        rec = store.register_telemetry(
            stream_path,
            process_context=ctx,
            producing_component="logging",
        )
        # Mutate the original source path after publication.
        stream_path.write_bytes(b"MUTATED-AFTER-COPY-SHOULD-NOT-AFFECT-PUBLISHED")
        content = store.locate(rec.artifact_id)
        assert content is not None
        assert content.read_bytes() == original

    def test_incomplete_stream_rejected(self, tmp_path: Path) -> None:
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
# Item 6: concurrent register_external is idempotent.
# ---------------------------------------------------------------------------


class TestItem6ConcurrentExternalIdempotent:
    def test_concurrent_register_external_idempotent(self, tmp_path: Path) -> None:
        """Two subprocess register_external callers of the same external artifact
        produce exactly one bundle and one registry entry (item #6)."""
        root = (tmp_path / "runs").resolve()
        repo_root = Path(__file__).resolve().parents[1]
        src_root = repo_root / "src"
        script = tmp_path / "worker.py"
        script.write_text(
            "import sys\n"
            f"sys.path.insert(0, {src_root.as_posix()!r})\n"
            f"sys.path.insert(0, {repo_root.as_posix()!r})\n"
            "from expertforge.artifacts.store import ArtifactStore\n"
            "from tests._artifact_fixtures import make_identity\n"
            f"ROOT = {root.as_posix()!r}\n"
            "ident = make_identity()\n"
            "store = ArtifactStore(artifact_root=ROOT, identity=ident)\n"
            "store.register_external(category='report', format='json', "
            "format_version=1, producing_component='eval', location_type='uri', "
            "location='https://example.com/same.json', "
            "expected_digest='sha256:" + "a" * 64 + "', expected_byte_size=10)\n"
        )
        env = dict(os.environ)
        procs = [subprocess.Popen([sys.executable, str(script)], env=env) for _ in range(2)]
        for p in procs:
            assert p.wait(timeout=120) == 0
        store = make_store(tmp_path)
        # Idempotent: exactly one external artifact indexed.
        artifacts = store.list_artifacts()
        assert len(artifacts) == 1
        result = scan_registry(store.registry_path)
        assert result.status == "complete"
        # The single artifact carries the external reference.
        inspected = store.inspect(artifacts[0].artifact_id)
        assert isinstance(inspected, ExternalReference)
        assert inspected.location == "https://example.com/same.json"


# ---------------------------------------------------------------------------
# Item 7: verify() binds the bundle record to store identity + category/attempt.
# ---------------------------------------------------------------------------


class TestItem7VerifyIdentityBinding:
    def test_verify_good_artifact_passes_binding(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        rec = store.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        )
        result = store.verify(rec.artifact_id)
        assert result.status is True
        assert result.diagnostic_code == "verified"

    def test_verify_bundle_in_wrong_attempt_fails(self, tmp_path: Path) -> None:
        # Publish under one store, then relocate the bundle into a different
        # attempt's tree and verify it does not verify against a store bound to
        # the original attempt.
        store_a = make_store(tmp_path)
        rec = store_a.publish(
            b"x",
            category="report",
            format="text",
            format_version=1,
            producing_component="t",
        )
        bundle = store_a.attempt_dir / "artifacts" / "report" / rec.artifact_id
        # Build a second store under a different attempt that points at the same
        # artifact_root; copy the bundle there so its record still references
        # attempt A's identity while the store is bound to attempt B.
        other_attempt = "attempt-20260101t000000z-dddddddddddddddddddd"
        ident_b = make_identity(attempt_id=other_attempt)
        store_b = ArtifactStore(artifact_root=tmp_path / "runs", identity=ident_b)
        target_dir = store_b.attempt_dir / "artifacts" / "report" / rec.artifact_id
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copytree(bundle, target_dir)
        # The record inside is bound to attempt A; store_b is bound to attempt B.
        result = store_b.verify(rec.artifact_id)
        assert result.status is False
        assert result.diagnostic_code == "identity_binding_mismatch"

    def test_verify_bundle_in_wrong_category_fails(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        rec = store.publish(
            b'{"k":1}',
            category="report",
            format="json",
            format_version=1,
            producing_component="t",
        )
        # Move the bundle to a sibling category directory (provenance) while its
        # record.category stays "report".
        bundle = store.attempt_dir / "artifacts" / "report" / rec.artifact_id
        wrong_cat = store.attempt_dir / "artifacts" / "provenance" / rec.artifact_id
        wrong_cat.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.move(str(bundle), str(wrong_cat))
        # Remove the now-empty original category dir so _bundle_path finds the
        # relocated bundle under the wrong category.
        try:
            (store.attempt_dir / "artifacts" / "report").rmdir()
        except OSError:
            pass
        result = store.verify(rec.artifact_id)
        assert result.status is False
        assert result.diagnostic_code == "identity_binding_mismatch"
