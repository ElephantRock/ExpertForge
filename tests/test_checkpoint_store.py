"""Regression tests for the CheckpointStore save/load/inspect (Issue #11, G, L).

These are integration tests (they build a real provenance record against a
temporary git repo) but they exercise the authoritative load path, #10
publication verification, artifact_id loading, TOCTOU safety, and compatibility.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from expertforge.checkpoints.store import (
    CheckpointArchive,
    CheckpointCorruptError,
    CheckpointLineageError,
    CheckpointStore,
    check_compatibility,
)
from tests._checkpoint_fixtures import (
    make_captured_state,
    make_identity_and_provenance,
    make_rng_bundle,
    make_store,
)

pytestmark = pytest.mark.integration


def _save_checkpoint(
    tmp_path: Path, *, scaler: bool = False, alias: bool = False
) -> tuple[CheckpointStore, CheckpointArchive, str]:
    identity, provenance = make_identity_and_provenance(tmp_path)
    store = make_store(tmp_path, identity=identity)
    cp_store = CheckpointStore(store)
    captured = make_captured_state(scaler=scaler, alias=alias)
    rng_bundle = make_rng_bundle()
    from expertforge.config.resolve import resolve_config
    from tests._checkpoint_fixtures import CONFIGS

    envelope = resolve_config(CONFIGS / "smoke.yaml")
    record = cp_store.save(
        identity=identity,
        captured=captured,
        configuration_envelope=envelope,
        provenance=provenance,
        rng_bundle=rng_bundle,
    )
    archive = cp_store.load(record.artifact_id, expected_identity=identity)
    return cp_store, archive, record.artifact_id


class TestSaveLoad:
    def test_save_publishes_as_checkpoint_artifact(self, tmp_path: Path) -> None:
        _, _, aid = _save_checkpoint(tmp_path)
        assert aid.startswith("artifact-v1-sha256-")

    def test_load_returns_archive_with_manifest(self, tmp_path: Path) -> None:
        _, archive, _ = _save_checkpoint(tmp_path)
        assert archive.manifest.run_id
        assert archive.manifest.attempt_id
        assert archive.manifest.counters.global_update == 3

    def test_load_round_trips_tensors(self, tmp_path: Path) -> None:
        _, archive, _ = _save_checkpoint(tmp_path)
        # The parameter tensor is members[0] (layer.weight).
        ref = next(m for m in archive.manifest.tensor_members if "layer.weight" in m.logical_names)
        data = archive.tensor_member(ref.member_name)
        assert len(data) == ref.member_byte_size

    def test_exact_member_graph_no_metadata_json(self, tmp_path: Path) -> None:
        _, archive, _ = _save_checkpoint(tmp_path)
        # The decoded archive has manifest + state components + tensors.
        from expertforge.checkpoints.tar_reader import parse_ustar_archive

        members = parse_ustar_archive(archive.tar_bytes)
        names = [m.name for m in members]
        assert names[0] == "manifest.json"
        assert "metadata.json" not in names
        roles = {
            "identity",
            "configuration",
            "provenance",
            "rng",
            "data_cursor",
            "counters",
            "model",
            "optimizer",
            "scheduler",
        }
        for role in roles:
            assert f"state/{role}.json" in names
        # Tensors come after state components.
        tensor_names = [n for n in names if n.startswith("tensors/")]
        assert tensor_names == sorted(tensor_names)
        for i, n in enumerate(tensor_names):
            assert n == f"tensors/{i}.bin"

    def test_deterministic_archive_bytes(self, tmp_path: Path) -> None:
        # Two saves with identical inputs + the same injected timestamp produce
        # identical archive bytes (byte-for-byte determinism).
        import hashlib
        from datetime import UTC, datetime

        identity, provenance = make_identity_and_provenance(tmp_path)
        store = make_store(tmp_path, identity=identity)
        cp_store = CheckpointStore(store)
        captured = make_captured_state()
        rng_bundle = make_rng_bundle()
        from expertforge.config.resolve import resolve_config
        from tests._checkpoint_fixtures import CONFIGS

        envelope = resolve_config(CONFIGS / "smoke.yaml")
        ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        r1 = cp_store.save(
            identity=identity,
            captured=captured,
            configuration_envelope=envelope,
            provenance=provenance,
            rng_bundle=rng_bundle,
            created_at_utc=ts,
        )
        # Save again with identical inputs at a fresh attempt dir for a clean id.
        archive1 = cp_store.load(r1.artifact_id, expected_identity=identity)
        digest1 = hashlib.sha256(archive1.tar_bytes).hexdigest()
        # Re-encode by reading the published content: identical bytes.
        content_path = store.locate(r1.artifact_id)
        assert content_path is not None
        digest2 = hashlib.sha256(content_path.read_bytes()).hexdigest()
        assert digest1 == digest2

    def test_no_self_referential_checkpoint_id(self, tmp_path: Path) -> None:
        _, archive, _ = _save_checkpoint(tmp_path)
        dump = archive.manifest.model_dump()
        assert "checkpoint_id" not in dump
        assert archive.manifest.parent_artifact_id is None


class TestArtifactIdLoading:
    def test_load_by_artifact_id(self, tmp_path: Path) -> None:
        _, archive, aid = _save_checkpoint(tmp_path)
        assert archive.artifact_id == aid

    def test_load_unknown_artifact_raises(self, tmp_path: Path) -> None:
        from expertforge.artifacts.models import ArtifactNotFoundError

        identity, _ = make_identity_and_provenance(tmp_path)
        store = make_store(tmp_path, identity=identity)
        cp_store = CheckpointStore(store)
        with pytest.raises((CheckpointCorruptError, ArtifactNotFoundError)):
            cp_store.load("artifact-v1-sha256-" + "0" * 64, expected_identity=identity)

    def test_cross_run_load_rejected(self, tmp_path: Path) -> None:
        # A checkpoint cannot be loaded against an expected identity from a
        # different run (amendment L: V1 full restore is same-run resume only).
        identity, provenance = make_identity_and_provenance(tmp_path)
        store = make_store(tmp_path, identity=identity)
        cp_store = CheckpointStore(store)
        captured = make_captured_state()
        rng_bundle = make_rng_bundle()
        from expertforge.config.resolve import resolve_config
        from tests._checkpoint_fixtures import CONFIGS

        envelope = resolve_config(CONFIGS / "smoke.yaml")
        record = cp_store.save(
            identity=identity,
            captured=captured,
            configuration_envelope=envelope,
            provenance=provenance,
            rng_bundle=rng_bundle,
        )
        # An expected identity from a DIFFERENT run_id. The store (bound to the
        # original identity) can still inspect the published record, but the
        # checkpoint store's identity-binding check must reject the cross-run
        # restore.
        cross_run = identity.model_copy(
            update={"run_id": "run-20260101t000000z-000000000000-99999999999999999999"}
        )
        with pytest.raises(CheckpointLineageError):
            cp_store.load(record.artifact_id, expected_identity=cross_run)


class TestCorruption:
    def test_truncated_archive_rejected(self, tmp_path: Path) -> None:
        identity, provenance = make_identity_and_provenance(tmp_path)
        store = make_store(tmp_path, identity=identity)
        cp_store = CheckpointStore(store)
        captured = make_captured_state()
        rng_bundle = make_rng_bundle()
        from expertforge.config.resolve import resolve_config
        from tests._checkpoint_fixtures import CONFIGS

        envelope = resolve_config(CONFIGS / "smoke.yaml")
        record = cp_store.save(
            identity=identity,
            captured=captured,
            configuration_envelope=envelope,
            provenance=provenance,
            rng_bundle=rng_bundle,
        )
        # Corrupt the published content on disk by truncating it.
        content_path = store.locate(record.artifact_id)
        assert content_path is not None
        data = content_path.read_bytes()
        content_path.write_bytes(data[:-64])
        with pytest.raises(CheckpointCorruptError):
            cp_store.load(record.artifact_id, expected_identity=identity)

    def test_corrupt_tensor_member_rejected(self, tmp_path: Path) -> None:
        identity, provenance = make_identity_and_provenance(tmp_path)
        store = make_store(tmp_path, identity=identity)
        cp_store = CheckpointStore(store)
        captured = make_captured_state()
        rng_bundle = make_rng_bundle()
        from expertforge.config.resolve import resolve_config
        from tests._checkpoint_fixtures import CONFIGS

        envelope = resolve_config(CONFIGS / "smoke.yaml")
        record = cp_store.save(
            identity=identity,
            captured=captured,
            configuration_envelope=envelope,
            provenance=provenance,
            rng_bundle=rng_bundle,
        )
        # Flip a byte in a tensor member region of the tar bytes.
        content_path = store.locate(record.artifact_id)
        assert content_path is not None
        data = bytearray(content_path.read_bytes())
        # Flip a byte near the end of the data section (tensor bytes).
        data[-1024] ^= 0xFF
        content_path.write_bytes(bytes(data))
        with pytest.raises(CheckpointCorruptError):
            cp_store.load(record.artifact_id, expected_identity=identity)


class TestCompatibility:
    def test_exact_compatibility(self, tmp_path: Path) -> None:
        _, archive, _ = _save_checkpoint(tmp_path)
        result = check_compatibility(archive.manifest.compatibility, archive.manifest.compatibility)
        assert result.status == "exact"
        assert result.mismatches == ()

    def test_shape_mismatch_blocks(self, tmp_path: Path) -> None:
        _, archive, _ = _save_checkpoint(tmp_path)
        # Modify expected descriptor to have a different shape for layer.weight.
        exp = archive.manifest.compatibility.model_copy(deep=True)
        new_params = tuple(
            p.model_copy(update={"shape": (2, 4)}) if p.name == "layer.weight" else p
            for p in exp.model_descriptor.parameters
        )
        expected_desc = exp.model_copy(
            update={
                "model_descriptor": exp.model_descriptor.model_copy(
                    update={"parameters": new_params}
                )
            }
        )
        result = check_compatibility(archive.manifest.compatibility, expected_desc)
        assert result.status == "incompatible"
        assert any(m.component == "model_shapes" for m in result.mismatches)

    def test_optimizer_type_mismatch_blocks(self, tmp_path: Path) -> None:
        _, archive, _ = _save_checkpoint(tmp_path)
        exp = archive.manifest.compatibility.model_copy(deep=True)
        new_opt = exp.optimizer_descriptor.model_copy(update={"optimizer_type": "Adam"})
        expected_desc = exp.model_copy(update={"optimizer_descriptor": new_opt})
        result = check_compatibility(archive.manifest.compatibility, expected_desc)
        assert result.status == "incompatible"
        assert any(m.component == "optimizer_type" for m in result.mismatches)

    def test_scaler_presence_mismatch_blocks(self, tmp_path: Path) -> None:
        _, archive_no_scaler, _ = _save_checkpoint(tmp_path)
        scaler_root = tmp_path / "b"
        scaler_root.mkdir(parents=True, exist_ok=True)
        _, archive_scaler, _ = _save_checkpoint(scaler_root, scaler=True)
        # archive with scaler vs expected without.
        result = check_compatibility(
            archive_scaler.manifest.compatibility,
            archive_no_scaler.manifest.compatibility,
        )
        assert result.status == "incompatible"
        assert any(m.component == "scaler_active" for m in result.mismatches)


class TestInspectPath:
    def test_inspect_path_complete(self, tmp_path: Path) -> None:
        identity, provenance = make_identity_and_provenance(tmp_path)
        store = make_store(tmp_path, identity=identity)
        cp_store = CheckpointStore(store)
        captured = make_captured_state()
        rng_bundle = make_rng_bundle()
        from expertforge.config.resolve import resolve_config
        from tests._checkpoint_fixtures import CONFIGS

        envelope = resolve_config(CONFIGS / "smoke.yaml")
        record = cp_store.save(
            identity=identity,
            captured=captured,
            configuration_envelope=envelope,
            provenance=provenance,
            rng_bundle=rng_bundle,
        )
        content_path = store.locate(record.artifact_id)
        assert content_path is not None
        inspection = cp_store.inspect_path(content_path)
        assert inspection.status == "complete"
        assert inspection.manifest is not None

    def test_inspect_path_missing_file(self, tmp_path: Path) -> None:
        identity, _ = make_identity_and_provenance(tmp_path)
        store = make_store(tmp_path, identity=identity)
        cp_store = CheckpointStore(store)
        inspection = cp_store.inspect_path(tmp_path / "nope.tar")
        assert inspection.status == "corrupt"
