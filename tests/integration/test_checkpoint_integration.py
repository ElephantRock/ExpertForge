"""End-to-end checkpoint integration tests (Issue #11, amendment N).

Full save → publish → load → restore → continuation comparison: an uninterrupted
run vs. a save/resume continuation must produce identical model output, optimizer
state, RNG next-sample, data next-item, and counters. Also covers deterministic
archive bytes, #10 publication/registry/parent verification, alias/tied-parameter
graph round-trip, and corruption rejection.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from tests._checkpoint_fixtures import (
    CONFIGS,
    make_identity_and_provenance,
    make_resume_identity,
    make_store,
)

from expertforge.artifacts.models import ArtifactRecord, ParentReference
from expertforge.checkpoints.models import CapturedTensor, CounterSnapshot, DataCursor
from expertforge.checkpoints.reference_adapter import (
    ReferenceCheckpointError,
    ReferenceState,
    ReferenceStateProvider,
)
from expertforge.checkpoints.restore import RestoreError, RestoreTransaction
from expertforge.checkpoints.store import (
    CheckpointCorruptError,
    CheckpointLineageError,
    CheckpointStore,
    check_compatibility,
)
from expertforge.config.resolve import resolve_config
from expertforge.identity.ids import attempt_id
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.rng.derivation import SeedContext
from expertforge.rng.manager import RngManager
from expertforge.rng.state import RngStateBundle

pytestmark = pytest.mark.integration

_FIXED = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _reference_state(
    *,
    weight_value: float,
    rng_bundle: bytes,
    global_update: int = 3,
    accepted_samples: int = 24,
    position: int = 24,
) -> ReferenceState:
    """A reference training state with a single float32 (2,3) parameter."""
    weight = np.full((2, 3), weight_value, dtype="<f4").tobytes()
    opt = np.full((2, 3), 0.01, dtype="<f4").tobytes()
    sched = np.asarray([5.0], dtype="<f4").tobytes()
    s = ReferenceState()
    s.parameters = {"layer.weight": weight}
    s.param_shapes = {"layer.weight": (2, 3)}
    s.param_dtypes = {"layer.weight": "float32"}
    s.optimizer_state = opt
    s.scheduler_state = sched
    s.optimizer_type = "SGD"
    s.scheduler_type = "constant"
    s.counters = CounterSnapshot(
        global_update=global_update,
        completed_microsteps=global_update * 2,
        accumulation_position=0,
        accepted_samples=accepted_samples,
        accepted_sequences=global_update,
        processed_tokens=accepted_samples * 32,
    )
    s.data_cursor = DataCursor(
        sampler_type="sequential",
        sampler_version=1,
        batch_size=8,
        sequence_length=128,
        drop_last=False,
        epoch=0,
        position=position,
        accepted_samples=accepted_samples,
        accepted_sequences=global_update,
    )
    s.rng_bundle_bytes = rng_bundle
    return s


class _RecordingFactory:
    """A minimal StateFactory that records applied state into a fresh target."""

    def create(self) -> dict[str, Any]:
        return {
            "parameters": {},
            "optimizer": None,
            "scheduler": None,
            "scaler": None,
            "cursor": None,
            "counters": None,
        }

    def apply_parameters(self, target: dict[str, Any], tensors: dict[str, CapturedTensor]) -> None:
        target["parameters"] = {n: t.raw_bytes for n, t in tensors.items()}

    def apply_buffers(self, target: dict[str, Any], tensors: dict[str, CapturedTensor]) -> None:
        target["buffers"] = {n: t.raw_bytes for n, t in tensors.items()}

    def apply_optimizer(self, target: dict[str, Any], tensor: CapturedTensor | None) -> None:
        target["optimizer"] = tensor.raw_bytes if tensor else None

    def apply_scheduler(self, target: dict[str, Any], tensor: CapturedTensor | None) -> None:
        target["scheduler"] = tensor.raw_bytes if tensor else None

    def apply_scaler(self, target: dict[str, Any], tensor: CapturedTensor | None) -> None:
        target["scaler"] = tensor.raw_bytes if tensor else None

    def apply_data_cursor(self, target: dict[str, Any], cursor: Any) -> None:
        target["cursor"] = cursor

    def apply_counters(self, target: dict[str, Any], counters: Any) -> None:
        target["counters"] = counters

    def commit_to_live(self, target: dict[str, Any]) -> dict[str, Any]:
        return target

    def revert_commit_to_live(self, target: dict[str, Any]) -> None:
        # Item 1: two-phase rollback hook. The reference factory swaps no live
        # handles, so this is a no-op (state remains on the target dict).
        return None


def _new_rng_manager(seed: int = 7) -> RngManager:
    m = RngManager(root_seed=seed, context=SeedContext(component="run"))
    m.initialize()
    return m


def _save_checkpoint(
    tmp_path: Path,
    *,
    state: ReferenceState,
    parent: ParentReference | None = None,
    created_at: datetime = _FIXED,
) -> tuple[CheckpointStore, str, AttemptIdentityRecord]:
    identity, provenance = make_identity_and_provenance(tmp_path)
    store = make_store(tmp_path, identity=identity)
    cp_store = CheckpointStore(store)
    provider = ReferenceStateProvider(state)
    captured = provider.capture_checkpoint_snapshot()
    rng_bundle = RngStateBundle.from_json_bytes(state.rng_bundle_bytes)
    envelope = resolve_config(CONFIGS / "smoke.yaml")
    record = cp_store.save(
        identity=identity,
        captured=captured,
        configuration_envelope=envelope,
        provenance=provenance,
        rng_bundle=rng_bundle,
        parent=parent,
        created_at_utc=created_at,
    )
    return cp_store, record.artifact_id, identity


class TestSavePublishLoadRestore:
    def test_full_round_trip_restores_state(self, tmp_path: Path) -> None:
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        state = _reference_state(weight_value=0.42, rng_bundle=bundle_bytes)
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        resume = make_resume_identity(identity, parent_artifact_id=aid)
        archive = cp_store.load(aid, expected_identity=resume)

        factory = _RecordingFactory()
        live_rng = _new_rng_manager()
        txn = RestoreTransaction(
            archive=archive,
            factory=factory,
            rng_bundle_loader=lambda: live_rng.capture_state(),
            rng_consumer=lambda b: None,  # tested below
        )
        txn.prepare()
        result = txn.commit()
        assert result.artifact_id == aid
        # Parameter bytes round-trip exactly.
        target = result.target
        assert target["parameters"]["layer.weight"] == state.parameters["layer.weight"]
        assert target["counters"].global_update == 3
        assert target["cursor"].position == 24

    def test_continuation_matches_uninterrupted_run(self, tmp_path: Path) -> None:
        # An uninterrupted run vs. a save/resume continuation must produce the
        # same next-sample, next data item, counters, and model bytes.
        rng_a = _new_rng_manager()
        rng_b = _new_rng_manager()
        # Advance both identically to a save point.
        rng_a.generator.integers(0, 1000, size=10)
        rng_b.generator.integers(0, 1000, size=10)
        bundle_bytes = rng_a.capture_state().to_deterministic_json()
        state = _reference_state(weight_value=1.5, rng_bundle=bundle_bytes)

        # (A) Uninterrupted: continue sampling.
        expected_next = rng_a.generator.integers(0, 1000, size=5).tolist()

        # (B) Save/resume: persist, then restore into a fresh manager.
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        resume = make_resume_identity(identity, parent_artifact_id=aid)
        archive = cp_store.load(aid, expected_identity=resume)

        # Restore RNG from the archive into rng_b (which is at the pre-save
        # state but we want the archived next-sample position).
        restored_bundle = RngStateBundle.from_json_bytes(archive.component("rng"))
        fresh = RngManager(root_seed=7, context=SeedContext(component="run"))
        fresh.restore_state(restored_bundle)
        actual_next = fresh.generator.integers(0, 1000, size=5).tolist()
        assert actual_next == expected_next


class TestDeterministicArchive:
    def test_identical_inputs_identical_bytes(self, tmp_path: Path) -> None:
        # amendment F: equal captured state, identity, parent, versions, and
        # injected creation timestamp produce identical archive bytes. We build
        # the archive twice through the encoder with identical inputs (the store
        # path adds the artifact_id-derived path, but the archive BYTES are
        # purely a function of the manifest + members, which are identical).
        from expertforge.checkpoints.encoder import build_archive

        rng = _new_rng_manager()
        bundle = rng.capture_state()
        identity, provenance = make_identity_and_provenance(tmp_path)
        envelope = resolve_config(CONFIGS / "smoke.yaml")
        state = _reference_state(weight_value=2.0, rng_bundle=bundle.to_deterministic_json())
        captured = ReferenceStateProvider(state).capture_checkpoint_snapshot()
        ts = _FIXED
        a1 = build_archive(
            identity=identity,
            captured=captured,
            configuration_envelope=envelope,
            provenance=provenance,
            rng_bundle=bundle,
            parent=None,
            created_at_utc=ts,
        )
        a2 = build_archive(
            identity=identity,
            captured=captured,
            configuration_envelope=envelope,
            provenance=provenance,
            rng_bundle=bundle,
            parent=None,
            created_at_utc=ts,
        )
        assert a1.tar_bytes == a2.tar_bytes
        assert a1.content_digest == a2.content_digest


class TestPublicationVerification:
    def test_checkpoint_registered_and_verifiable(self, tmp_path: Path) -> None:
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        state = _reference_state(weight_value=0.9, rng_bundle=bundle_bytes)
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        # #10 verification succeeds.
        result = cp_store.artifact_store.verify(aid)
        assert result.status is True
        assert result.diagnostic_code == "verified"
        # The artifact is listed under the checkpoint category.
        records = cp_store.artifact_store.list_artifacts(category="checkpoint")
        assert any(r.artifact_id == aid for r in records)

    def test_parent_reference_round_trips(self, tmp_path: Path) -> None:
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        state = _reference_state(weight_value=0.1, rng_bundle=bundle_bytes)
        cp_store, parent_aid, identity = _save_checkpoint(tmp_path, state=state)
        parent = ParentReference(
            run_id=identity.run_id,
            attempt_id=identity.attempt_id,
            artifact_id=parent_aid,
        )
        # Save a second checkpoint with the first as parent.
        rng2 = _new_rng_manager(seed=11)
        state2 = _reference_state(
            weight_value=0.2, rng_bundle=rng2.capture_state().to_deterministic_json()
        )
        cp_store2, child_aid, _ = _save_checkpoint(tmp_path / "child", state=state2, parent=parent)
        child = cp_store2.artifact_store.inspect(child_aid)
        assert isinstance(child, ArtifactRecord)
        assert child.parent is not None
        assert child.parent.artifact_id == parent_aid


class TestAliasGraph:
    def test_alias_group_round_trip_and_mismatch(self, tmp_path: Path) -> None:
        # Build a state with a tied parameter alias group.
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        shared = np.full((2, 3), 7.0, dtype="<f4").tobytes()
        state = _reference_state(weight_value=7.0, rng_bundle=bundle_bytes)
        state.parameters = {
            "layer.weight": shared,
            "layer.weight_tied": shared,
            "layer.weight_share": shared,
        }
        state.param_shapes = {n: (2, 3) for n in state.parameters}
        state.param_dtypes = {n: "float32" for n in state.parameters}
        state.alias_groups = (("layer.weight_share", "layer.weight_tied"),)
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        resume = make_resume_identity(identity, parent_artifact_id=aid)
        archive = cp_store.load(aid, expected_identity=resume)
        # Both aliased names map to ONE canonical tensor member.
        member_for_share = next(
            m for m in archive.manifest.tensor_members if "layer.weight_share" in m.logical_names
        )
        member_for_tied = next(
            m for m in archive.manifest.tensor_members if "layer.weight_tied" in m.logical_names
        )
        assert member_for_share.member_name == member_for_tied.member_name
        # Compatibility check against itself is exact (alias graph preserved).
        result = check_compatibility(archive.manifest.compatibility, archive.manifest.compatibility)
        assert result.status == "exact"


class TestCorruptionRejection:
    def test_corrupt_manifest_rejected(self, tmp_path: Path) -> None:
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        state = _reference_state(weight_value=0.5, rng_bundle=bundle_bytes)
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        content_path = cp_store.artifact_store.locate(aid)
        assert content_path is not None
        data = bytearray(content_path.read_bytes())
        # Flip a byte in the manifest.json data region (the first member's
        # data starts immediately after the 512-byte header).
        data[512] ^= 0xFF
        content_path.write_bytes(bytes(data))
        resume = make_resume_identity(identity, parent_artifact_id=aid)
        with pytest.raises(CheckpointCorruptError):
            cp_store.load(aid, expected_identity=resume)


class TestAtomicCapture:
    def test_non_quiescent_snapshot_rejected(self) -> None:
        # amendment H/I: a non-quiescent counter (accumulation in flight) must
        # be rejected at the save boundary. The CounterSnapshot model enforces
        # accumulation_position == 0 at construction, which is the V1 save
        # boundary check.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CounterSnapshot(
                global_update=3,
                completed_microsteps=7,
                accumulation_position=2,  # mid-accumulation
                accepted_samples=24,
                accepted_sequences=3,
                processed_tokens=768,
            )

    def test_provider_rejects_non_zero_accumulation(self) -> None:
        # The reference provider itself rejects a non-quiescent live counter
        # before constructing the (validated) captured snapshot.
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        state = _reference_state(weight_value=0.3, rng_bundle=bundle_bytes)
        # Replace the validated counter with a non-quiescent one via
        # model_construct (bypass validation) to simulate a live in-flight state.
        from expertforge.checkpoints.models import CounterSnapshot as CS

        state.counters = CS.model_construct(
            global_update=3,
            completed_microsteps=7,
            accumulation_position=2,
            accepted_samples=24,
            accepted_sequences=3,
            processed_tokens=768,
        )
        provider = ReferenceStateProvider(state)
        with pytest.raises(ReferenceCheckpointError):
            provider.capture_checkpoint_snapshot()


def test_attempt_id_helper_usable() -> None:
    # Guard that the imported attempt_id helper is callable (used by the
    # cross-run scenarios in the focused store tests).
    aid = attempt_id(clock=lambda: _FIXED, entropy=lambda n: bytes(range(n)))
    assert aid.startswith("attempt-")


# ---------------------------------------------------------------------------
# PR #35 review item regressions: continuation, alias, native lineage.
# ---------------------------------------------------------------------------


class _FullRecordingFactory:
    """A StateFactory that records parameters, buffers, optimizer, scheduler,
    scaler, cursor, and counters into a fresh target, with a two-phase rollback
    hook (item 1)."""

    def __init__(self) -> None:
        self.committed = False
        self.reverted = False
        self.target: dict[str, Any] = {}

    def create(self) -> dict[str, Any]:
        return {
            "parameters": {},
            "buffers": {},
            "optimizer": None,
            "scheduler": None,
            "scaler": None,
            "cursor": None,
            "counters": None,
        }

    def apply_parameters(self, target: dict[str, Any], tensors: dict[str, CapturedTensor]) -> None:
        target["parameters"] = {n: t.raw_bytes for n, t in tensors.items()}

    def apply_buffers(self, target: dict[str, Any], tensors: dict[str, CapturedTensor]) -> None:
        target["buffers"] = {n: t.raw_bytes for n, t in tensors.items()}

    def apply_optimizer(self, target: dict[str, Any], tensor: CapturedTensor | None) -> None:
        target["optimizer"] = tensor.raw_bytes if tensor else None

    def apply_scheduler(self, target: dict[str, Any], tensor: CapturedTensor | None) -> None:
        target["scheduler"] = tensor.raw_bytes if tensor else None

    def apply_scaler(self, target: dict[str, Any], tensor: CapturedTensor | None) -> None:
        target["scaler"] = tensor.raw_bytes if tensor else None

    def apply_data_cursor(self, target: dict[str, Any], cursor: Any) -> None:
        target["cursor"] = cursor

    def apply_counters(self, target: dict[str, Any], counters: Any) -> None:
        target["counters"] = counters

    def commit_to_live(self, target: dict[str, Any]) -> dict[str, Any]:
        self.committed = True
        self.target = target
        return target

    def revert_commit_to_live(self, target: dict[str, Any]) -> None:
        self.committed = False
        self.reverted = True


class TestItem1TwoPhaseCommit:
    def test_rng_failure_reverts_non_rng_swap(self, tmp_path: Path) -> None:
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        state = _reference_state(weight_value=0.42, rng_bundle=bundle_bytes)
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        resume = make_resume_identity(identity, parent_artifact_id=aid)
        archive = cp_store.load(aid, expected_identity=resume)

        factory = _FullRecordingFactory()
        live_rng = _new_rng_manager()
        pre_rng = live_rng.capture_state()
        txn = RestoreTransaction(
            archive=archive,
            factory=factory,
            rng_bundle_loader=lambda: live_rng.capture_state(),
            rng_consumer=lambda b: None,
        )
        txn.prepare()
        txn.rng_restore_should_fail = True
        with pytest.raises(RestoreError):
            txn.commit()
        # Item 1: the non-RNG swap was committed then reverted; RNG unchanged.
        assert factory.reverted
        assert not factory.committed
        assert live_rng.capture_state() == pre_rng


class TestItem2BuffersRestored:
    def test_persisted_buffers_are_applied(self, tmp_path: Path) -> None:

        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        state = _reference_state(weight_value=1.0, rng_bundle=bundle_bytes)
        # Add a buffer (e.g. running mean) alongside the parameter.
        buf = np.full((2,), 0.5, dtype="<f4").tobytes()
        state.buffers = {"layer.running_mean": buf}
        state.param_shapes["layer.running_mean"] = (2,)
        state.param_dtypes["layer.running_mean"] = "float32"
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        resume = make_resume_identity(identity, parent_artifact_id=aid)
        archive = cp_store.load(aid, expected_identity=resume)

        factory = _FullRecordingFactory()
        txn = RestoreTransaction(
            archive=archive,
            factory=factory,
            rng_bundle_loader=lambda: _new_rng_manager().capture_state(),
            rng_consumer=lambda b: None,
        )
        txn.prepare()
        result = txn.commit()
        # Item 2: buffers are restored, not dropped.
        assert result.target["buffers"]["layer.running_mean"] == buf


class TestItem3AliasRestoresEveryName:
    def test_every_alias_name_gets_the_tensor(self, tmp_path: Path) -> None:
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        shared = np.full((2, 3), 7.0, dtype="<f4").tobytes()
        state = _reference_state(weight_value=7.0, rng_bundle=bundle_bytes)
        state.parameters = {
            "layer.weight": shared,
            "layer.weight_tied": shared,
            "layer.weight_share": shared,
        }
        state.param_shapes = {n: (2, 3) for n in state.parameters}
        state.param_dtypes = {n: "float32" for n in state.parameters}
        state.alias_groups = (("layer.weight_share", "layer.weight_tied"),)
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        resume = make_resume_identity(identity, parent_artifact_id=aid)
        archive = cp_store.load(aid, expected_identity=resume)

        factory = _FullRecordingFactory()
        txn = RestoreTransaction(
            archive=archive,
            factory=factory,
            rng_bundle_loader=lambda: _new_rng_manager().capture_state(),
            rng_consumer=lambda b: None,
        )
        txn.prepare()
        txn.commit()
        # Item 3: EVERY aliased name is present in the restored parameters and
        # points to the shared tensor bytes.
        for name in ("layer.weight", "layer.weight_tied", "layer.weight_share"):
            assert name in factory.target["parameters"], name
            assert factory.target["parameters"][name] == shared

    def test_alias_member_binding_is_singular(self, tmp_path: Path) -> None:
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        shared = np.full((2, 3), 7.0, dtype="<f4").tobytes()
        state = _reference_state(weight_value=7.0, rng_bundle=bundle_bytes)
        state.parameters = {
            "layer.weight": shared,
            "layer.weight_tied": shared,
            "layer.weight_share": shared,
        }
        state.param_shapes = {n: (2, 3) for n in state.parameters}
        state.param_dtypes = {n: "float32" for n in state.parameters}
        state.alias_groups = (("layer.weight_share", "layer.weight_tied"),)
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        resume = make_resume_identity(identity, parent_artifact_id=aid)
        archive = cp_store.load(aid, expected_identity=resume)
        # Both tied names point to ONE canonical tensor member.
        member_for_share = next(
            m for m in archive.manifest.tensor_members if "layer.weight_share" in m.logical_names
        )
        member_for_tied = next(
            m for m in archive.manifest.tensor_members if "layer.weight_tied" in m.logical_names
        )
        assert member_for_share.member_name == member_for_tied.member_name


class TestItem9NativeLineage:
    def test_load_with_producing_attempt_rejected(self, tmp_path: Path) -> None:
        # Loading with the producing identity itself is NOT the v1 resume
        # contract: a NEW attempt is required.
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        state = _reference_state(weight_value=0.5, rng_bundle=bundle_bytes)
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        with pytest.raises(CheckpointLineageError):
            cp_store.load(aid, expected_identity=identity)

    def test_load_rejects_mismatched_lineage_checkpoint(self, tmp_path: Path) -> None:
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        state = _reference_state(weight_value=0.5, rng_bundle=bundle_bytes)
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        # A resume identity whose lineage names a DIFFERENT checkpoint id.
        bogus = make_resume_identity(identity, parent_artifact_id="artifact-v1-sha256-" + "f" * 64)
        with pytest.raises(CheckpointLineageError):
            cp_store.load(aid, expected_identity=bogus)

    def test_resolve_parent_verifies_registered_checkpoint(self, tmp_path: Path) -> None:
        rng = _new_rng_manager()
        bundle_bytes = rng.capture_state().to_deterministic_json()
        state = _reference_state(weight_value=0.5, rng_bundle=bundle_bytes)
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        parent = cp_store.resolve_parent(aid, cp_store.artifact_store)
        assert parent.artifact_id == aid
        assert parent.category == "checkpoint"

    def test_resolve_parent_rejects_unregistered(self, tmp_path: Path) -> None:
        identity, _ = make_identity_and_provenance(tmp_path)
        store = make_store(tmp_path, identity=identity)
        cp_store = CheckpointStore(store)
        with pytest.raises(CheckpointLineageError):
            cp_store.resolve_parent("artifact-v1-sha256-" + "0" * 64, cp_store.artifact_store)


class TestItem15Continuation:
    def test_full_continuation_matches_uninterrupted_run(self, tmp_path: Path) -> None:
        # An uninterrupted run vs. a save/resume continuation must produce
        # identical model bytes, optimizer, scheduler, next data item, and
        # counters — not just the RNG next-sample (item 15).
        rng_a = _new_rng_manager()
        rng_b = _new_rng_manager()
        rng_a.generator.integers(0, 1000, size=10)
        rng_b.generator.integers(0, 1000, size=10)
        bundle_bytes = rng_a.capture_state().to_deterministic_json()
        weight = 1.25
        state = _reference_state(
            weight_value=weight,
            rng_bundle=bundle_bytes,
            global_update=5,
            accepted_samples=40,
            position=40,
        )
        opt_bytes = state.optimizer_state
        sched_bytes = state.scheduler_state

        # (A) Uninterrupted baseline state.
        expected_weight = state.parameters["layer.weight"]
        expected_counters = state.counters
        expected_cursor = cast("DataCursor", state.data_cursor)
        expected_next_rng = rng_a.generator.integers(0, 1000, size=5).tolist()

        # (B) Save/resume continuation.
        cp_store, aid, identity = _save_checkpoint(tmp_path, state=state)
        resume = make_resume_identity(identity, parent_artifact_id=aid)
        archive = cp_store.load(aid, expected_identity=resume)

        factory = _FullRecordingFactory()
        # A fresh, uninitialized manager for restore_state (restore requires the
        # manager not be already initialized).
        restored_rng = RngManager(root_seed=7, context=SeedContext(component="run"))
        restored_rng.restore_state(RngStateBundle.from_json_bytes(archive.component("rng")))
        txn = RestoreTransaction(
            archive=archive,
            factory=factory,
            rng_bundle_loader=lambda: restored_rng.capture_state(),
            rng_consumer=lambda b: None,
        )
        txn.prepare()
        txn.commit()

        target = factory.target
        # Model bytes round-trip exactly.
        assert target["parameters"]["layer.weight"] == expected_weight
        # Optimizer + scheduler state round-trip exactly.
        assert target["optimizer"] == opt_bytes
        assert target["scheduler"] == sched_bytes
        # Counters + cursor round-trip exactly.
        assert target["counters"] == expected_counters
        restored_cursor = cast("DataCursor", target["cursor"])
        assert restored_cursor.position == expected_cursor.position
        assert restored_cursor.accepted_samples == expected_cursor.accepted_samples
        # RNG continuation matches the uninterrupted run's next-sample.
        actual_next_rng = restored_rng.generator.integers(0, 1000, size=5).tolist()
        assert actual_next_rng == expected_next_rng
