"""Regression tests for the restore transaction (Issue #11, K).

Covers transaction semantics, RNG rollback, and failure injection at every
restore/commit stage. These are integration tests (they build a real published
checkpoint through the store to obtain a :class:`CheckpointArchive`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from expertforge.checkpoints.models import CapturedTensor
from expertforge.checkpoints.restore import (
    RestoreError,
    RestoreStage,
    RestoreTransaction,
)
from expertforge.checkpoints.store import CheckpointArchive, CheckpointStore
from expertforge.rng.derivation import SeedContext
from expertforge.rng.manager import RngManager
from expertforge.rng.state import RngStateBundle
from tests._checkpoint_fixtures import (
    CONFIGS,
    make_captured_state,
    make_identity_and_provenance,
    make_store,
)

pytestmark = pytest.mark.integration


def _save_and_load(
    tmp_path: Path, *, scaler: bool = False
) -> tuple[CheckpointStore, CheckpointArchive, RngStateBundle, RngManager]:
    identity, provenance = make_identity_and_provenance(tmp_path)
    store = make_store(tmp_path, identity=identity)
    cp_store = CheckpointStore(store)
    captured = make_captured_state(scaler=scaler)
    # Use a dedicated RNG manager so we can independently restore it.
    rng_manager = RngManager(root_seed=7, context=SeedContext(component="run"))
    rng_manager.initialize()
    # Advance the RNG a little so the restored next-sample differs from initial.
    rng_manager.generator.integers(0, 1000, size=10)
    bundle = rng_manager.capture_state()
    from expertforge.config.resolve import resolve_config

    envelope = resolve_config(CONFIGS / "smoke.yaml")
    record = cp_store.save(
        identity=identity,
        captured=captured,
        configuration_envelope=envelope,
        provenance=provenance,
        rng_bundle=bundle,
    )
    archive = cp_store.load(record.artifact_id, expected_identity=identity)
    return cp_store, archive, bundle, rng_manager


class _RecordingFactory:
    """A minimal StateFactory recording applied state and exposing the target."""

    def __init__(self) -> None:
        self.target: dict[str, Any] = {
            "parameters": {},
            "optimizer": None,
            "scheduler": None,
            "scaler": None,
            "cursor": None,
            "counters": None,
        }
        self.committed = False

    def create(self) -> dict[str, Any]:
        # Return a fresh target dict (isolated from any live state).
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
        self.committed = True
        self.target = target
        return target


def _make_txn(
    archive: CheckpointArchive,
    rng_manager: RngManager,
) -> tuple[RestoreTransaction, _RecordingFactory, dict[str, RngManager]]:
    factory = _RecordingFactory()

    # The live RNG state is held in a single manager. Restoration models a
    # process restart: build a fresh manager, restore the bundle into it, and
    # swap it in as the new live manager. The "previous bundle" rollback
    # rebuilds from the previously-captured bundle.
    holder: dict[str, RngManager] = {"live": rng_manager}

    def load_bundle() -> RngStateBundle:
        return holder["live"].capture_state()

    def consume(bundle: RngStateBundle) -> None:
        fresh = RngManager(
            root_seed=bundle.root_seed,
            context=bundle.context,
            determinism_mode=bundle.determinism_mode,
            unsupported_determinism=bundle.unsupported_determinism,
        )
        fresh.restore_state(bundle)
        holder["live"] = fresh

    txn = RestoreTransaction(
        archive=archive,
        factory=factory,
        rng_bundle_loader=load_bundle,
        rng_consumer=consume,
    )
    return txn, factory, holder


class TestRestoreTransaction:
    def test_commit_applies_all_state(self, tmp_path: Path) -> None:
        _, archive, _, rng_manager = _save_and_load(tmp_path)
        txn, factory, _ = _make_txn(archive, rng_manager)
        txn.prepare()
        result = txn.commit()
        assert factory.committed
        assert result.artifact_id == archive.artifact_id
        assert "layer.weight" in factory.target["parameters"]
        assert factory.target["counters"] is not None
        assert factory.target["cursor"] is not None

    def test_abort_does_not_commit(self, tmp_path: Path) -> None:
        _, archive, _, rng_manager = _save_and_load(tmp_path)
        txn, factory, _ = _make_txn(archive, rng_manager)
        txn.prepare()
        txn.abort()
        assert not factory.committed
        # Commit after abort is rejected.
        with pytest.raises(RestoreError):
            txn.commit()

    def test_commit_after_commit_rejected(self, tmp_path: Path) -> None:
        _, archive, _, rng_manager = _save_and_load(tmp_path)
        txn, _, _ = _make_txn(archive, rng_manager)
        txn.prepare()
        txn.commit()
        with pytest.raises(RestoreError):
            txn.commit()


class TestRngRollback:
    def test_rng_restored_last(self, tmp_path: Path) -> None:
        # The archived RNG bundle, once restored, must produce the same next
        # samples as the saved bundle. RNG is restored last so loading does not
        # alter the resumed stream.
        _, archive, saved_bundle, rng_manager = _save_and_load(tmp_path)
        # Advance the live manager AFTER save so the un-restored continuation
        # would produce different next-samples than the saved bundle.
        rng_manager.generator.integers(0, 1000, size=32)
        live_before = rng_manager.generator.integers(0, 1000, size=5).tolist()
        txn, _, holder = _make_txn(archive, rng_manager)
        txn.prepare()
        txn.commit()
        # After commit, the live manager's state equals the archived bundle.
        # The next sample must match what the saved bundle would produce.
        fresh = RngManager(root_seed=7, context=SeedContext(component="run"))
        fresh.restore_state(saved_bundle)
        expected = fresh.generator.integers(0, 1000, size=5).tolist()
        actual = holder["live"].generator.integers(0, 1000, size=5).tolist()
        assert actual == expected
        # And differ from the un-restored live continuation.
        assert live_before != actual

    def test_rng_restore_failure_rolls_back_previous(self, tmp_path: Path) -> None:
        _, archive, _, rng_manager = _save_and_load(tmp_path)
        # Snapshot the live manager's pre-commit next-sample.
        pre_commit_bundle = rng_manager.capture_state()
        txn, _, holder = _make_txn(archive, rng_manager)
        txn.prepare()
        # Inject an RNG restore failure. The transaction must restore the
        # previous (pre-commit) RNG bundle and raise.
        txn.rng_restore_should_fail = True
        with pytest.raises(RestoreError):
            txn.commit()
        # The live RNG state must equal the pre-commit bundle.
        restored_bundle = holder["live"].capture_state()
        assert restored_bundle == pre_commit_bundle


class TestFailureInjection:
    @pytest.mark.parametrize(
        "stage",
        [
            "factory",
            "parameters",
            "buffers",
            "optimizer",
            "scheduler",
            "scaler",
            "data_cursor",
            "counters",
            "rng_prevalidate",
            "rng_capture_previous",
            "commit",
        ],
    )
    def test_failure_after_stage_leaves_live_untouched(
        self, tmp_path: Path, stage: RestoreStage
    ) -> None:
        _, archive, _, rng_manager = _save_and_load(tmp_path, scaler=True)
        pre = rng_manager.capture_state()
        txn, factory, holder = _make_txn(archive, rng_manager)
        txn.fail_after = stage
        with pytest.raises(RestoreError):
            txn.prepare() if stage in (
                "factory",
                "parameters",
                "buffers",
                "optimizer",
                "scheduler",
                "scaler",
                "data_cursor",
                "counters",
            ) else txn.commit()
        # Existing live state untouched: RNG unchanged, factory not committed
        # (unless the failure was injected exactly at "commit", in which case
        # the swap was prepared but RNG restored — RNG still equals pre).
        if stage != "commit":
            assert not factory.committed
        assert holder["live"].capture_state() == pre

    def test_prepare_failure_does_not_call_factory_commit(self, tmp_path: Path) -> None:
        _, archive, _, rng_manager = _save_and_load(tmp_path)
        txn, factory, _ = _make_txn(archive, rng_manager)
        txn.fail_after = "parameters"
        with pytest.raises(RestoreError):
            txn.prepare()
        assert not factory.committed
