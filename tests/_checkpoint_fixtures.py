"""Shared construction helpers for Issue #11 checkpoint tests.

Pure helpers that build a valid :class:`AttemptIdentityRecord`, a typed
:class:`ProvenanceRecord` (against a temporary git repo), a resolved
configuration envelope, an :class:`ArtifactStore`, and a quiescent
:class:`CapturedCheckpointState`. Importing this module has no side effects.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from expertforge.artifacts import ArtifactStore
from expertforge.checkpoints.models import (
    CapturedCheckpointState,
    CapturedTensor,
    CompatibilityDescriptor,
    CounterSnapshot,
    DataCursor,
    DataDescriptor,
    DataIdentity,
    ModelDescriptor,
    ModelParameterDescriptor,
    OptimizerDescriptor,
    OptimizerParamGroup,
    OptimizerStateSlot,
    RngDescriptor,
    ScalerDescriptor,
    SchedulerDescriptor,
    TopologyDescriptor,
)
from expertforge.config.resolve import ResolutionEnvelope, resolve_config
from expertforge.identity.ids import attempt_id
from expertforge.identity.lineage import ResumeLineage
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.orchestrate import prepare_run
from expertforge.provenance.record import ProvenanceRecord
from expertforge.rng.derivation import SeedContext
from expertforge.rng.manager import RngManager
from expertforge.rng.state import RngStateBundle

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FIXED_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def init_repo(repo: Path) -> None:
    """Initialize a clean git repo with one committed file."""
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def resolve_envelope() -> ResolutionEnvelope:
    return resolve_config(CONFIGS / "smoke.yaml")


def make_identity_and_provenance(
    tmp_path: Path,
    *,
    config: ResolutionEnvelope | None = None,
) -> tuple[AttemptIdentityRecord, ProvenanceRecord]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    init_repo(repo)
    envelope = config or resolve_envelope()
    identity, provenance, _ = prepare_run(
        artifact_root=tmp_path / "runs",
        config_envelope=envelope,
        repo=repo,
        clock=lambda: FIXED_TS,
        entropy=lambda n: bytes(n),
    )
    return identity, provenance


def make_store(tmp_path: Path, *, identity: AttemptIdentityRecord) -> ArtifactStore:
    return ArtifactStore(artifact_root=tmp_path / "runs", identity=identity)


def make_rng_bundle(*, root_seed: int = 7) -> RngStateBundle:
    manager = RngManager(
        root_seed=root_seed,
        context=SeedContext(component="run"),
    )
    manager.initialize()
    return manager.capture_state()


def _float32_bytes(values: list[float]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def make_captured_state(
    *,
    scaler: bool = False,
    alias: bool = False,
    global_update: int = 3,
    completed_microsteps: int = 6,
    accepted_samples: int = 24,
    accepted_sequences: int = 6,
    processed_tokens: int = 768,
    position: int = 24,
) -> CapturedCheckpointState:
    """A valid quiescent checkpoint snapshot for tests.

    One parameter ``layer.weight`` (2,3) float32, one optimizer momentum slot
    tensor, a scheduler state tensor, and an optional scaler tensor.
    """
    rng_bytes = make_rng_bundle().to_deterministic_json()
    weight = _float32_bytes([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parameters: tuple[CapturedTensor, ...] = (
        CapturedTensor(
            logical_name="layer.weight", dtype="float32", shape=(2, 3), raw_bytes=weight
        ),
    )
    if alias:
        bias_bytes = _float32_bytes([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        parameters = (
            CapturedTensor(
                logical_name="layer.weight", dtype="float32", shape=(2, 3), raw_bytes=weight
            ),
            CapturedTensor(
                logical_name="layer.weight_tied",
                dtype="float32",
                shape=(2, 3),
                raw_bytes=bias_bytes,
            ),
        )
    optimizer_state_bytes = _float32_bytes([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    optimizer_tensor = CapturedTensor(
        logical_name="optimizer.state.momentum",
        dtype="float32",
        shape=(2, 3),
        raw_bytes=optimizer_state_bytes,
    )
    scheduler_bytes = _float32_bytes([5.0])
    scheduler_tensor = CapturedTensor(
        logical_name="scheduler.state", dtype="float32", shape=(1,), raw_bytes=scheduler_bytes
    )
    scaler_tensor = None
    scaler_descriptor = None
    if scaler:
        scaler_bytes = _float32_bytes([1.0])
        scaler_tensor = CapturedTensor(
            logical_name="scaler.state", dtype="float32", shape=(1,), raw_bytes=scaler_bytes
        )
        scaler_descriptor = ScalerDescriptor(scaler_type="grad_scaler", state_shape=(1,))

    param_desc = tuple(
        ModelParameterDescriptor(name=t.logical_name, shape=tuple(t.shape), dtype=t.dtype)
        for t in parameters
    )
    alias_groups: tuple[tuple[str, ...], ...] = ()
    if alias:
        alias_groups = (("layer.weight_share", "layer.weight_tied"),)
        # Use a single canonical tensor shared by both names.
        shared_bytes = _float32_bytes([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        parameters = (
            CapturedTensor(
                logical_name="layer.weight", dtype="float32", shape=(2, 3), raw_bytes=weight
            ),
            CapturedTensor(
                logical_name="layer.weight_share",
                dtype="float32",
                shape=(2, 3),
                raw_bytes=shared_bytes,
            ),
            CapturedTensor(
                logical_name="layer.weight_tied",
                dtype="float32",
                shape=(2, 3),
                raw_bytes=shared_bytes,
            ),
        )
        param_desc = tuple(
            ModelParameterDescriptor(name=t.logical_name, shape=tuple(t.shape), dtype=t.dtype)
            for t in parameters
        )

    optimizer_desc = OptimizerDescriptor(
        optimizer_type="SGD",
        param_groups=(
            OptimizerParamGroup(
                group_index=0,
                param_names=("layer.weight",),
                options=(),
            ),
        ),
        state_slots=(
            OptimizerStateSlot(
                group_index=0,
                param_name="layer.weight",
                slot_name="momentum",
                shape=(2, 3),
                dtype="float32",
            ),
        ),
    )

    data_desc = DataDescriptor(
        identity=DataIdentity(
            dataset_digest="a" * 64,
            split="train",
            length=1000,
            preprocessing_identity="prep.v1",
            tokenizer_identity="tok.v1",
            packing_policy="no_packing",
            sequence_policy="fixed_128",
            shard_selection="single",
            data_config_digest="b" * 64,
        ),
        sampler_type="sequential",
        sampler_version=1,
        batch_size=8,
        sequence_length=128,
        drop_last=False,
    )

    return CapturedCheckpointState(
        parameters=parameters,
        buffers=(),
        alias_groups=alias_groups,
        optimizer=optimizer_tensor,
        scheduler=scheduler_tensor,
        scaler=scaler_tensor,
        rng_bundle_bytes=rng_bytes,
        data_cursor=DataCursor(
            sampler_type="sequential",
            sampler_version=1,
            batch_size=8,
            sequence_length=128,
            drop_last=False,
            epoch=0,
            position=position,
            accepted_samples=accepted_samples,
            accepted_sequences=accepted_sequences,
        ),
        counters=CounterSnapshot(
            global_update=global_update,
            completed_microsteps=completed_microsteps,
            accumulation_position=0,
            accepted_samples=accepted_samples,
            accepted_sequences=accepted_sequences,
            processed_tokens=processed_tokens,
        ),
        model_descriptor=ModelDescriptor(parameters=param_desc, buffers=()),
        optimizer_descriptor=optimizer_desc,
        scheduler_descriptor=SchedulerDescriptor(scheduler_type="constant", state_shape=(1,)),
        scaler_descriptor=scaler_descriptor,
        rng_descriptor=RngDescriptor(adapter_set=(), framework_versions=()),
        data_descriptor=data_desc,
        topology_descriptor=TopologyDescriptor(world_size=1, rank_assignment=(0,)),
    )


def expected_descriptor(captured: CapturedCheckpointState) -> Any:
    """Build the CompatibilityDescriptor matching a captured state."""
    return CompatibilityDescriptor(
        specification_fingerprint="spec-v1-sha256-" + "0" * 64,
        model_descriptor=captured.model_descriptor,
        optimizer_descriptor=captured.optimizer_descriptor,
        scheduler_descriptor=captured.scheduler_descriptor,
        scaler_descriptor=captured.scaler_descriptor,
        rng_descriptor=captured.rng_descriptor,
        data_descriptor=captured.data_descriptor,
        topology_descriptor=captured.topology_descriptor,
    )


def make_resume_identity(
    source: AttemptIdentityRecord,
    *,
    parent_artifact_id: str,
    clock: Any = None,
    entropy: Any = None,
) -> AttemptIdentityRecord:
    """Build a NEW attempt identity that resumes from ``source`` (item 9).

    The resume identity shares the source's run and specification fingerprint
    but has a distinct attempt_id and a :class:`ResumeLineage` naming the source
    checkpoint's run/attempt/artifact_id. This is the v1 native-resume contract
    that :meth:`CheckpointStore.load` enforces.
    """
    clk = clock or (lambda: FIXED_TS)
    ent = entropy or (lambda n: bytes(range(60, 60 + n)))
    new_attempt = attempt_id(clock=clk, entropy=ent)
    lineage = ResumeLineage(
        parent_run_id=source.run_id,
        parent_attempt_id=source.attempt_id,
        parent_checkpoint_id=parent_artifact_id,
    )
    return source.model_copy(update={"attempt_id": new_attempt, "lineage": lineage})
