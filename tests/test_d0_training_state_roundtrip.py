"""End-to-end D0.3 state round-trip: capture → save → load → restore → continue.

Proves that a D0 training state (D0.2 torch model + frozen D0Optimizer +
frozen D0Schedule) survives the full checkpoint path bit-exactly: an
uninterrupted run and a save/restore continuation produce identical
parameters after one more identical update — the exact-resume property the
smoke gate established on the NumPy fixture, now over the real torch
training state.

Requires the ``d0-model`` extra; skips cleanly when torch is absent. Marked
plain (not ``integration``) deliberately: the CI CPU-integration job does not
install the ``d0-model`` extra, so gating here would silently skip; the
quality job has torch and runs this file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from expertforge.artifacts.models import ParentReference  # noqa: F401,E402
from expertforge.checkpoints.encoder import (  # noqa: E402
    safe_value_from_native,
    safe_value_to_native,
)
from expertforge.checkpoints.models import (  # noqa: E402
    CapturedCheckpointState,
    CapturedTensor,
    CompatibilityDescriptor,
    CounterSnapshot,
    DataCursor,
    DataDescriptor,
    DataIdentity,
    ModelDescriptor,
    ModelParameterDescriptor,
    RngDescriptor,
    SchedulerDescriptor,
    TopologyDescriptor,
)
from expertforge.checkpoints.restore import RestoreTransaction  # noqa: E402
from expertforge.checkpoints.store import CheckpointStore  # noqa: E402
from expertforge.config.d0_models import D0OptimizerConfig  # noqa: E402
from expertforge.d0.model.config import D0ModelConfig  # noqa: E402
from expertforge.d0.model.initialization import initialize_model  # noqa: E402
from expertforge.d0.model.transformer import D0Model  # noqa: E402
from expertforge.d0.training.optimizer import D0Optimizer  # noqa: E402
from expertforge.d0.training.schedule import (  # noqa: E402
    SCHEDULER_TYPE,
    D0Schedule,
)
from tests._checkpoint_fixtures import (  # noqa: E402
    make_identity_and_provenance,
    make_resume_identity,
    make_rng_bundle,
    make_store,
    resolve_envelope,
)

_FIXED_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _tiny_config() -> D0ModelConfig:
    return D0ModelConfig(n_layers=2, dim=32, n_heads=4, head_dim=8, ffn_dim=48, vocab_size=100)


def _new_state(seed: int) -> tuple[D0Model, D0Optimizer, D0Schedule]:
    model = D0Model(_tiny_config())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initialize_model(model, n_layers=_tiny_config().n_layers, generator=generator)
    optimizer = D0Optimizer(
        model,
        D0OptimizerConfig(
            name="AdamW",
            beta1=0.9,
            beta2=0.95,
            epsilon=1.0e-08,
            weight_decay=0.1,
            weight_decay_includes=("attention_matrix_weights", "swiglu_matrix_weights"),
            weight_decay_excludes=("token_embedding_weight", "rmsnorm_weights"),
            gradient_clip_global_l2_norm=1.0,
            loss_reduction="mean_over_all_target_tokens_in_optimizer_update",
        ),
    )
    schedule = D0Schedule(
        peak_learning_rate=0.0006,
        final_learning_rate=6.0e-05,
        warmup_updates=200,
        optimizer_updates=4000,
    )
    return model, optimizer, schedule


def _one_update(
    model: D0Model, optimizer: D0Optimizer, schedule: D0Schedule, last_update: int, seed: int
) -> None:
    tokens = torch.randint(
        0, _tiny_config().vocab_size, (2, 16), generator=torch.Generator().manual_seed(seed)
    )
    optimizer.zero_grad()
    model(tokens).float().sum().backward()
    optimizer.clip_and_step(learning_rate=schedule.learning_rate_at(last_update + 1))


def _param_bytes(model: D0Model) -> tuple[CapturedTensor, ...]:
    tensors = []
    for name, parameter in sorted(model.named_parameters()):
        raw = parameter.detach().to("cpu").contiguous().numpy().astype("<f4", copy=False).tobytes()
        tensors.append(
            CapturedTensor(
                logical_name=name,
                dtype="float32",
                shape=tuple(parameter.shape),
                raw_bytes=raw,
            )
        )
    return tuple(tensors)


def _capture(
    model: D0Model, optimizer: D0Optimizer, schedule: D0Schedule, *, last_update: int
) -> CapturedCheckpointState:
    scalar = schedule.scalar_state(last_applied_update=last_update)
    scheduler_bytes = np.asarray([scalar.last_learning_rate], dtype="<f4").tobytes()
    return CapturedCheckpointState(
        parameters=_param_bytes(model),
        buffers=(),
        alias_groups=(),
        optimizer=None,
        scheduler=CapturedTensor(
            logical_name="scheduler.state",
            dtype="float32",
            shape=(1,),
            raw_bytes=scheduler_bytes,
        ),
        scaler=None,
        rng_bundle_bytes=make_rng_bundle(root_seed=2026080200).to_deterministic_json(),
        optimizer_slots=tuple(
            CapturedTensor(
                logical_name=name,
                dtype="float32",
                shape=tuple(tensor.shape),
                raw_bytes=tensor.contiguous().numpy().astype("<f4", copy=False).tobytes(),
            )
            for name, tensor in sorted(optimizer.slot_tensors().items())
        ),
        optimizer_scalar_state=safe_value_from_native(optimizer.scalar_state()),
        scheduler_scalar_state=safe_value_from_native(
            {
                "last_applied_update": scalar.last_applied_update,
                "last_learning_rate": scalar.last_learning_rate,
            }
        ),
        data_cursor=DataCursor(
            sampler_type="sequential",
            sampler_version=1,
            batch_size=64,
            sequence_length=1024,
            drop_last=False,
            epoch=0,
            position=last_update * 65536,
            accepted_samples=last_update * 64,
            accepted_sequences=last_update * 64,
        ),
        counters=CounterSnapshot(
            global_update=last_update,
            completed_microsteps=last_update * 16,
            accumulation_position=0,
            accepted_samples=last_update * 64,
            accepted_sequences=last_update * 64,
            processed_tokens=last_update * 65536,
        ),
        model_descriptor=ModelDescriptor(
            parameters=tuple(
                ModelParameterDescriptor(name=t.logical_name, shape=t.shape, dtype=t.dtype)
                for t in _param_bytes(model)
            ),
            buffers=(),
        ),
        optimizer_descriptor=optimizer.descriptor(),
        scheduler_descriptor=SchedulerDescriptor(scheduler_type=SCHEDULER_TYPE, state_shape=(1,)),
        scaler_descriptor=None,
        rng_descriptor=RngDescriptor(adapter_set=(), framework_versions=()),
        data_descriptor=DataDescriptor(
            identity=DataIdentity(
                dataset_digest="a" * 64,
                split="train",
                length=2097152000,
                preprocessing_identity="d0.v1",
                tokenizer_identity="d0-gpt-neox-corrected-v1",
                packing_policy="d0_packed_v1",
                sequence_policy="packed_1024",
                shard_selection="all",
                data_config_digest="b" * 64,
            ),
            sampler_type="sequential",
            sampler_version=1,
            batch_size=64,
            sequence_length=1024,
            drop_last=False,
        ),
        topology_descriptor=TopologyDescriptor(world_size=1, rank_assignment=(0,)),
    )


class _D0TorchTarget:
    def __init__(self) -> None:
        self.model, self.optimizer, self.schedule = _new_state(seed=1)  # init differs on purpose
        self.counters: CounterSnapshot | None = None
        self.scheduler_scalar: dict[str, Any] | None = None
        self.cursor: DataCursor | None = None
        self.applied_scaler: CapturedTensor | None = None


class _D0TorchFactory:
    """StateFactory applying a D0 checkpoint into fresh torch objects."""

    def __init__(self) -> None:
        self.target: _D0TorchTarget | None = None
        self.committed = False

    def create(self) -> _D0TorchTarget:
        self.target = _D0TorchTarget()
        return self.target

    def apply_parameters(self, target: _D0TorchTarget, tensors: dict[str, CapturedTensor]) -> None:
        by_name = dict(target.model.named_parameters())
        for name, captured in tensors.items():
            parameter = by_name[name]
            restored = (
                torch.frombuffer(bytearray(captured.raw_bytes), dtype=torch.float32)
                .reshape(tuple(captured.shape))
                .clone()
            )
            with torch.no_grad():
                parameter.copy_(restored)

    def apply_buffers(self, target: _D0TorchTarget, tensors: dict[str, CapturedTensor]) -> None:
        if tensors:
            raise AssertionError("D0 model captured unexpected buffers")

    def apply_optimizer(self, target: _D0TorchTarget, tensor: CapturedTensor | None) -> None:
        if tensor is not None:
            raise AssertionError("D0 uses slot-based optimizer state, not a legacy tensor")

    def apply_optimizer_state(
        self,
        target: _D0TorchTarget,
        state_tensor: CapturedTensor | None,
        slots: dict[tuple[int, str, str], CapturedTensor],
        scalar_state: Any,
    ) -> None:
        if state_tensor is not None:
            raise AssertionError("unexpected legacy optimizer state tensor")
        by_key = {
            f"optimizer.{slot_name}.{param_name}": torch.frombuffer(
                bytearray(captured.raw_bytes), dtype=torch.float32
            ).reshape(tuple(captured.shape))
            for (_group, param_name, slot_name), captured in slots.items()
        }
        target.optimizer.apply_state(by_key, safe_value_to_native(scalar_state))

    def apply_scheduler(self, target: _D0TorchTarget, tensor: CapturedTensor | None) -> None:
        if tensor is None:
            raise AssertionError("D0 scheduler captures a state tensor")

    def apply_scheduler_state(self, target: _D0TorchTarget, bundle: Any) -> None:
        scalar = getattr(bundle, "scalar_state", None)
        native = safe_value_to_native(scalar) if scalar is not None else {}
        target.scheduler_scalar = dict(native)

    def apply_scaler(self, target: _D0TorchTarget, tensor: CapturedTensor | None) -> None:
        if tensor is not None:
            raise AssertionError("D0 uses bf16 autocast: no scaler state")

    def apply_data_cursor(self, target: _D0TorchTarget, cursor: Any) -> None:
        target.cursor = cursor

    def apply_counters(self, target: _D0TorchTarget, counters: Any) -> None:
        target.counters = counters

    def commit_to_live(self, target: _D0TorchTarget) -> _D0TorchTarget:
        self.committed = True
        return target

    def revert_commit_to_live(self, target: _D0TorchTarget) -> None:
        self.committed = False


class TestD0StateRoundTrip:
    def test_restore_continues_bit_identically(self, tmp_path: Path) -> None:
        identity, provenance = make_identity_and_provenance(tmp_path)
        store = CheckpointStore(make_store(tmp_path, identity=identity))

        # Attempt A: initialize, apply one update, capture at u=1.
        model, optimizer, schedule = _new_state(seed=2026080200)
        _one_update(model, optimizer, schedule, last_update=0, seed=101)
        captured = _capture(model, optimizer, schedule, last_update=1)

        record = store.save(
            identity=identity,
            captured=captured,
            configuration_envelope=resolve_envelope(),
            provenance=provenance,
            rng_bundle=make_rng_bundle(root_seed=2026080200),
            parent=None,
        )

        # Attempt B (resume): load through the parent's store (its registry
        # indexes the artifact) with the resume identity, mirroring the smoke
        # runner's reconstructed-parent-store pattern.
        resume_identity = make_resume_identity(identity, parent_artifact_id=record.artifact_id)
        parent_store = CheckpointStore(make_store(tmp_path, identity=identity))
        archive = parent_store.load(record.artifact_id, expected_identity=resume_identity)

        expected = CompatibilityDescriptor(
            specification_fingerprint=identity.fingerprint_digest_str(),
            model_descriptor=captured.model_descriptor,
            optimizer_descriptor=captured.optimizer_descriptor,
            scheduler_descriptor=captured.scheduler_descriptor,
            scaler_descriptor=None,
            rng_descriptor=captured.rng_descriptor,
            data_descriptor=captured.data_descriptor,
            topology_descriptor=captured.topology_descriptor,
        )
        factory = _D0TorchFactory()

        def _load_rng() -> Any:
            return make_rng_bundle(root_seed=2026080200)

        consumed: list[Any] = []

        def _consume_rng(bundle: Any) -> None:
            consumed.append(bundle)

        transaction = RestoreTransaction(
            archive=archive,
            factory=cast(Any, factory),
            rng_bundle_loader=_load_rng,
            rng_consumer=_consume_rng,
            expected_descriptor=expected,
        )
        transaction.prepare()
        transaction.commit()
        assert factory.committed
        restored = factory.target
        assert restored is not None
        assert restored.counters is not None and restored.counters.global_update == 1
        assert restored.counters.accumulation_position == 0
        assert restored.scheduler_scalar is not None
        assert restored.scheduler_scalar["last_applied_update"] == 1
        assert consumed

        # Continuation: one more identical update on both attempts must
        # produce bit-identical parameters and moment slots.
        _one_update(model, optimizer, schedule, last_update=1, seed=202)
        _one_update(restored.model, restored.optimizer, schedule, last_update=1, seed=202)

        original_params = dict(model.named_parameters())
        for name, parameter in restored.model.named_parameters():
            assert torch.equal(parameter.detach(), original_params[name].detach()), name
        original_slots = optimizer.slot_tensors()
        for name, tensor in restored.optimizer.slot_tensors().items():
            assert torch.equal(tensor, original_slots[name]), name

    def test_captured_state_is_quiescent(self, tmp_path: Path) -> None:
        """The captured snapshot must satisfy the V1 save boundary."""

        model, optimizer, schedule = _new_state(seed=5)
        _one_update(model, optimizer, schedule, last_update=0, seed=77)
        captured = _capture(model, optimizer, schedule, last_update=1)
        assert captured.counters.accumulation_position == 0
        assert captured.counters.global_update == 1
        assert captured.counters.processed_tokens == 65536
        assert captured.scheduler_scalar_state is not None
        scheduler_native = safe_value_to_native(captured.scheduler_scalar_state)
        assert scheduler_native["last_applied_update"] == 1
        assert scheduler_native["last_learning_rate"] == schedule.learning_rate_at(1)
        optimizer_native = safe_value_to_native(captured.optimizer_scalar_state)
        assert optimizer_native["optimizer_type"] == "AdamW"
        assert set(optimizer_native["steps"].values()) == {1}
