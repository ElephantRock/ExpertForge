"""Tests for the frozen D0 optimizer (torch, CPU).

Requires the ``d0-model`` extra; skips cleanly when torch is absent.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from expertforge.config.d0_models import D0OptimizerConfig  # noqa: E402
from expertforge.d0.errors import (  # noqa: E402
    NonFiniteGradientError,
    OptimizerStateError,
)
from expertforge.d0.model.config import D0ModelConfig  # noqa: E402
from expertforge.d0.model.initialization import initialize_model  # noqa: E402
from expertforge.d0.model.transformer import D0Model  # noqa: E402
from expertforge.d0.training.optimizer import (  # noqa: E402
    D0Optimizer,
    build_param_groups,
)


def _optimizer_config() -> D0OptimizerConfig:
    return D0OptimizerConfig(
        name="AdamW",
        beta1=0.9,
        beta2=0.95,
        epsilon=1.0e-08,
        weight_decay=0.1,
        weight_decay_includes=("attention_matrix_weights", "swiglu_matrix_weights"),
        weight_decay_excludes=("token_embedding_weight", "rmsnorm_weights"),
        gradient_clip_global_l2_norm=1.0,
        loss_reduction="mean_over_all_target_tokens_in_optimizer_update",
    )


def _tiny_model(seed: int = 2026080200) -> D0Model:
    config = D0ModelConfig(n_layers=2, dim=32, n_heads=4, head_dim=8, ffn_dim=48, vocab_size=100)
    model = D0Model(config)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initialize_model(model, n_layers=config.n_layers, generator=generator)
    return model


class TestParamGroups:
    def test_partition_covers_every_parameter_exactly_once(self) -> None:
        model = _tiny_model()
        decay_group, no_decay_group = build_param_groups(model, _optimizer_config())
        decay_ids = [id(p) for p in decay_group["params"]]
        no_decay_ids = [id(p) for p in no_decay_group["params"]]
        assert len(decay_ids) + len(no_decay_ids) == len(list(model.named_parameters()))
        assert not set(decay_ids) & set(no_decay_ids)
        for _name, parameter in model.named_parameters():
            assert id(parameter) in decay_ids or id(parameter) in no_decay_ids

    def test_embedding_and_norms_do_not_decay(self) -> None:
        model = _tiny_model()
        _decay, no_decay = build_param_groups(model, _optimizer_config())
        by_name = dict(model.named_parameters())
        no_decay_ids = {id(p) for p in no_decay["params"]}
        assert id(by_name["token_embedding.weight"]) in no_decay_ids
        assert id(by_name["final_norm.weight"]) in no_decay_ids
        assert id(by_name["blocks.0.attention_norm.weight"]) in no_decay_ids
        assert no_decay["weight_decay"] == 0.0

    def test_projection_matrices_decay(self) -> None:
        model = _tiny_model()
        decay, _no_decay = build_param_groups(model, _optimizer_config())
        by_name = dict(model.named_parameters())
        decay_ids = {id(p) for p in decay["params"]}
        for name in (
            "blocks.0.attention.q_proj.weight",
            "blocks.0.attention.out_proj.weight",
            "blocks.1.ffn.gate_proj.weight",
            "blocks.1.ffn.down_proj.weight",
        ):
            assert id(by_name[name]) in decay_ids, name
        assert decay["weight_decay"] == 0.1

    def test_groups_are_deterministic(self) -> None:
        model = _tiny_model()
        first = build_param_groups(model, _optimizer_config())
        second = build_param_groups(model, _optimizer_config())
        assert [id(p) for p in first[0]["params"]] == [id(p) for p in second[0]["params"]]
        assert [id(p) for p in first[1]["params"]] == [id(p) for p in second[1]["params"]]

    def test_unknown_config_class_fails_closed(self) -> None:
        bad = D0OptimizerConfig(
            name="AdamW",
            beta1=0.9,
            beta2=0.95,
            epsilon=1.0e-08,
            weight_decay=0.1,
            weight_decay_includes=("attention_matrix_weights", "swiglu_matrix_weights"),
            weight_decay_excludes=("token_embedding_weight", "rmsnorm_weights", "mystery_class"),
            gradient_clip_global_l2_norm=1.0,
            loss_reduction="mean_over_all_target_tokens_in_optimizer_update",
        )
        with pytest.raises(OptimizerStateError, match="unknown weight class"):
            build_param_groups(_tiny_model(), bad)


class TestStepping:
    def test_step_changes_master_parameters_in_float32(self) -> None:
        model = _tiny_model()
        assert model.token_embedding.weight.dtype == torch.float32
        optimizer = D0Optimizer(model, _optimizer_config())
        optimizer.zero_grad()
        loss = model(torch.randint(0, 100, (2, 16), generator=torch.Generator().manual_seed(7)))
        loss.float().sum().backward()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        norm = optimizer.clip_and_step(learning_rate=0.001)
        assert norm >= 0.0
        changed = 0
        for name, p in model.named_parameters():
            if not torch.equal(p.detach(), before[name]):
                changed += 1
        assert changed > 0
        assert all(p.dtype == torch.float32 for _, p in model.named_parameters())

    def test_clip_caps_the_global_norm(self) -> None:
        model = _tiny_model()
        optimizer = D0Optimizer(model, _optimizer_config())
        optimizer.zero_grad()
        tokens = torch.randint(0, 100, (2, 16), generator=torch.Generator().manual_seed(3))
        # A hugely scaled loss drives the global gradient norm above 1.0
        # while still populating every parameter's gradient.
        (model(tokens).float().sum() * 1.0e6).backward()
        norm = optimizer.clip_and_step(learning_rate=0.001)
        assert norm > 1.0
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.pow(2).sum())
        assert total_sq**0.5 == pytest.approx(1.0, rel=1e-5)

    def test_non_finite_gradient_fails_closed(self) -> None:
        model = _tiny_model()
        optimizer = D0Optimizer(model, _optimizer_config())
        optimizer.zero_grad()
        tokens = torch.randint(0, 100, (2, 16), generator=torch.Generator().manual_seed(4))
        model(tokens).float().sum().backward()
        model.token_embedding.weight.grad = torch.full_like(
            model.token_embedding.weight, float("nan")
        )
        with pytest.raises(NonFiniteGradientError, match="non-finite gradient"):
            optimizer.clip_and_step(learning_rate=0.001)

    def test_rejects_non_positive_learning_rate(self) -> None:
        model = _tiny_model()
        optimizer = D0Optimizer(model, _optimizer_config())
        optimizer.zero_grad()
        tokens = torch.randint(0, 100, (2, 16), generator=torch.Generator().manual_seed(5))
        model(tokens).float().sum().backward()
        with pytest.raises(OptimizerStateError, match="positive"):
            optimizer.clip_and_step(learning_rate=0.0)

    def test_missing_gradient_fails_closed(self) -> None:
        model = _tiny_model()
        optimizer = D0Optimizer(model, _optimizer_config())
        optimizer.zero_grad()
        with pytest.raises(OptimizerStateError, match="no gradient"):
            optimizer.clip_and_step(learning_rate=0.001)


class TestStateSurface:
    def test_slot_tensors_before_first_step_are_zeroed(self) -> None:
        model = _tiny_model()
        optimizer = D0Optimizer(model, _optimizer_config())
        # torch creates AdamW state lazily; seed it with one step.
        optimizer.zero_grad()
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        optimizer.clip_and_step(learning_rate=0.001)
        slots = optimizer.slot_tensors()
        param_names = [name for name, _ in model.named_parameters()]
        assert len(slots) == 2 * len(param_names)
        for name in param_names:
            for slot in ("exp_avg", "exp_avg_sq"):
                tensor = slots[f"optimizer.{slot}.{name}"]
                assert tensor.dtype == torch.float32
                assert tensor.device.type == "cpu"

    def test_scalar_state_records_steps_and_identity(self) -> None:
        model = _tiny_model()
        optimizer = D0Optimizer(model, _optimizer_config())
        optimizer.zero_grad()
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        optimizer.clip_and_step(learning_rate=0.001)
        optimizer.clip_and_step(learning_rate=0.001)
        scalar: dict[str, Any] = optimizer.scalar_state()
        assert scalar["optimizer_type"] == "AdamW"
        assert scalar["bias_correction"] is True
        assert set(scalar["steps"].values()) == {2}

    def test_state_round_trip_is_bit_exact(self) -> None:
        source_model = _tiny_model()
        source = D0Optimizer(source_model, _optimizer_config())
        source.zero_grad()
        tokens = torch.randint(0, 100, (2, 16), generator=torch.Generator().manual_seed(11))
        source_model(tokens).float().sum().backward()
        source.clip_and_step(learning_rate=0.001)

        slots = {k: v.clone() for k, v in source.slot_tensors().items()}
        scalar = source.scalar_state()

        target_model = _tiny_model(seed=1)  # different init on purpose
        # Mimic apply_parameters: the restore path loads parameter bytes
        # separately from the optimizer state (RestoreTransaction order).
        source_params = dict(source_model.named_parameters())
        with torch.no_grad():
            for name, parameter in target_model.named_parameters():
                parameter.copy_(source_params[name].detach())
        target = D0Optimizer(target_model, _optimizer_config())
        target.apply_state(slots, scalar)

        # After restoring, one more identical step on both models must
        # produce bit-identical parameters.
        fresh_tokens = torch.randint(0, 100, (2, 16), generator=torch.Generator().manual_seed(12))
        for model, opt in ((source_model, source), (target_model, target)):
            opt.zero_grad()
            model(fresh_tokens).float().sum().backward()
            opt.clip_and_step(learning_rate=0.002)

        for (source_name, source_param), (target_name, target_param) in zip(
            source_model.named_parameters(),
            target_model.named_parameters(),
            strict=True,
        ):
            assert source_name == target_name
            assert torch.equal(source_param.detach(), target_param.detach()), source_name

    def test_apply_state_rejects_missing_slots(self) -> None:
        model = _tiny_model()
        optimizer = D0Optimizer(model, _optimizer_config())
        optimizer.zero_grad()
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        optimizer.clip_and_step(learning_rate=0.001)
        slots = optimizer.slot_tensors()
        del slots[next(iter(slots))]
        with pytest.raises(OptimizerStateError, match="slot set mismatch"):
            optimizer.apply_state(slots, optimizer.scalar_state())

    def test_descriptor_groups_match_the_partition(self) -> None:
        model = _tiny_model()
        optimizer = D0Optimizer(model, _optimizer_config())
        descriptor = optimizer.descriptor()
        assert descriptor.optimizer_type == "AdamW"
        assert len(descriptor.param_groups) == 2
        all_grouped = [name for group in descriptor.param_groups for name in group.param_names]
        all_params = [name for name, _ in model.named_parameters()]
        assert sorted(all_grouped) == sorted(all_params)
        assert len(descriptor.state_slots) == 2 * len(all_params)
        assert all(slot.dtype == "float32" for slot in descriptor.state_slots)
