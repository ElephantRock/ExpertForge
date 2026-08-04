"""Tests for the frozen deterministic D0 initialisation scheme."""

from __future__ import annotations

import math

import torch

from expertforge.d0.model.config import D0ModelConfig, qualification_config
from expertforge.d0.model.initialization import initialize_model
from expertforge.d0.model.transformer import D0Model

_SEED = 2_026_080_200


def _build_and_init() -> D0Model:
    model = D0Model(qualification_config())
    generator = torch.Generator(device="cpu").manual_seed(_SEED)
    initialize_model(model, qualification_config().n_layers, generator)
    return model


def test_initialisation_is_deterministic() -> None:
    model_a = _build_and_init()
    model_b = _build_and_init()

    values_a = list(model_a.state_dict().values())
    values_b = list(model_b.state_dict().values())
    for a, b in zip(values_a, values_b, strict=True):
        assert torch.equal(a, b)


def test_different_seeds_produce_different_weights() -> None:
    model_a = D0Model(qualification_config())
    initialize_model(
        model_a, qualification_config().n_layers, torch.Generator(device="cpu").manual_seed(1)
    )
    model_b = D0Model(qualification_config())
    initialize_model(
        model_b, qualification_config().n_layers, torch.Generator(device="cpu").manual_seed(2)
    )

    # At least the token embedding (a large tensor) must differ.
    assert not torch.equal(model_a.token_embedding.weight, model_b.token_embedding.weight)


def test_embedding_std_approximately_002() -> None:
    model = _build_and_init()

    # Large tensor -> the empirical std converges to the nominal 0.02.
    std = float(model.token_embedding.weight.std().detach())
    assert abs(std - 0.02) < 1e-3


def test_qkv_and_swiglu_gate_up_std_approximately_002() -> None:
    model = _build_and_init()

    for name in (
        "blocks.0.attention.q_proj.weight",
        "blocks.0.attention.k_proj.weight",
        "blocks.0.attention.v_proj.weight",
        "blocks.0.ffn.gate_proj.weight",
        "blocks.0.ffn.up_proj.weight",
    ):
        tensor = model.state_dict()[name]
        std = float(tensor.std().detach())
        assert abs(std - 0.02) < 1e-3, name


def test_residual_scaled_std_matches_formula() -> None:
    cfg = qualification_config()
    model = _build_and_init()
    expected_std = 0.02 / math.sqrt(2 * cfg.n_layers)

    for name in (
        "blocks.0.attention.out_proj.weight",
        "blocks.0.ffn.down_proj.weight",
    ):
        tensor = model.state_dict()[name]
        std = float(tensor.std().detach())
        assert abs(std - expected_std) < 1e-4, name


def test_rmsnorm_weights_are_exactly_one() -> None:
    model = _build_and_init()

    for name, tensor in model.state_dict().items():
        if name.endswith("norm.weight"):
            assert torch.allclose(tensor, torch.ones_like(tensor)), name


def test_no_biases_after_initialisation() -> None:
    model = _build_and_init()

    assert not any("bias" in name for name in model.state_dict())
    for _, param in model.named_parameters():
        # Sanity: every parameter is a leaf tensor participating in autograd.
        assert param.requires_grad


def test_initialisation_residual_scaling_on_small_config() -> None:
    # A smaller config keeps the test fast while exercising the same code path.
    # Small tensors have noticeable sampling noise, so use a relative tolerance.
    cfg = D0ModelConfig(n_layers=2, dim=32, n_heads=2, head_dim=16, ffn_dim=48, vocab_size=100)
    model = D0Model(cfg)
    initialize_model(model, cfg.n_layers, torch.Generator(device="cpu").manual_seed(_SEED))

    expected_residual = 0.02 / math.sqrt(2 * cfg.n_layers)
    out_std = float(model.state_dict()["blocks.0.attention.out_proj.weight"].std().detach())
    # Empirical std over a 32x32 tensor is within ~5% of the nominal value.
    assert abs(out_std - expected_residual) / expected_residual < 0.05


def test_initialisation_residual_std_is_smaller_than_embedding_std() -> None:
    # The residual-scaled std (0.02/sqrt(2L)) must be strictly smaller than the
    # embedding std (0.02), regardless of empirical noise.
    model = _build_and_init()

    out_std = float(model.state_dict()["blocks.0.attention.out_proj.weight"].std().detach())
    emb_std = float(model.token_embedding.weight.std().detach())

    assert out_std < emb_std
