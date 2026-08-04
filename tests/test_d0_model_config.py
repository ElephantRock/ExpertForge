"""Tests for the D0 model configuration profiles."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from expertforge.d0.model.config import (  # noqa: E402
    D0ModelConfig,
    canonical_config,
    qualification_config,
)


def test_qualification_config_dimensions() -> None:
    cfg = qualification_config()

    assert cfg.n_layers == 8
    assert cfg.dim == 256
    assert cfg.n_heads == 4
    assert cfg.head_dim == 64
    assert cfg.ffn_dim == 768


def test_canonical_config_dimensions() -> None:
    cfg = canonical_config()

    assert cfg.n_layers == 12
    assert cfg.dim == 576
    assert cfg.n_heads == 9
    assert cfg.head_dim == 64
    assert cfg.ffn_dim == 1536


def test_qualification_head_dim_consistency() -> None:
    cfg = qualification_config()

    assert cfg.head_dim == cfg.dim // cfg.n_heads


def test_canonical_head_dim_consistency() -> None:
    cfg = canonical_config()

    assert cfg.head_dim == cfg.dim // cfg.n_heads


def test_contract_defaults_are_frozen() -> None:
    cfg = qualification_config()

    # Defaults mandated by the ratified D0 contract.
    assert cfg.vocab_size == 50257
    assert cfg.rmsnorm_epsilon == 1e-5
    assert cfg.rope_base == 10000.0
    assert cfg.max_seq_len == 1024
    assert cfg.tie_embeddings is True
    assert cfg.bias is False


def test_config_is_frozen() -> None:
    cfg = qualification_config()

    # ``FrozenInstanceError`` subclasses ``AttributeError``.
    with pytest.raises(AttributeError):
        cfg.dim = 999  # type: ignore[misc]


def test_rejects_bias_or_untied_embeddings() -> None:
    with pytest.raises(ValueError):
        D0ModelConfig(
            n_layers=2,
            dim=32,
            n_heads=2,
            head_dim=16,
            ffn_dim=48,
            bias=True,
        )
    with pytest.raises(ValueError):
        D0ModelConfig(
            n_layers=2,
            dim=32,
            n_heads=2,
            head_dim=16,
            ffn_dim=48,
            tie_embeddings=False,
        )


def test_rejects_head_dim_mismatch() -> None:
    with pytest.raises(ValueError):
        D0ModelConfig(
            n_layers=2,
            dim=32,
            n_heads=2,
            head_dim=8,  # 32 // 2 == 16, not 8.
            ffn_dim=48,
        )
