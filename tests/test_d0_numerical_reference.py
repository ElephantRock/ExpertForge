"""Numerical-reference tests: PyTorch primitives vs independent NumPy oracles.

Each oracle is a small, standalone NumPy implementation written directly from
the ratified primitive specification. Agreement between the oracle and the
PyTorch module is a strong correctness signal: a divergence means the two
implementations disagree on the spec, not that one imported the other.
"""

from __future__ import annotations

import numpy as np
import torch

from expertforge.d0.model.attention import CausalSelfAttention
from expertforge.d0.model.config import D0ModelConfig
from expertforge.d0.model.initialization import initialize_model
from expertforge.d0.model.numpy_oracle import (
    numpy_attention,
    numpy_rmsnorm,
    numpy_rope,
    numpy_swiglu,
)
from expertforge.d0.model.rmsnorm import RMSNorm
from expertforge.d0.model.rope import apply_rope
from expertforge.d0.model.swiglu import SwiGLU
from expertforge.d0.model.transformer import D0Model

# Tolerances: float32 accumulation in both paths, so ~1e-4 relative agreement is
# comfortable for these small inputs.
ATOL = 1e-4
RTOL = 1e-4


def test_rmsnorm_matches_numpy_oracle() -> None:
    torch.manual_seed(0)
    module = RMSNorm(5, epsilon=1e-5)
    module.weight.data = torch.randn(5)
    x = torch.randn(2, 3, 5)

    got = module(x).detach().numpy()
    expected = numpy_rmsnorm(x.numpy(), module.weight.detach().numpy(), 1e-5)

    assert np.allclose(got, expected, atol=ATOL, rtol=RTOL)


def test_rope_matches_numpy_oracle() -> None:
    torch.manual_seed(0)
    # (batch=1, heads=2, seq=4, head_dim=6) — adjacent-pair rotation.
    head_dim = 6
    x = torch.randn(1, 2, 4, head_dim)
    positions = torch.arange(4).view(1, 1, 4).expand(1, 2, 4).contiguous()

    got = apply_rope(x, positions, 10000.0, head_dim).detach().numpy()
    expected = numpy_rope(x.numpy(), positions.detach().numpy(), 10000.0, head_dim)

    assert np.allclose(got, expected, atol=ATOL, rtol=RTOL)


def test_rope_zero_positions_is_identity() -> None:
    head_dim = 8
    x = torch.randn(1, 1, 3, head_dim)
    positions = torch.zeros(1, 1, 3, dtype=torch.long)

    out = apply_rope(x, positions, 10000.0, head_dim)

    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_swiglu_matches_numpy_oracle() -> None:
    torch.manual_seed(0)
    module = SwiGLU(4, 8)
    x = torch.randn(2, 4)

    got = module(x).detach().numpy()
    expected = numpy_swiglu(
        x.numpy(),
        module.gate_proj.weight.detach().numpy(),
        module.up_proj.weight.detach().numpy(),
        module.down_proj.weight.detach().numpy(),
    )

    assert np.allclose(got, expected, atol=ATOL, rtol=RTOL)


def test_attention_single_head_matches_numpy_oracle() -> None:
    """End-to-end attention: torch module vs a single-head numpy oracle.

    We drive the torch attention module on a single-head config and reconstruct
    the inputs the oracle expects by extracting the post-projection Q/K/V via
    hooks. The oracle then computes the full causal attention independently.
    """
    torch.manual_seed(0)
    dim = head_dim = 4
    seq_len = 5
    module = CausalSelfAttention(dim=dim, n_heads=1, head_dim=head_dim, rope_base=10000.0)
    module.eval()

    x = torch.randn(1, seq_len, dim)

    # Capture Q, K, V after projection but before RoPE by calling the
    # projections directly (operation order: project -> reshape -> RoPE).
    with torch.no_grad():
        q_proj = module.q_proj(x).view(seq_len, head_dim)
        k_proj = module.k_proj(x).view(seq_len, head_dim)
        v_proj = module.v_proj(x).view(seq_len, head_dim)

        positions = torch.arange(seq_len).view(1, 1, seq_len)
        q_rot = apply_rope(q_proj.view(1, 1, seq_len, head_dim), positions, 10000.0, head_dim)
        k_rot = apply_rope(k_proj.view(1, 1, seq_len, head_dim), positions, 10000.0, head_dim)

        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
        expected_values = numpy_attention(
            q_rot.view(seq_len, head_dim).numpy(),
            k_rot.view(seq_len, head_dim).numpy(),
            v_proj.view(seq_len, head_dim).numpy(),
            head_dim,
            mask.numpy(),
        )

        # The torch module applies the output projection to the attention output;
        # compare against that by composing oracle attention + torch out_proj.
        expected_context = torch.from_numpy(expected_values).float().view(1, seq_len, head_dim)
        expected_logits_pre_out = expected_context
        # Re-derive full module output via out_proj to keep the oracle honest:
        oracle_full = module.out_proj(expected_logits_pre_out.view(1, seq_len, dim)).numpy()

        full = module(x).detach().numpy()

    assert np.allclose(full, oracle_full, atol=ATOL, rtol=RTOL)


def test_causal_invariance_future_tokens_do_not_affect_earlier_logits() -> None:
    torch.manual_seed(0)
    cfg = D0ModelConfig(n_layers=2, dim=32, n_heads=2, head_dim=16, ffn_dim=48, vocab_size=64)
    model = D0Model(cfg)
    initialize_model(model, cfg.n_layers, torch.Generator(device="cpu").manual_seed(7))
    model.eval()

    ids = torch.tensor([[37, 43, 12, 8, 63, 9]])
    ids_modified = ids.clone()
    ids_modified[0, 5] = 16  # change only the future (last) token

    with torch.no_grad():
        logits = model(ids)
        logits_modified = model(ids_modified)

    # Positions strictly before the changed token must be byte-for-byte equal.
    assert torch.equal(logits[0, :5], logits_modified[0, :5])
    # The changed position itself must differ (its own input changed).
    assert not torch.equal(logits[0, 5], logits_modified[0, 5])


def test_attention_causality_via_zeroed_future() -> None:
    """Replacing future value entries with zeros cannot change present outputs.

    This directly exercises the causal mask: the value matrix at future
    positions is irrelevant to earlier queries, so zeroing it must not move the
    outputs at earlier positions.
    """
    torch.manual_seed(0)
    dim = head_dim = 4
    seq_len = 6
    module = CausalSelfAttention(dim=dim, n_heads=1, head_dim=head_dim, rope_base=10000.0)
    module.eval()
    x = torch.randn(1, seq_len, dim)

    with torch.no_grad():
        out_full = module(x)

        # Monkeypatch v_proj to zero future rows and confirm earlier outputs hold.
        original_v = module.v_proj.weight.clone()
        try:
            # Zeroing the input at future positions zeros their V contributions.
            x_future_zeroed = x.clone()
            x_future_zeroed[0, 4:] = 0.0
            out_future_zeroed = module(x_future_zeroed)
            # The last positions can change (their own V is now zero), but the
            # earliest positions are unaffected because they never attended to
            # the future anyway.
            assert torch.allclose(out_full[0, :4], out_future_zeroed[0, :4], atol=1e-6)
        finally:
            module.v_proj.weight.data.copy_(original_v)
