"""Numerical-reference tests: PyTorch primitives vs independent NumPy oracles.

Each oracle is a small, standalone NumPy implementation written directly from
the ratified primitive specification. Agreement between the oracle and the
PyTorch module is a strong correctness signal: a divergence means the two
implementations disagree on the spec, not that one imported the other.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from expertforge.d0.model.attention import CausalSelfAttention  # noqa: E402
from expertforge.d0.model.config import D0ModelConfig  # noqa: E402
from expertforge.d0.model.initialization import initialize_model  # noqa: E402
from expertforge.d0.model.numpy_oracle import (  # noqa: E402
    numpy_attention,
    numpy_full_model_forward,
    numpy_rmsnorm,
    numpy_rope,
    numpy_swiglu,
)
from expertforge.d0.model.rmsnorm import RMSNorm  # noqa: E402
from expertforge.d0.model.rope import apply_rope  # noqa: E402
from expertforge.d0.model.swiglu import SwiGLU  # noqa: E402
from expertforge.d0.model.transformer import D0Model  # noqa: E402

# Tolerances: float32 accumulation in both paths, so ~1e-4 relative agreement is
# comfortable for these small inputs.
ATOL = 1e-4
RTOL = 1e-4

# Tiny-model dimensions used by the full-model oracle and the serialization /
# gradient evidence categories. Small enough to run in milliseconds while still
# exercising every primitive (multi-head attention, RoPE, SwiGLU, RMSNorm) and
# the tied output projection.
TINY_CONFIG = D0ModelConfig(
    n_layers=2,
    dim=32,
    n_heads=4,
    head_dim=8,
    ffn_dim=48,
    vocab_size=100,
)
_TINY_SEED = 2_026_080_200


def _build_tiny_model() -> D0Model:
    model = D0Model(TINY_CONFIG)
    initialize_model(
        model, TINY_CONFIG.n_layers, torch.Generator(device="cpu").manual_seed(_TINY_SEED)
    )
    return model


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

    The oracle computes the **entire** attention module — including the output
    projection — in NumPy. Only the input ``x`` and the four stored projection
    weights leave the torch module; nothing from the torch forward pass is
    reused. Agreement therefore demonstrates that the torch ``CausalSelfAttention``
    and the independent oracle implement the same ratified attention semantics.
    """
    torch.manual_seed(0)
    dim = head_dim = 4
    seq_len = 5
    module = CausalSelfAttention(dim=dim, n_heads=1, head_dim=head_dim, rope_base=10000.0)
    module.eval()

    x = torch.randn(1, seq_len, dim)

    x_np = x.detach().numpy()
    q_w = module.q_proj.weight.detach().numpy()
    k_w = module.k_proj.weight.detach().numpy()
    v_w = module.v_proj.weight.detach().numpy()
    out_w = module.out_proj.weight.detach().numpy()

    # Project x -> Q, K, V (bias-free nn.Linear: x @ W.T).
    q = (x_np.astype(np.float32) @ q_w.astype(np.float32).T).reshape(seq_len, head_dim)
    k = (x_np.astype(np.float32) @ k_w.astype(np.float32).T).reshape(seq_len, head_dim)
    v = (x_np.astype(np.float32) @ v_w.astype(np.float32).T).reshape(seq_len, head_dim)

    # Apply RoPE to Q and K (never V) using the same adjacent-pair convention.
    positions = np.arange(seq_len, dtype=np.float32).reshape(1, 1, seq_len)
    q_rot = numpy_rope(q.reshape(1, 1, seq_len, head_dim), positions, 10000.0, head_dim).reshape(
        seq_len, head_dim
    )
    k_rot = numpy_rope(k.reshape(1, 1, seq_len, head_dim), positions, 10000.0, head_dim).reshape(
        seq_len, head_dim
    )

    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    context = numpy_attention(q_rot, k_rot, v, head_dim, mask)  # (seq, head_dim)

    # Output projection computed entirely in NumPy — no torch out_proj reuse.
    oracle_full = (
        context.reshape(1, seq_len, head_dim).astype(np.float32) @ out_w.astype(np.float32).T
    )

    with torch.no_grad():
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


# ---------------------------------------------------------------------------
# Evidence category 3d — tiny full-model NumPy oracle.
# ---------------------------------------------------------------------------


def test_full_model_matches_numpy_oracle() -> None:
    """The PyTorch D0Model and the NumPy full-model oracle must agree.

    Both paths run the same tiny model (2 layers, dim=32, 4 heads, head_dim=8,
    ffn_dim=48, vocab=100) on the same input. The oracle consumes only the
    state-dict weights and the config — it never calls into the torch forward
    pass — so agreement validates the entire architecture and its composition.
    """
    model = _build_tiny_model().eval()
    ids = torch.randint(
        0, TINY_CONFIG.vocab_size, (2, 6), generator=torch.Generator().manual_seed(3)
    )

    with torch.no_grad():
        torch_logits = model(ids).detach().numpy()

    weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    oracle_logits = numpy_full_model_forward(ids.numpy(), weights, TINY_CONFIG)

    assert oracle_logits.shape == torch_logits.shape
    assert np.allclose(torch_logits, oracle_logits, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Evidence category 3e — serialization and gradient tests.
# ---------------------------------------------------------------------------


def test_state_dict_round_trip_preserves_logits() -> None:
    """Saving and reloading the state dict must reproduce logits exactly."""
    model = _build_tiny_model().eval()
    ids = torch.tensor([[1, 5, 9, 42, 7, 0]])

    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    reloaded = D0Model(TINY_CONFIG).eval()
    reloaded.load_state_dict(state_dict)

    with torch.no_grad():
        before = model(ids)
        after = reloaded(ids)

    assert torch.allclose(before, after, atol=0.0, rtol=0.0)


def test_tie_holds_after_state_dict_round_trip() -> None:
    """The tied output projection must still alias the embedding after reload."""
    model = _build_tiny_model().eval()
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    reloaded = D0Model(TINY_CONFIG).eval()
    reloaded.load_state_dict(state_dict)

    assert reloaded.output_weight is reloaded.token_embedding.weight


def test_backward_propagates_finite_gradients_to_every_family() -> None:
    """Every expected parameter family must receive a finite gradient.

    Runs forward + backward on the tiny model and asserts each of the 11
    parameter families has at least one member whose ``.grad`` is finite and
    non-trivial (not all-zero), confirming the tied output projection actually
    routes gradient into ``token_embedding.weight``.
    """
    model = _build_tiny_model()
    ids = torch.tensor([[1, 5, 9, 42, 7, 0]])
    target = torch.tensor([[5, 9, 42, 7, 0, 1]])

    logits = model(ids)
    # Next-token cross-entropy averaged over all positions.
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, TINY_CONFIG.vocab_size),
        target.reshape(-1),
    )
    loss.backward()

    family_re = re.compile(r"blocks\.\d+\.")

    # Collect the set of families (with the layer index normalised to {i}) that
    # are present in the model. ``param.grad`` is ``Optional[Tensor]``.
    families_present: dict[str, Any] = {}
    for name, param in model.named_parameters():
        family = family_re.sub("blocks.{i}.", name)
        families_present[family] = param.grad

    # The model is bias-free and has no RoPE parameters; every named parameter
    # belongs to one of the 11 ratified families.
    assert len(families_present) == 11

    for family, grad in families_present.items():
        assert grad is not None, f"{family} received no gradient"
        assert torch.isfinite(grad).all(), f"{family} has non-finite gradients"
        # The tied output projection routes gradient into the embedding through
        # a real path; assert at least one entry moved off zero.
        assert not torch.equal(grad, torch.zeros_like(grad)), (
            f"{family} gradient is exactly zero everywhere"
        )


def test_no_bias_or_rope_keys_in_state_dict_after_backward() -> None:
    """Backward must not introduce bias parameters or RoPE cache keys."""
    model = _build_tiny_model()
    ids = torch.tensor([[1, 5, 9, 42, 7, 0]])
    target = torch.tensor([[5, 9, 42, 7, 0, 1]])

    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, TINY_CONFIG.vocab_size),
        target.reshape(-1),
    )
    loss.backward()

    keys = list(model.state_dict())
    assert not any("bias" in k for k in keys)
    assert not any("rope" in k.lower() for k in keys)
    assert not any("inv_freq" in k for k in keys)
    assert not any("cos" in k for k in keys)
    assert not any("sin" in k for k in keys)
    assert not any("cached" in k.lower() for k in keys)
