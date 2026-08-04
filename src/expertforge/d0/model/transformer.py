"""D0 decoder block and full decoder-only Transformer model.

Assembles the ratified primitives into the D0 architecture:

- Pre-norm RMSNorm -> attention -> residual.
- Pre-norm RMSNorm -> SwiGLU -> residual.
- Final pre-norm RMSNorm followed by the tied output projection.

The forward pass returns logits of shape ``(batch, seq, vocab_size)``.

Parameter names follow the inventory exactly:

- ``token_embedding.weight``
- ``blocks.{i}.attention_norm.weight``
- ``blocks.{i}.attention.{q,k,v,out}_proj.weight``
- ``blocks.{i}.ffn_norm.weight``
- ``blocks.{i}.ffn.{gate,up,down}_proj.weight``
- ``final_norm.weight``
"""

from __future__ import annotations

from expertforge.d0.model._torch import require_torch as _require_torch

_require_torch()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from expertforge.d0.model.attention import CausalSelfAttention  # noqa: E402
from expertforge.d0.model.config import D0ModelConfig  # noqa: E402
from expertforge.d0.model.embedding import TokenEmbedding  # noqa: E402
from expertforge.d0.model.rmsnorm import RMSNorm  # noqa: E402
from expertforge.d0.model.swiglu import SwiGLU  # noqa: E402

__all__ = [
    "D0DecoderBlock",
    "D0Model",
]


class D0DecoderBlock(nn.Module):
    """Standard pre-norm Transformer decoder block.

    Order::

        x -> attention_norm -> attention -> +x
          -> ffn_norm      -> ffn       -> +residual
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        ffn_dim: int,
        epsilon: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(dim, epsilon=epsilon)
        self.attention = CausalSelfAttention(dim, n_heads, head_dim, rope_base)
        self.ffn_norm = RMSNorm(dim, epsilon=epsilon)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class D0Model(nn.Module):
    """Full D0 decoder-only Transformer.

    Parameters
    ----------
    config:
        Frozen architectural configuration. Use
        :func:`expertforge.d0.model.config.qualification_config` or
        :func:`~expertforge.d0.model.config.canonical_config` to obtain the
        two ratified profiles.
    """

    def __init__(self, config: D0ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [
                D0DecoderBlock(
                    dim=config.dim,
                    n_heads=config.n_heads,
                    head_dim=config.head_dim,
                    ffn_dim=config.ffn_dim,
                    epsilon=config.rmsnorm_epsilon,
                    rope_base=config.rope_base,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.dim, epsilon=config.rmsnorm_epsilon)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # Tied output projection: x @ embedding.weight.T (no new parameter).
        logits: torch.Tensor = x @ self.token_embedding.weight.t()
        return logits
