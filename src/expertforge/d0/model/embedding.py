"""Tied token embedding and output projection.

The D0 contract ties the token embedding and the output projection into a single
``[vocab_size, dim]`` parameter. The output projection is ``x @ weight.T`` and
introduces no separate parameter (see the parameter inventory alias
``output_head.weight``).

The embedding weight is stored directly on this module as ``self.weight`` (an
:class:`~torch.nn.Parameter`) so that the owning model exposes it under the
state-dict name ``token_embedding.weight`` exactly as the parameter inventory
requires (no intermediate ``embedding`` submodule).
"""

from __future__ import annotations

from expertforge.d0.model._torch import require_torch as _require_torch

_require_torch()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

__all__ = ["TokenEmbedding"]


class TokenEmbedding(nn.Module):
    """Token embedding lookup sharing weight with the output projection.

    Parameters
    ----------
    vocab_size:
        Number of tokens in the vocabulary.
    dim:
        Embedding (model) width.
    """

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        # Direct parameter (not an nn.Embedding submodule) so the state-dict
        # key is ``token_embedding.weight`` per the parameter inventory.
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(token_ids, self.weight)
