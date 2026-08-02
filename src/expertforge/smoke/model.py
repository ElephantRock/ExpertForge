"""Minimal causal decoder for the Milestone 0 smoke gate (Issue #14).

This is an **infrastructure gate toy model**, not a reusable training framework
or D0 precursor (amendment B). It exists to exercise forward, backward, update,
validation, generation, checkpoint, and restoration through the real substrate
subsystems.

Design (every declared configuration dimension controls real behavior — review
finding F9):

- token embedding ``embed`` ``[V, dim]``;
- a stack of ``n_layers`` identical causal self-attention blocks, each with a
  fused ``qkv`` projection ``[dim, 3*dim]`` factored into ``n_heads`` heads,
  scaled dot-product causal attention, and an output projection ``attn_out``
  ``[dim, dim]``;
- each block also has an ordinary (ReLU) feed-forward block: ``ffn_in``
  ``[dim, ffn_dim]`` and ``ffn_out`` ``[ffn_dim, dim]`` (SwiGLU omitted);
- an output projection ``lm_head`` ``[dim, V]``.

There is no dropout and no nondeterministic behavior. Canonical state is float32
throughout. ``n_layers`` controls the block stack depth; ``n_heads`` controls the
attention head factorization; ``dim`` must be divisible by ``n_heads``.
Deterministic initialization is drawn from the :class:`~expertforge.rng.manager.
RngManager` generator so the model identity is reproducible.

The backward pass is **analytical** (verified against finite differences in
float64; finite differences are used ONLY as a fast-test gradient check, never as
the gate's update path — review correction #3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from expertforge.config.models import ModelConfig

__all__ = [
    "MODEL_ARCHITECTURE",
    "ForwardCache",
    "SmokeModel",
    "softmax",
]

MODEL_ARCHITECTURE = "smoke-toy-causal-decoder"


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax over the last axis."""
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    out: np.ndarray = e / e.sum(axis=-1, keepdims=True)
    return out


@dataclass
class BlockCache:
    """Intermediate activations for one transformer block (for backprop)."""

    h_in: np.ndarray
    q: np.ndarray
    q_act: np.ndarray
    k_act: np.ndarray
    v_act: np.ndarray
    qh: np.ndarray
    kh: np.ndarray
    vh: np.ndarray
    attn: np.ndarray
    ctx: np.ndarray
    h_attn: np.ndarray
    pre: np.ndarray
    ffn: np.ndarray
    h_out: np.ndarray


@dataclass
class ForwardCache:
    """Intermediate activations retained for the analytical backward pass."""

    x: np.ndarray
    h0: np.ndarray
    blocks: list[BlockCache] = field(default_factory=list)
    h2: np.ndarray = field(default_factory=lambda: np.array([]))
    logits: np.ndarray = field(default_factory=lambda: np.array([]))


class SmokeModel:
    """The minimal causal decoder.

    Parameters are held as a ``dict[str, np.ndarray]`` of float32 C-contiguous
    arrays, keyed by stable logical names: ``embed``, ``lm_head``, and per-block
    ``blocks.<i>.<param>``. The model is mutable in place; capture and restore
    happen through :mod:`expertforge.smoke.state`.
    """

    def __init__(self, config: ModelConfig) -> None:
        if config.dim % config.n_heads != 0:
            raise ValueError("model.dim must be divisible by model.n_heads")
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.ffn_dim = config.ffn_dim
        self.head_dim = config.dim // config.n_heads
        self.parameters: dict[str, np.ndarray] = {}
        self._initialized = False

    # -- construction -----------------------------------------------------

    @classmethod
    def uninitialized(cls, config: ModelConfig) -> SmokeModel:
        return cls(config)

    def initialize(self, generator: np.random.Generator) -> None:
        """Initialize parameters deterministically from ``generator``.

        All arrays are float32, C-contiguous. Initialization is a pure function
        of the generator state, so two runs with identical RNG produce identical
        parameters (amendment B).
        """
        v = FIXTURE_VOCAB_SIZE
        scale = 0.3
        params: dict[str, np.ndarray] = {
            "embed": _f32(generator.standard_normal((v, self.dim)) * scale),
            "lm_head": _f32(generator.standard_normal((self.dim, v)) * scale),
        }
        for i in range(self.n_layers):
            prefix = f"blocks.{i}"
            params[f"{prefix}.qkv"] = _f32(
                generator.standard_normal((self.dim, 3 * self.dim)) * scale
            )
            params[f"{prefix}.attn_out"] = _f32(
                generator.standard_normal((self.dim, self.dim)) * scale
            )
            params[f"{prefix}.ffn_in"] = _f32(
                generator.standard_normal((self.dim, self.ffn_dim)) * scale
            )
            params[f"{prefix}.ffn_out"] = _f32(
                generator.standard_normal((self.ffn_dim, self.dim)) * scale
            )
        self.parameters = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in params.items()}
        self._initialized = True

    def assert_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("SmokeModel is not initialized")

    # -- introspection ----------------------------------------------------

    def parameter_count(self) -> int:
        """Exact parameter count computed from the actual arrays."""
        return int(sum(arr.size for arr in self.parameters.values()))

    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.parameters.keys())

    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        return {k: tuple(arr.shape) for k, arr in self.parameters.items()}

    def _block_param(self, layer: int, name: str) -> np.ndarray:
        return self.parameters[f"blocks.{layer}.{name}"]

    # -- forward / backward ----------------------------------------------

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, ForwardCache]:
        """Forward pass. ``x`` is int token ids ``[B, T]``; returns logits ``[B,T,V]``."""
        self.assert_initialized()
        B, T = x.shape
        cache = ForwardCache(x=x, h0=np.array([]))
        h = self.parameters["embed"][x]  # [B,T,dim]
        cache.h0 = h
        cache.blocks = []
        for layer in range(self.n_layers):
            bc = self._forward_block(h, layer, T)
            cache.blocks.append(bc)
            h = bc.h_out
        h2 = h
        cache.h2 = h2
        logits = h2 @ self.parameters["lm_head"]  # [B,T,V]
        cache.logits = logits
        return logits, cache

    def _forward_block(self, h: np.ndarray, layer: int, T: int) -> BlockCache:
        dim = self.dim
        n_h = self.n_heads
        hd = self.head_dim
        qkv = self._block_param(layer, "qkv")
        attn_out = self._block_param(layer, "attn_out")
        ffn_in = self._block_param(layer, "ffn_in")
        ffn_out = self._block_param(layer, "ffn_out")
        q = h @ qkv  # [B,T,3dim]
        q_act, k_act, v_act = q[..., :dim], q[..., dim : 2 * dim], q[..., 2 * dim :]
        # Multi-head attention: reshape [B,T,dim] -> [B,n_h,T,hd].
        qh = q_act.reshape(*q_act.shape[:-1], n_h, hd).transpose(0, 2, 1, 3)
        kh = k_act.reshape(*k_act.shape[:-1], n_h, hd).transpose(0, 2, 1, 3)
        vh = v_act.reshape(*v_act.shape[:-1], n_h, hd).transpose(0, 2, 1, 3)
        scores = (qh @ kh.transpose(0, 1, 3, 2)) / np.sqrt(hd)  # [B,n_h,T,T]
        causal = np.triu(np.full((T, T), _MASK_VALUE, dtype=np.float32), k=1)
        attn = softmax(scores + causal)  # [B,n_h,T,T]
        ctx_h = attn @ vh  # [B,n_h,T,hd]
        # Merge heads back: [B,n_h,T,hd] -> [B,T,dim]
        ctx = ctx_h.transpose(0, 2, 1, 3).reshape(*q_act.shape[:-1], dim)
        h_attn = ctx @ attn_out  # [B,T,dim]
        pre = h_attn @ ffn_in  # [B,T,ffn_dim]
        ffn = np.maximum(pre, 0.0)  # ReLU
        h_out = ffn @ ffn_out  # [B,T,dim]
        return BlockCache(
            h_in=h,
            q=q,
            q_act=q_act,
            k_act=k_act,
            v_act=v_act,
            qh=qh,
            kh=kh,
            vh=vh,
            attn=attn,
            ctx=ctx,
            h_attn=h_attn,
            pre=pre,
            ffn=ffn,
            h_out=h_out,
        )

    def backward(self, cache: ForwardCache, targets: np.ndarray) -> dict[str, np.ndarray]:
        """Analytical backward returning per-parameter gradients.

        Loss is mean cross-entropy over all ``B*T`` positions. Gradients are
        float32, C-contiguous, and shape-matched to ``self.parameters``.
        """
        self.assert_initialized()
        logits = cache.logits
        B, T, V = logits.shape
        n_h = self.n_heads
        hd = self.head_dim
        probs = softmax(logits)
        dlogits = probs.copy()
        b_idx = np.arange(B)[:, None]
        t_idx = np.arange(T)[None, :]
        dlogits[b_idx, t_idx, targets] -= 1.0
        dlogits /= B * T

        dh = dlogits @ self.parameters["lm_head"].T  # [B,T,dim]
        dlm_head = np.tensordot(cache.h2, dlogits, axes=([0, 1], [0, 1]))
        grads: dict[str, np.ndarray] = {
            "embed": np.zeros_like(self.parameters["embed"]),
            "lm_head": _contig_f32(dlm_head),
        }
        # Backprop through the block stack in reverse.
        dh0 = dh
        for layer in reversed(range(self.n_layers)):
            dh0 = self._backward_block(cache.blocks[layer], layer, dh0, n_h, hd, grads)
        # Embedding gradient: dh0 is the gradient w.r.t. the embedding output h0;
        # scatter-add it into the looked-up token rows.
        np.add.at(grads["embed"], cache.x, dh0)
        # Canonicalize.
        return {k: _contig_f32(v) for k, v in grads.items()}

    def _backward_block(
        self,
        bc: BlockCache,
        layer: int,
        dh_out: np.ndarray,
        n_h: int,
        hd: int,
        grads: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Backprop one block; returns the gradient w.r.t. the block input."""
        attn_out = self._block_param(layer, "attn_out")
        ffn_in = self._block_param(layer, "ffn_in")
        ffn_out = self._block_param(layer, "ffn_out")
        qkv = self._block_param(layer, "qkv")
        prefix = f"blocks.{layer}"

        dffn = dh_out @ ffn_out.T
        grads[f"{prefix}.ffn_out"] = np.tensordot(bc.ffn, dh_out, axes=([0, 1], [0, 1]))
        dpre = dffn * (bc.pre > 0).astype(np.float32)
        grads[f"{prefix}.ffn_in"] = np.tensordot(bc.h_attn, dpre, axes=([0, 1], [0, 1]))
        dh_attn = dpre @ ffn_in.T
        # attn_out: h_attn = ctx @ attn_out
        grads[f"{prefix}.attn_out"] = np.tensordot(bc.ctx, dh_attn, axes=([0, 1], [0, 1]))
        dctx = dh_attn @ attn_out.T  # [B,T,dim]
        # Unmerge heads: [B,T,dim] -> [B,n_h,T,hd]
        dctx_h = dctx.reshape(*dctx.shape[:-1], n_h, hd).transpose(0, 2, 1, 3)
        # ctx_h = attn @ vh  -> dattn = dctx_h @ vh^T ; dvh = attn^T @ dctx_h
        dattn = dctx_h @ bc.vh.transpose(0, 1, 3, 2)
        dvh = bc.attn.transpose(0, 1, 3, 2) @ dctx_h
        # softmax backward per (head, row)
        dscores = bc.attn * (dattn - (dattn * bc.attn).sum(axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(hd)
        dqh = dscores @ bc.kh
        dkh = dscores.transpose(0, 1, 3, 2) @ bc.qh
        # Merge head grads back to [B,T,dim]
        dq_act = dqh.transpose(0, 2, 1, 3).reshape(*bc.q_act.shape)
        dk_act = dkh.transpose(0, 2, 1, 3).reshape(*bc.k_act.shape)
        dv_act = dvh.transpose(0, 2, 1, 3).reshape(*bc.v_act.shape)
        dq_act_total = np.concatenate([dq_act, dk_act, dv_act], axis=-1)
        grads[f"{prefix}.qkv"] = np.tensordot(bc.h_in, dq_act_total, axes=([0, 1], [0, 1]))
        dh_in: np.ndarray = dq_act_total @ qkv.T
        return dh_in

    # -- generation -------------------------------------------------------

    def generate(self, prompt: np.ndarray, n_new: int) -> np.ndarray:
        """Deterministic greedy generation (no RNG consumption).

        Review correction #9, option 1: greedy argmax sampling. The RNG probe
        is a separate, independent comparison that never runs through generation.
        """
        self.assert_initialized()
        out = prompt.astype(np.int64).copy()
        for _ in range(n_new):
            logits, _ = self.forward(out[None, :])  # [1,T,V]
            next_id = int(np.argmax(logits[0, -1, :]))
            out = np.concatenate([out, np.array([next_id], dtype=np.int64)])
        return out


def _f32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _contig_f32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


# Large negative value for the causal mask (float32-safe; not -inf so masked
# softmax rows remain finite and stable).
_MASK_VALUE: float = -1.0e9

# The fixture tokenizer vocabulary. Hard-coded to match the committed
# tokenizer.json (smoke-printable16-v1, vocab_size=16). The report records this.
FIXTURE_VOCAB_SIZE: int = 16
