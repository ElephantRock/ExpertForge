"""Deterministic sequential data cursor for the smoke gate (Issue #14).

Reads the committed 16-symbol fixture corpus, produces sequential batches, and
carries a fully reconstructible continuation cursor. Semantics are explicit and
mutually consistent (review clarification 5152259715 #1):

- ``length`` (tokens): the total number of trainable tokens in the corpus
  (corpus bytes interpreted through the byte-mapped 16-symbol tokenizer, one
  token per byte). The dataset identity's ``length`` is this token count.
- ``position`` (cursor): the index of the NEXT unconsumed logical token in the
  flattened corpus. Sequential sampling: tokens are consumed in order; the
  permutation is reconstructible solely from the persisted dataset identity,
  sampler type/version, and epoch.
- ``accepted_samples``: the cumulative number of training samples (sequences)
  accepted by the consumer so far. Each update consumes ``batch_size`` samples,
  so ``accepted_samples`` advances by ``batch_size`` per update.
- ``accepted_sequences``: same as ``accepted_samples`` here (one sample == one
  sequence; no packing). Tracked separately because the contract distinguishes
  samples from sequences for packed-data futures.

Update-boundary invariants (review correction #4), one microstep per update:

    global_update      == completed_microsteps
    accumulation_position == 0
    processed_tokens   == global_update * tokens_per_update
    accepted_samples   == global_update * batch_size
    accepted_sequences == accepted_samples
    position           == (accepted_samples * sequence_length) mod length

``DataIdentity.data_config_digest`` independently hashes the canonical
data-continuation contract (sampler type/version, batch/sequence/drop_last
policy, and the corpus identity) — it does NOT reuse the corpus-byte digest
(clarification #1).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from expertforge.smoke import fixtures

__all__ = [
    "SMOKE_DATA_TOKENIZER_IDENTITY",
    "SmokeDataCursor",
    "compute_data_config_digest_hex",
    "load_corpus_tokens",
]

# The byte-mapped tokenizer identity recorded on the data identity.
SMOKE_DATA_TOKENIZER_IDENTITY = "smoke-printable16-v1"


def load_corpus_tokens() -> np.ndarray:
    """The corpus bytes mapped through the byte-mapped tokenizer (one token/byte).

    The fixture tokenizer maps each of the 16 symbols to token id 0..15 equal to
    its position in the symbol table. Because the symbols are exactly the bytes
    present in the corpus, the token id for a byte is its index in the committed
    ``tokenizer.json`` symbol table.
    """
    definition = fixtures.load_tokenizer_definition()
    symbol_to_token: dict[str, int] = {s: int(i) for i, s in enumerate(definition["symbols"])}
    corpus = fixtures.corpus_bytes().decode("utf-8")
    return np.array([symbol_to_token[ch] for ch in corpus], dtype=np.int64)


# The number of tokens reserved as a held-out validation window. Validation uses
# a DISJOINT suffix of the corpus that the sequential training cursor never
# consumes (review finding F2): training consumes the first
# ``total_training_tokens`` tokens; validation reads a fixed-length window at the
# corpus end that does not overlap that prefix.
VALIDATION_WINDOW_TOKENS = 16


def validation_tokens(total_training_tokens: int) -> np.ndarray:
    """A held-out validation slice DISJOINT from the consumed training prefix.

    Training consumes ``total_training_tokens`` tokens sequentially from
    position 0. This returns a ``[seq_len]``-shaped validation sequence taken
    from a corpus suffix that starts strictly after the training prefix, so the
    validation data never overlaps the training data (finding F2). The slice is
    deterministic and reproducible.
    """
    tokens = load_corpus_tokens()
    length = int(tokens.shape[0])
    if total_training_tokens + VALIDATION_WINDOW_TOKENS > length:
        raise ValueError(
            f"corpus length {length} cannot hold {total_training_tokens} training "
            f"tokens plus a {VALIDATION_WINDOW_TOKENS}-token disjoint validation window"
        )
    start = length - VALIDATION_WINDOW_TOKENS
    return tokens[start : start + VALIDATION_WINDOW_TOKENS]


def compute_data_config_digest_hex(
    *,
    dataset_digest_hex: str,
    sampler_type: str,
    sampler_version: int,
    batch_size: int,
    sequence_length: int,
    drop_last: bool,
    tokenizer_identity: str,
) -> str:
    """Independent SHA-256 over the canonical data-continuation contract.

    This is a SEPARATE digest from the corpus-byte digest (clarification #1).
    It captures the sampler contract that makes the cursor reconstructible.
    """
    contract: dict[str, Any] = {
        "schema": "expertforge.smoke.data-config-contract",
        "schema_version": 1,
        "dataset_digest": dataset_digest_hex,
        "sampler_type": sampler_type,
        "sampler_version": sampler_version,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "drop_last": drop_last,
        "tokenizer_identity": tokenizer_identity,
    }
    payload = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class SmokeDataCursor:
    """The live, mutable sequential data cursor.

    Constructed once from the corpus + sampler contract; advanced per batch by
    :meth:`next_batch`. The frozen :class:`~expertforge.checkpoints.models.DataCursor`
    snapshot is produced by :meth:`snapshot` for checkpoint capture and consumed
    by :meth:`from_snapshot` for restore.
    """

    tokens: np.ndarray  # the full flattened corpus tokens (int64)
    batch_size: int
    sequence_length: int
    drop_last: bool
    epoch: int = 0
    position: int = 0
    accepted_samples: int = 0
    accepted_sequences: int = 0

    @classmethod
    def create(
        cls, batch_size: int, sequence_length: int, drop_last: bool = True
    ) -> SmokeDataCursor:
        tokens = load_corpus_tokens()
        return cls(
            tokens=tokens,
            batch_size=batch_size,
            sequence_length=sequence_length,
            drop_last=drop_last,
        )

    @property
    def length(self) -> int:
        """Total trainable tokens in the corpus."""
        return int(self.tokens.shape[0])

    def tokens_per_update(self) -> int:
        return self.batch_size * self.sequence_length

    def next_batch(self) -> np.ndarray:
        """Return the next batch as ``[batch_size, sequence_length]`` int tokens.

        Sequential sampling with wrap-around: when the corpus is exhausted, the
        epoch increments and the position wraps to 0. The cursor advances by
        ``batch_size * sequence_length`` consumed tokens per call.
        """
        n = self.batch_size * self.sequence_length
        if n > self.length:
            raise ValueError(
                f"batch_size*sequence_length ({n}) exceeds corpus length ({self.length})"
            )
        out = np.empty((self.batch_size, self.sequence_length), dtype=np.int64)
        # Fill sequentially from the flattened corpus, wrapping at the end.
        for i in range(self.batch_size):
            for j in range(self.sequence_length):
                out[i, j] = self.tokens[self.position % self.length]
                self.position += 1
                if self.position >= self.length:
                    self.position = 0
                    self.epoch += 1
        self.accepted_samples += self.batch_size
        self.accepted_sequences += self.batch_size
        return out

    # -- checkpoint snapshot / restore -----------------------------------

    def snapshot(self) -> tuple[Any, Any]:
        """Return ``(DataCursor, DataIdentity)`` for checkpoint capture.

        Imports the frozen checkpoint models lazily to keep this module free of
        import-time Pydantic work.
        """
        from expertforge.checkpoints.models import DataCursor, DataIdentity

        data_config_digest = compute_data_config_digest_hex(
            dataset_digest_hex=fixtures.corpus_digest_hex(),
            sampler_type="sequential",
            sampler_version=1,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            drop_last=self.drop_last,
            tokenizer_identity=SMOKE_DATA_TOKENIZER_IDENTITY,
        )
        identity = DataIdentity(
            dataset_digest=fixtures.corpus_digest_hex(),
            split="train",
            length=self.length,
            preprocessing_identity="byte_map.v1",
            tokenizer_identity=SMOKE_DATA_TOKENIZER_IDENTITY,
            packing_policy="no_packing",
            sequence_policy=f"fixed_{self.sequence_length}",
            shard_selection="single",
            data_config_digest=data_config_digest,
        )
        cursor = DataCursor(
            sampler_type="sequential",
            sampler_version=1,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            drop_last=self.drop_last,
            epoch=self.epoch,
            position=self.position,
            permutation_seed=None,
            accepted_samples=self.accepted_samples,
            accepted_sequences=self.accepted_sequences,
        )
        return cursor, identity

    @classmethod
    def restore_from(
        cls,
        cursor: Any,
        batch_size: int,
        sequence_length: int,
        drop_last: bool,
    ) -> SmokeDataCursor:
        """Reconstruct a live cursor from a restored :class:`DataCursor`."""
        tokens = load_corpus_tokens()
        return cls(
            tokens=tokens,
            batch_size=batch_size,
            sequence_length=sequence_length,
            drop_last=drop_last,
            epoch=int(cursor.epoch),
            position=int(cursor.position),
            accepted_samples=int(cursor.accepted_samples),
            accepted_sequences=int(cursor.accepted_sequences),
        )
