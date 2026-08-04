"""Derive the corrected D0 tokenizer (Decision 0012, Resolution 1).

This is the deterministic transformation that produces the corrected D0
tokenizer artifact from the rejected upstream GPT-NeoX ``tokenizer.json``.

Frozen derivation rule (Decision 0012, ratified Resolution 1):

- Preserve every existing token id in ``0..50256`` without renumbering.
- Remove every ``added_tokens`` entry whose ``id`` is greater than ``50256``.
- Preserve ``<|endoftext|>`` at id 0.
- Do not configure a padding token (the ``padding`` field remains ``null``).
- Preserve the existing BPE model, normalizer, pre-tokenizer (ByteLevel,
  ``add_prefix_space=false``), post-processor, and decoder unchanged.

Required postconditions (verified before the artifact is published):

- ``get_vocab_size(with_added_tokens=True) == 50257``
- every encoding result satisfies ``0 <= token_id <= 50256``
- ``padding is None`` and ``truncation is None``
- ``token_to_id("<|endoftext|>") == 0``

Removing the high-id whitespace-run tokens (ids 50257..50276) causes those
strings to fall back to ordinary byte-level BPE segmentation. That behavioral
change is represented by new golden vectors, not treated as equivalent to the
upstream tokenizer.

Usage::

    uv run --extra d0-data python scripts/derive_d0_tokenizer.py \\
        --source-cache .cache/d0-tokenizer-original \\
        --output-dir tokenizers/gpt-neox-corrected-v1 \\
        --record-dir experiments/d0/tokenizer-amendment-evidence

The script is idempotent: running it twice on the same source produces
byte-identical output and the same digests. It refuses to overwrite a
pre-existing output whose digest differs (fail-closed).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Frozen identities of the rejected source artifact (Decision 0012).
REJECTED_TOKENIZER_JSON_SHA256 = "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"
REJECTED_MANIFEST_SHA256 = "eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c"

# Frozen postconditions.
REQUIRED_VOCAB_SIZE = 50257
MAX_REACHABLE_ID = 50256
BOUNDARY_TOKEN = "<|endoftext|>"
BOUNDARY_TOKEN_ID = 0

# Canonical serialization: 2-space indent (matches upstream style), unicode
# preserved, no NaN, trailing newline. Key order is preserved as-is from the
# source (json object key order is not semantically meaningful, and preserving
# it keeps the diff against the upstream artifact minimal and reviewable).


def canonical_serialize(spec: Mapping[str, Any]) -> bytes:
    return json.dumps(spec, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


class DerivationError(RuntimeError):
    """The derivation transform or a postcondition check failed."""


@dataclass(frozen=True, slots=True)
class DerivationResult:
    """Content-addressed result of the derivation."""

    output_bytes: bytes
    output_sha256: str
    output_size: int
    added_tokens_before: int
    added_tokens_after: int
    dropped_ids: tuple[int, ...]
    kept_ids: tuple[int, ...]


def derive(spec: Mapping[str, Any]) -> DerivationResult:
    """Apply the frozen derivation rule and return content-addressed output."""

    if "added_tokens" not in spec:
        raise DerivationError("source tokenizer.json has no added_tokens field")
    added_before = list(spec["added_tokens"])
    for index, entry in enumerate(added_before):
        if not isinstance(entry, dict) or "id" not in entry:
            raise DerivationError(f"added_tokens[{index}] is malformed")

    derived = copy.deepcopy(dict(spec))
    kept = [entry for entry in added_before if int(entry["id"]) <= MAX_REACHABLE_ID]
    dropped = [int(entry["id"]) for entry in added_before if int(entry["id"]) > MAX_REACHABLE_ID]
    derived["added_tokens"] = kept

    output = canonical_serialize(derived)
    return DerivationResult(
        output_bytes=output,
        output_sha256=hashlib.sha256(output).hexdigest(),
        output_size=len(output),
        added_tokens_before=len(added_before),
        added_tokens_after=len(kept),
        dropped_ids=tuple(sorted(dropped)),
        kept_ids=tuple(sorted(int(e["id"]) for e in kept)),
    )


def verify_source_tokenizer(path: Path) -> None:
    """Confirm the source is the exact rejected artifact."""

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != REJECTED_TOKENIZER_JSON_SHA256:
        raise DerivationError(
            f"source tokenizer.json digest mismatch: expected "
            f"{REJECTED_TOKENIZER_JSON_SHA256}, got {digest}"
        )


def verify_postconditions(output_bytes: bytes) -> None:
    """Load the derived bytes through the real tokenizers library and assert."""

    try:
        tokenizers_module = importlib.import_module("tokenizers")
    except ModuleNotFoundError as exc:
        raise DerivationError(
            "postcondition verification requires the optional 'd0-data' extra; "
            "run with: uv run --extra d0-data python scripts/derive_d0_tokenizer.py ..."
        ) from exc
    # Resolve via getattr so mypy does not chase the repo's top-level
    # ``tokenizers/`` namespace directory (which shadows the installed package
    # when the extra is absent). ``Tokenizer`` is a C-extension class.
    tokenizer_factory = getattr(tokenizers_module, "Tokenizer", None)
    if tokenizer_factory is None:
        raise DerivationError(
            "postcondition verification requires the optional 'd0-data' extra; "
            "the installed 'tokenizers' module has no Tokenizer attribute"
        )
    tokenizer = tokenizer_factory.from_buffer(output_bytes)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab_size != REQUIRED_VOCAB_SIZE:
        raise DerivationError(
            f"postcondition failed: get_vocab_size == {vocab_size}, expected {REQUIRED_VOCAB_SIZE}"
        )
    if tokenizer.padding is not None:
        raise DerivationError("postcondition failed: padding is not None")
    if tokenizer.truncation is not None:
        raise DerivationError("postcondition failed: truncation is not None")
    boundary = tokenizer.token_to_id(BOUNDARY_TOKEN)
    if boundary != BOUNDARY_TOKEN_ID:
        raise DerivationError(
            f"postcondition failed: token_to_id({BOUNDARY_TOKEN!r}) == "
            f"{boundary!r}, expected {BOUNDARY_TOKEN_ID}"
        )
    # Probe that no ordinary input produces an out-of-range id.
    probes = [
        "",
        "hello",
        " hello",
        "  hello",
        "   hello",
        "    hello",
        "     hello",
        "hello\nworld",
        "café",
        "مرحبا",
        "你好",
        "🙂",
        "a" + " " * 30 + "b",
        "          ",
        "\t\t",
    ]
    for probe in probes:
        ids = tokenizer.encode(probe, add_special_tokens=False).ids
        for token_id in ids:
            if not 0 <= token_id <= MAX_REACHABLE_ID:
                raise DerivationError(
                    f"postcondition failed: probe {probe!r} produced id "
                    f"{token_id} outside [0, {MAX_REACHABLE_ID}]"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive the corrected D0 tokenizer (Decision 0012, Resolution 1)."
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path(".cache/d0-tokenizer-original"),
        help="Directory containing the descriptor-verified rejected tokenizer.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tokenizers" / "gpt-neox-corrected-v1",
        help="Directory to write the derived tokenizer.json into.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=ROOT / "experiments" / "d0" / "tokenizer-amendment-evidence",
        help="Directory to write the execution record into.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a pre-existing output even if its digest differs.",
    )
    args = parser.parse_args(argv)

    source_path = args.source_cache / "tokenizer.json"
    if not source_path.is_file():
        raise DerivationError(
            f"source tokenizer.json not found at {source_path}; acquire the rejected "
            f"inventory first via acquire_source_inventory"
        )
    verify_source_tokenizer(source_path)

    spec = json.loads(source_path.read_bytes())
    result = derive(spec)
    verify_postconditions(result.output_bytes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "tokenizer.json"
    if output_path.exists():
        existing = output_path.read_bytes()
        if hashlib.sha256(existing).hexdigest() != result.output_sha256 and not args.force:
            raise DerivationError(
                f"refusing to overwrite {output_path}: existing digest differs from "
                f"derived digest (use --force to override)"
            )
    output_path.write_bytes(result.output_bytes)

    record = {
        "schema_version": "expertforge-d0-tokenizer-derivation-record/1",
        "decision": "doctrine/decisions/0012-d0-tokenizer-vocabulary-divergence.md",
        "resolution": 1,
        "transformation_rule": (
            "remove every added_tokens entry whose id > 50256; preserve all other "
            "fields unchanged (BPE model, normalizer, ByteLevel pre-tokenizer with "
            "add_prefix_space=false, post_processor, decoder); padding remains null"
        ),
        "source": {
            "artifact": "EleutherAI/gpt-neox-20b tokenizer.json (rejected for D0)",
            "tokenizer_json_sha256": REJECTED_TOKENIZER_JSON_SHA256,
            "manifest_sha256": REJECTED_MANIFEST_SHA256,
        },
        "output": {
            "tokenizer_json_path": output_path.relative_to(ROOT).as_posix(),
            "tokenizer_json_sha256": result.output_sha256,
            "tokenizer_json_size_bytes": result.output_size,
            "vocabulary_size": REQUIRED_VOCAB_SIZE,
            "max_reachable_id": MAX_REACHABLE_ID,
        },
        "added_tokens_before": result.added_tokens_before,
        "added_tokens_after": result.added_tokens_after,
        "dropped_ids": list(result.dropped_ids),
        "kept_ids": list(result.kept_ids),
        "postconditions_verified": [
            "get_vocab_size(with_added_tokens=True) == 50257",
            "padding is None",
            "truncation is None",
            "token_to_id('<|endoftext|>') == 0",
            "every probed encoding id in [0, 50256]",
        ],
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    args.record_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.record_dir / "derivation-record.json"
    record_bytes = canonical_serialize(record)
    record_path.write_bytes(record_bytes)

    print(f"derived tokenizer.json -> {output_path}")
    print(f"  sha256 = {result.output_sha256}")
    print(f"  size   = {result.output_size} bytes")
    print(f"  added_tokens: {result.added_tokens_before} -> {result.added_tokens_after}")
    print(f"  dropped ids ({len(result.dropped_ids)}): {list(result.dropped_ids)}")
    print(f"  kept ids ({len(result.kept_ids)}): {list(result.kept_ids)}")
    print(f"derivation record -> {record_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
