"""Dual-tokenizer corpus scan for the D0.0 tokenizer-vocabulary amendment (#55).

Runs BOTH the rejected upstream GPT-NeoX tokenizer and the corrected tokenizer
against every retained, normalized, deduplicated document in the frozen D0
FineWeb-Edu ``sample-10BT`` corpus, and produces a content-addressed report
quantifying the divergence.

This is Evidence 7 for Decision 0012 Resolution 1. The amendment cannot be
ratified until this scan executes on a durable host against the verified 14-shard
cache and proves the corrected tokenizer never emits an id above 50256.

The traversal is identical to the contamination scan (``scan_runner``): the same
``iter_parquet_documents`` + ``DedupIndex`` + ``normalize_text`` pipeline visits
the same documents in the same order. The accepted dedup population must match
the source evidence: 9,672,101 physical rows, 403,945 duplicates suppressed,
9,268,156 retained documents. A mismatch fails closed.

Cross-checks recorded in the report:
- dataset manifest digest
- accepted dedup population identity (physical rows, duplicates, retained docs)
- original tokenizer manifest + tokenizer.json digests
- corrected tokenizer manifest + tokenizer.json digests
- scanner source commit (the exact git HEAD this runs from)
- tokenizers version
- D0 split + training/validation order algorithm identities
- report self-digest

Required output fields (per amendment ratification gate):
- physical rows visited; duplicates suppressed; unique documents tokenized
- documents emitting >= 1 original id above 50256
- total out-of-range token occurrences (original)
- per-id counts for ids 50257..50276 (original)
- maximum original emitted id; maximum corrected emitted id
- corrected out-of-range occurrence count (must be zero)
- train/validation document counts and token totals (corrected)
- document terminator counts
- original-vs-corrected token-count delta

Usage (on the durable host)::

    uv run --extra d0-data python scripts/run_d0_tokenizer_corpus_scan.py \\
        --dataset-root .cache/d0-fineweb-edu-sample-10bt \\
        --original-tokenizer-root .cache/d0-tokenizer-original \\
        --corrected-tokenizer-root .cache/d0-tokenizer-corrected \\
        --work-dir experiments/d0/tokenizer-amendment-evidence/corpus-scan \\
        --source-commit <exact 40-char SHA>

The corrected tokenizer.json is regenerated on the host via
``scripts/derive_d0_tokenizer.py`` if absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from expertforge.d0.dedup_index import DedupIndex
from expertforge.d0.normalization import normalize_text
from expertforge.d0.parquet_source import iter_parquet_documents
from expertforge.d0.scan_report import ShardScanStats
from expertforge.d0.source_manifest import load_source_manifest
from expertforge.d0.source_verification import (
    resolve_inventory_path,
    verify_source_file_bytes,
    verify_source_inventory,
)

# Frozen D0.0 split/seed contract values (the canonical home for these is
# src/expertforge/d0/data/identity.py on #53, which rebases onto this amendment
# after merge). Inlined here so the corpus scan is self-contained on #55.
_D0_DATA_SEED = 4657843784274978659
_SPLIT_DOMAIN = b"expertforge-d0-split-v1\x00"
_SPLIT_MODULUS = 1000
_TRAIN_THRESHOLD = 995


def _split_bucket(document_id: str) -> int:
    import hashlib
    import struct

    digest = hashlib.sha256(_SPLIT_DOMAIN + document_id.encode("utf-8")).digest()
    value: int = struct.unpack(">Q", digest[:8])[0]
    return value % _SPLIT_MODULUS


def _is_train(document_id: str) -> bool:
    return _split_bucket(document_id) < _TRAIN_THRESHOLD


ROOT = Path(__file__).resolve().parents[1]

# Frozen source identities.
DATASET_MANIFEST_PATH = ROOT / "data/manifests/d0-fineweb-edu-sample-10bt-source-v1.json"
DATASET_MANIFEST_SHA256 = "85fab524b13c49da78264eb03124ce113395657d6403627e53008aade7eea7d7"
ORIGINAL_TOKENIZER_MANIFEST_PATH = ROOT / "tokenizers/manifests/d0-gpt-neox-v1.json"
ORIGINAL_TOKENIZER_MANIFEST_SHA256 = (
    "eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c"
)
CORRECTED_TOKENIZER_MANIFEST_PATH = ROOT / "tokenizers/manifests/d0-gpt-neox-corrected-v1.json"
CORRECTED_TOKENIZER_MANIFEST_SHA256 = (
    "1142baf61c330a318be6e1b153f6769892a7f5af3b7241363a1d0a1f2ca453e1"
)
MAX_REACHABLE_ID = 50256

# Accepted dedup population cross-check (from the contamination scan evidence).
EXPECTED_PHYSICAL_ROWS = 9672101
EXPECTED_DUPLICATES_SUPPRESSED = 403945
EXPECTED_RETAINED_DOCUMENTS = 9268156


class CorpusScanError(RuntimeError):
    """The dual-tokenizer corpus scan failed."""


@dataclass(frozen=True, slots=True)
class TokenizerHandle:
    """A loaded tokenizer bound to its verified manifest identity."""

    manifest_id: str
    manifest_sha256: str
    tokenizer_json_sha256: str
    impl: Any  # tokenizers.Tokenizer; typed loosely to avoid importing the lib here


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _load_tokenizer_from_cache(
    cache_root: Path, manifest_path: Path, expected_manifest_sha256: str
) -> TokenizerHandle:
    """Verify the tokenizer inventory through the descriptor-bound path and load
    tokenizer.json via from_buffer over the exact verified bytes."""

    manifest = load_source_manifest(
        manifest_path, expected_manifest_sha256=expected_manifest_sha256
    )
    if manifest.kind != "tokenizer":
        raise CorpusScanError(f"manifest {manifest_path} is not a tokenizer manifest")
    verify_source_inventory(cache_root, manifest)
    tokenizer_identity = manifest.file("tokenizer.json")
    tokenizer_path = resolve_inventory_path(cache_root, "tokenizer.json")
    verified = verify_source_file_bytes(tokenizer_path, tokenizer_identity)
    try:
        import importlib

        tokenizers_module = importlib.import_module("tokenizers")
        tokenizer_factory = getattr(tokenizers_module, "Tokenizer", None)
        if tokenizer_factory is None:
            raise CorpusScanError("tokenizers library has no Tokenizer attribute")
        impl = tokenizer_factory.from_buffer(verified.payload)
    except ModuleNotFoundError as exc:
        raise CorpusScanError(
            "the d0-data extra (tokenizers) is required for the corpus scan"
        ) from exc
    return TokenizerHandle(
        manifest_id=manifest.manifest_id,
        manifest_sha256=manifest.manifest_sha256,
        tokenizer_json_sha256=verified.identity.sha256,
        impl=impl,
    )


@dataclass
class ScanAccumulator:
    """Mutable per-scan counters for one tokenizer."""

    documents_tokenized: int = 0
    documents_with_out_of_range: int = 0
    out_of_range_occurrences: int = 0
    per_oor_id_count: dict[int, int] = field(default_factory=dict)
    max_emitted_id: int = -1
    token_count: int = 0
    document_terminators: int = 0
    train_documents: int = 0
    validation_documents: int = 0
    train_tokens: int = 0
    validation_tokens: int = 0


def _scan_one_document(
    handle: TokenizerHandle, document_id: str, normalized_text: str, acc: ScanAccumulator
) -> None:
    """Encode one document under one tokenizer and update the accumulator."""

    try:
        ids = handle.impl.encode(normalized_text, add_special_tokens=False).ids
    except Exception as exc:  # noqa: BLE001 - surface any tokenizer failure
        raise CorpusScanError(f"tokenizer {handle.manifest_id} failed on {document_id}") from exc
    oor_in_doc = False
    for token_id in ids:
        if token_id > acc.max_emitted_id:
            acc.max_emitted_id = token_id
        if token_id > MAX_REACHABLE_ID:
            oor_in_doc = True
            acc.out_of_range_occurrences += 1
            acc.per_oor_id_count[token_id] = acc.per_oor_id_count.get(token_id, 0) + 1
    if oor_in_doc:
        acc.documents_with_out_of_range += 1
    acc.token_count += len(ids)
    # Document form appends the boundary terminator (id 0).
    acc.document_terminators += 1
    acc.token_count += 1
    # Split membership (frozen D0 contract).
    if _is_train(document_id):
        acc.train_documents += 1
        acc.train_tokens += len(ids) + 1
    else:
        acc.validation_documents += 1
        acc.validation_tokens += len(ids) + 1


def execute_dual_tokenizer_scan(
    *,
    dataset_root: Path,
    original_tokenizer_root: Path,
    corrected_tokenizer_root: Path,
    state_database_path: Path,
    report_path: Path,
    scanner_source_commit: str,
    parquet_batch_size: int = 4096,
) -> dict[str, object]:
    """Run both tokenizers over the full dedup population and emit the report."""

    dataset_manifest = load_source_manifest(
        DATASET_MANIFEST_PATH, expected_manifest_sha256=DATASET_MANIFEST_SHA256
    )
    original = _load_tokenizer_from_cache(
        original_tokenizer_root,
        ORIGINAL_TOKENIZER_MANIFEST_PATH,
        ORIGINAL_TOKENIZER_MANIFEST_SHA256,
    )
    corrected = _load_tokenizer_from_cache(
        corrected_tokenizer_root,
        CORRECTED_TOKENIZER_MANIFEST_PATH,
        CORRECTED_TOKENIZER_MANIFEST_SHA256,
    )

    original_acc = ScanAccumulator()
    corrected_acc = ScanAccumulator()
    physical_rows_visited = 0
    duplicates_suppressed = 0

    with DedupIndex(state_database_path) as index:
        verify_source_inventory(dataset_root, dataset_manifest)
        for identity in dataset_manifest.files:
            index.begin_shard(identity.path)
            local_path = resolve_inventory_path(dataset_root, identity.path)
            shard_physical_rows = 0
            shard_accepted = 0
            shard_duplicates = 0
            try:
                for document in iter_parquet_documents(
                    local_path,
                    source_file_path=identity.path,
                    batch_size=parquet_batch_size,
                ):
                    physical_rows_visited += 1
                    shard_physical_rows += 1
                    normalized = normalize_text(document.text)
                    decision = index.register(normalized, document.physical_row_index)
                    if not decision.accepted:
                        duplicates_suppressed += 1
                        shard_duplicates += 1
                        continue
                    shard_accepted += 1
                    _scan_one_document(original, document.document_id, normalized, original_acc)
                    _scan_one_document(corrected, document.document_id, normalized, corrected_acc)
            except Exception:
                index.rollback_shard()
                raise
            stats = ShardScanStats(
                path=identity.path,
                sha256=identity.sha256,
                size_bytes=identity.size_bytes,
                physical_rows_visited=shard_physical_rows,
                accepted_normalized_documents=shard_accepted,
                rejected_documents=0,
                duplicates_suppressed=shard_duplicates,
                unique_documents_scanned=shard_accepted,
                normalized_utf8_bytes_scanned=0,
                normalized_codepoints_scanned=0,
                complete=True,
            )
            index.commit_shard(stats)

    # Population cross-check: fail closed if the dedup population disagrees with
    # the accepted source evidence.
    retained = original_acc.documents_tokenized
    if physical_rows_visited != EXPECTED_PHYSICAL_ROWS:
        raise CorpusScanError(
            f"physical rows {physical_rows_visited} != expected {EXPECTED_PHYSICAL_ROWS}"
        )
    if duplicates_suppressed != EXPECTED_DUPLICATES_SUPPRESSED:
        raise CorpusScanError(
            f"duplicates suppressed {duplicates_suppressed} != expected "
            f"{EXPECTED_DUPLICATES_SUPPRESSED}"
        )
    if retained != EXPECTED_RETAINED_DOCUMENTS:
        raise CorpusScanError(
            f"retained documents {retained} != expected {EXPECTED_RETAINED_DOCUMENTS}"
        )

    # Corrected tokenizer must never emit an id above 50256.
    if corrected_acc.out_of_range_occurrences != 0:
        raise CorpusScanError(
            f"corrected tokenizer emitted {corrected_acc.out_of_range_occurrences} "
            f"out-of-range occurrences (must be zero)"
        )

    try:
        import tokenizers

        tokenizers_version = getattr(tokenizers, "__version__", "unknown")
    except ModuleNotFoundError:
        tokenizers_version = "unknown"

    report: dict[str, object] = {
        "schema_version": "expertforge-d0-tokenizer-corpus-scan-report/1",
        "decision": "doctrine/decisions/0012-d0-tokenizer-vocabulary-divergence.md",
        "resolution": 1,
        "status": "complete",
        "scan_complete": True,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_manifest_sha256": dataset_manifest.manifest_sha256,
        "accepted_dedup_population": {
            "physical_rows_visited": physical_rows_visited,
            "duplicates_suppressed": duplicates_suppressed,
            "retained_documents": retained,
        },
        "split_algorithm": {
            "formula": "first_u64_be(sha256('expertforge-d0-split-v1\\0' || utf8(document_id))) mod 1000",
            "train_threshold": "bucket < 995",
            "validation_threshold": "bucket >= 995",
        },
        "training_order_algorithm": {
            "formula": (
                "sha256('expertforge-d0-order-v1\\0' || uint64_be(epoch) || "
                "uint64_be(data_seed) || utf8(document_id))"
            ),
            "data_seed": _D0_DATA_SEED,
        },
        "validation_order_algorithm": {
            "formula": "sha256('expertforge-d0-validation-v1\\0' || utf8(document_id))",
        },
        "scanner_source_commit": scanner_source_commit,
        "tokenizers_version": tokenizers_version,
        "original_tokenizer": {
            "manifest_id": original.manifest_id,
            "manifest_sha256": original.manifest_sha256,
            "tokenizer_json_sha256": original.tokenizer_json_sha256,
            "documents_tokenized": original_acc.documents_tokenized,
            "documents_emitting_id_above_50256": original_acc.documents_with_out_of_range,
            "out_of_range_occurrences": original_acc.out_of_range_occurrences,
            "per_id_count_50257_through_50276": {
                str(i): original_acc.per_oor_id_count.get(i, 0) for i in range(50257, 50277)
            },
            "maximum_emitted_id": original_acc.max_emitted_id,
            "total_token_count": original_acc.token_count,
            "document_terminators": original_acc.document_terminators,
            "train_documents": original_acc.train_documents,
            "validation_documents": original_acc.validation_documents,
            "train_tokens": original_acc.train_tokens,
            "validation_tokens": original_acc.validation_tokens,
        },
        "corrected_tokenizer": {
            "manifest_id": corrected.manifest_id,
            "manifest_sha256": corrected.manifest_sha256,
            "tokenizer_json_sha256": corrected.tokenizer_json_sha256,
            "documents_tokenized": corrected_acc.documents_tokenized,
            "documents_emitting_id_above_50256": corrected_acc.documents_with_out_of_range,
            "out_of_range_occurrences": corrected_acc.out_of_range_occurrences,
            "maximum_emitted_id": corrected_acc.max_emitted_id,
            "total_token_count": corrected_acc.token_count,
            "document_terminators": corrected_acc.document_terminators,
            "train_documents": corrected_acc.train_documents,
            "validation_documents": corrected_acc.validation_documents,
            "train_tokens": corrected_acc.train_tokens,
            "validation_tokens": corrected_acc.validation_tokens,
        },
        "original_vs_corrected_token_delta": (corrected_acc.token_count - original_acc.token_count),
        "corrected_out_of_range_count_must_be_zero": corrected_acc.out_of_range_occurrences,
    }
    report["report_sha256"] = hashlib.sha256(
        _canonical_json({k: v for k, v in report.items() if k != "report_sha256"})
    ).hexdigest()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(
        (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original-tokenizer-root", type=Path, required=True)
    parser.add_argument("--corrected-tokenizer-root", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Durable directory for the restart database and report.",
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional report path; defaults to WORK_DIR/tokenizer-corpus-scan-report.json.",
    )
    parser.add_argument("--parquet-batch-size", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    state_db = work_dir / "dedup-state.sqlite"
    report_path = args.report or (work_dir / "tokenizer-corpus-scan-report.json")
    try:
        execute_dual_tokenizer_scan(
            dataset_root=args.dataset_root,
            original_tokenizer_root=args.original_tokenizer_root,
            corrected_tokenizer_root=args.corrected_tokenizer_root,
            state_database_path=state_db,
            report_path=report_path,
            scanner_source_commit=args.source_commit,
            parquet_batch_size=args.parquet_batch_size,
        )
    except CorpusScanError as exc:
        sys.stderr.write(f"D0 TOKENIZER CORPUS SCAN FAILED: {exc}\n")
        return 1
    print(f"corpus scan report -> {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
