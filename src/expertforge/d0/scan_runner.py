"""Restartable orchestration for the source-bound D0 contamination scan."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from expertforge.d0.contamination import ContaminationMatcher, HitRecord
from expertforge.d0.dedup_index import DedupIndex
from expertforge.d0.errors import ContaminationError, SourceOrderError
from expertforge.d0.normalization import normalize_text
from expertforge.d0.parquet_source import iter_parquet_documents
from expertforge.d0.scan_identity import ScanIdentity, bind_scan_identity
from expertforge.d0.scan_report import (
    ShardScanStats,
    build_scan_report,
    canonical_report_bytes,
)
from expertforge.d0.source_manifest import SourceManifest
from expertforge.d0.source_verification import (
    resolve_inventory_path,
    verify_source_inventory,
)

SCANNER_ALGORITHM_VERSION = "d0-contamination-aho-corasick-v1"


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync; a no-op on platforms without directory fsync."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def runtime_versions() -> dict[str, str]:
    """Return the minimum runtime identity required by the scan report."""

    return {
        "expertforge": _distribution_version("expertforge"),
        "pyarrow": _distribution_version("pyarrow"),
        "python": platform.python_version(),
    }


def load_prompt_manifest(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    """Load the frozen prompt manifest after strict raw-byte identity verification."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContaminationError(f"cannot read prompt manifest {path}: {exc}") from exc

    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ContaminationError(
            f"prompt manifest SHA-256 mismatch: expected {expected_sha256}, actual {actual_sha256}"
        )
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContaminationError(f"invalid prompt manifest {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ContaminationError("prompt manifest must be a JSON object")
    payload_sha256 = value.get("prompt_payload_sha256")
    if not isinstance(payload_sha256, str):
        raise ContaminationError("prompt payload SHA-256 is missing")
    ContaminationMatcher.from_manifest(value)
    return value, payload_sha256


def _validate_resume_prefix(
    completed: tuple[ShardScanStats, ...],
    dataset_manifest: SourceManifest,
) -> None:
    if len(completed) > len(dataset_manifest.files):
        raise SourceOrderError("restart state contains more shards than the frozen inventory")
    for index, stats in enumerate(completed):
        identity = dataset_manifest.files[index]
        if stats.path != identity.path:
            raise SourceOrderError("restart shard sequence is not a frozen-inventory prefix")
        if stats.sha256 != identity.sha256 or stats.size_bytes != identity.size_bytes:
            raise SourceOrderError("restart shard evidence does not match the frozen inventory")
        if not stats.complete:
            raise SourceOrderError("restart state contains incomplete committed shard evidence")


def _atomic_write_report(path: Path, report: Mapping[str, object]) -> None:
    payload = canonical_report_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".part",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        published = True
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            temporary_path.unlink(missing_ok=True)


def _report(
    *,
    dataset_manifest: SourceManifest,
    tokenizer_manifest: SourceManifest,
    prompt_manifest_sha256: str,
    prompt_payload_sha256: str,
    scanner_source_commit: str,
    shards: tuple[ShardScanStats, ...],
    hits: tuple[HitRecord, ...],
    scan_complete: bool,
    terminal_diagnostic: str,
) -> dict[str, object]:
    return build_scan_report(
        dataset_manifest_sha256=dataset_manifest.manifest_sha256,
        tokenizer_manifest_sha256=tokenizer_manifest.manifest_sha256,
        prompt_manifest_sha256=prompt_manifest_sha256,
        prompt_payload_sha256=prompt_payload_sha256,
        scanner_source_commit=scanner_source_commit,
        scanner_algorithm_version=SCANNER_ALGORITHM_VERSION,
        runtime_versions=runtime_versions(),
        expected_shard_count=len(dataset_manifest.files),
        expected_total_size_bytes=dataset_manifest.total_size_bytes,
        shards=shards,
        hits=hits,
        scan_complete=scan_complete,
        terminal_diagnostic=terminal_diagnostic,
    )


def execute_contamination_scan(
    *,
    dataset_root: Path,
    tokenizer_root: Path,
    dataset_manifest: SourceManifest,
    tokenizer_manifest: SourceManifest,
    prompt_manifest: Mapping[str, Any],
    prompt_manifest_sha256: str,
    prompt_payload_sha256: str,
    state_database_path: Path,
    scan_identity_path: Path,
    report_path: Path,
    scanner_source_commit: str,
    parquet_batch_size: int = 4096,
) -> dict[str, object]:
    """Verify all frozen sources, resume at a shard boundary, and scan to completion."""

    if dataset_manifest.kind != "dataset":
        raise ContaminationError("dataset_manifest is not a dataset source manifest")
    if tokenizer_manifest.kind != "tokenizer":
        raise ContaminationError("tokenizer_manifest is not a tokenizer source manifest")
    if prompt_manifest.get("prompt_payload_sha256") != prompt_payload_sha256:
        raise ContaminationError("prompt payload identity disagrees with the frozen manifest")
    matcher = ContaminationMatcher.from_manifest(prompt_manifest)
    bind_scan_identity(
        scan_identity_path,
        ScanIdentity(
            dataset_manifest_sha256=dataset_manifest.manifest_sha256,
            tokenizer_manifest_sha256=tokenizer_manifest.manifest_sha256,
            prompt_manifest_sha256=prompt_manifest_sha256,
            prompt_payload_sha256=prompt_payload_sha256,
            scanner_algorithm_version=SCANNER_ALGORITHM_VERSION,
            scanner_source_commit=scanner_source_commit,
        ),
    )

    with DedupIndex(state_database_path) as index:
        try:
            verify_source_inventory(dataset_root, dataset_manifest)
            verify_source_inventory(tokenizer_root, tokenizer_manifest)
            completed = index.completed_shard_stats()
            _validate_resume_prefix(completed, dataset_manifest)

            for identity in dataset_manifest.files[len(completed) :]:
                index.begin_shard(identity.path)
                shard_hits: list[HitRecord] = []
                physical_rows_visited = 0
                accepted_documents = 0
                duplicates_suppressed = 0
                unique_documents_scanned = 0
                normalized_utf8_bytes_scanned = 0
                normalized_codepoints_scanned = 0
                try:
                    local_path = resolve_inventory_path(dataset_root, identity.path)
                    for document in iter_parquet_documents(
                        local_path,
                        source_file_path=identity.path,
                        batch_size=parquet_batch_size,
                    ):
                        physical_rows_visited += 1
                        normalized = normalize_text(document.text)
                        accepted_documents += 1
                        decision = index.register(normalized, document.physical_row_index)
                        if not decision.accepted:
                            duplicates_suppressed += 1
                            continue
                        unique_documents_scanned += 1
                        normalized_utf8_bytes_scanned += len(normalized.encode("utf-8"))
                        normalized_codepoints_scanned += len(normalized)
                        shard_hits.extend(
                            matcher.scan_document(
                                document_id=document.document_id,
                                source_file_path=identity.path,
                                physical_row_index=document.physical_row_index,
                                text=normalized,
                            )
                        )
                    stats = ShardScanStats(
                        path=identity.path,
                        sha256=identity.sha256,
                        size_bytes=identity.size_bytes,
                        physical_rows_visited=physical_rows_visited,
                        accepted_normalized_documents=accepted_documents,
                        rejected_documents=0,
                        duplicates_suppressed=duplicates_suppressed,
                        unique_documents_scanned=unique_documents_scanned,
                        normalized_utf8_bytes_scanned=normalized_utf8_bytes_scanned,
                        normalized_codepoints_scanned=normalized_codepoints_scanned,
                        complete=True,
                    )
                    index.commit_shard(stats, tuple(shard_hits))
                except Exception:
                    index.rollback_shard()
                    raise

            final_shards = index.completed_shard_stats()
            final_hits = index.completed_hits()
            report = _report(
                dataset_manifest=dataset_manifest,
                tokenizer_manifest=tokenizer_manifest,
                prompt_manifest_sha256=prompt_manifest_sha256,
                prompt_payload_sha256=prompt_payload_sha256,
                scanner_source_commit=scanner_source_commit,
                shards=final_shards,
                hits=final_hits,
                scan_complete=True,
                terminal_diagnostic="complete_frozen_inventory_scan",
            )
            _atomic_write_report(report_path, report)
            return report
        except Exception as exc:
            partial_report = _report(
                dataset_manifest=dataset_manifest,
                tokenizer_manifest=tokenizer_manifest,
                prompt_manifest_sha256=prompt_manifest_sha256,
                prompt_payload_sha256=prompt_payload_sha256,
                scanner_source_commit=scanner_source_commit,
                shards=index.completed_shard_stats(),
                hits=index.completed_hits(),
                scan_complete=False,
                terminal_diagnostic=f"scan_failed:{type(exc).__name__}:{exc}",
            )
            _atomic_write_report(report_path, partial_report)
            raise
