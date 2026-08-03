from __future__ import annotations

from pathlib import Path

import pytest

from expertforge.d0.contamination import HitRecord
from expertforge.d0.dedup_index import DedupIndex
from expertforge.d0.errors import SourceOrderError
from expertforge.d0.scan_report import ShardScanStats


def _stats(
    path: str,
    *,
    rows: int,
    accepted: int,
    duplicates: int,
    unique: int,
) -> ShardScanStats:
    return ShardScanStats(
        path=path,
        sha256="1" * 64,
        size_bytes=100,
        physical_rows_visited=rows,
        accepted_normalized_documents=accepted,
        rejected_documents=rows - accepted,
        duplicates_suppressed=duplicates,
        unique_documents_scanned=unique,
        normalized_utf8_bytes_scanned=10,
        normalized_codepoints_scanned=10,
        complete=True,
    )


def test_first_in_source_order_is_retained_across_shards(tmp_path: Path) -> None:
    index_path = tmp_path / "dedup.sqlite3"
    with DedupIndex(index_path) as index:
        index.begin_shard("sample/000.parquet")
        first = index.register("alpha", 0)
        duplicate = index.register("alpha", 1)
        index.commit_shard(_stats("sample/000.parquet", rows=2, accepted=2, duplicates=1, unique=1))
        assert first.accepted is True
        assert duplicate.accepted is False
        assert duplicate.first_source_file_path == "sample/000.parquet"
        assert duplicate.first_physical_row_index == 0

    with DedupIndex(index_path) as index:
        assert index.last_completed_shard == "sample/000.parquet"
        index.begin_shard("sample/001.parquet")
        cross_shard_duplicate = index.register("alpha", 0)
        new_document = index.register("beta", 1)
        index.commit_shard(_stats("sample/001.parquet", rows=2, accepted=2, duplicates=1, unique=1))
        assert cross_shard_duplicate.accepted is False
        assert new_document.accepted is True
        assert index.unique_document_count == 2
        assert tuple(item.path for item in index.completed_shard_stats()) == (
            "sample/000.parquet",
            "sample/001.parquet",
        )


def test_incomplete_shard_rolls_back_dedup_and_evidence_for_replay(tmp_path: Path) -> None:
    index_path = tmp_path / "dedup.sqlite3"
    with DedupIndex(index_path) as index:
        index.begin_shard("sample/000.parquet")
        index.register("alpha", 0)
        index.commit_shard(_stats("sample/000.parquet", rows=1, accepted=1, duplicates=0, unique=1))
        index.begin_shard("sample/001.parquet")
        index.register("beta", 0)
        index.rollback_shard()
        assert index.last_completed_shard == "sample/000.parquet"
        assert index.unique_document_count == 1
        assert len(index.completed_shard_stats()) == 1
        index.begin_shard("sample/001.parquet")
        replayed = index.register("beta", 0)
        index.commit_shard(_stats("sample/001.parquet", rows=1, accepted=1, duplicates=0, unique=1))
        assert replayed.accepted is True


def test_hit_records_commit_with_their_shard(tmp_path: Path) -> None:
    hit = HitRecord(
        prompt_id="d0-gen-01",
        check="exact_normalized_probe_substring",
        document_id="doc-1",
        source_file_path="sample/000.parquet",
        physical_row_index=0,
        match_start_codepoint=2,
        match_end_codepoint=66,
    )
    with DedupIndex(tmp_path / "dedup.sqlite3") as index:
        index.begin_shard("sample/000.parquet")
        index.register("contaminated text", 0)
        index.commit_shard(
            _stats("sample/000.parquet", rows=1, accepted=1, duplicates=0, unique=1),
            (hit,),
        )
        assert index.completed_hits() == (hit,)


def test_shard_and_row_order_are_fail_closed(tmp_path: Path) -> None:
    with DedupIndex(tmp_path / "dedup.sqlite3") as index:
        index.begin_shard("sample/001.parquet")
        index.register("alpha", 1)
        with pytest.raises(SourceOrderError, match="row order"):
            index.register("beta", 1)
        index.commit_shard(_stats("sample/001.parquet", rows=2, accepted=1, duplicates=0, unique=1))
        with pytest.raises(SourceOrderError, match="shard order"):
            index.begin_shard("sample/000.parquet")


def test_incomplete_or_mismatched_evidence_is_rejected(tmp_path: Path) -> None:
    with DedupIndex(tmp_path / "dedup.sqlite3") as index:
        index.begin_shard("sample/000.parquet")
        index.register("alpha", 0)
        incomplete = _stats("sample/000.parquet", rows=1, accepted=1, duplicates=0, unique=1)
        incomplete = ShardScanStats(
            path=incomplete.path,
            sha256=incomplete.sha256,
            size_bytes=incomplete.size_bytes,
            physical_rows_visited=incomplete.physical_rows_visited,
            accepted_normalized_documents=incomplete.accepted_normalized_documents,
            rejected_documents=incomplete.rejected_documents,
            duplicates_suppressed=incomplete.duplicates_suppressed,
            unique_documents_scanned=incomplete.unique_documents_scanned,
            normalized_utf8_bytes_scanned=incomplete.normalized_utf8_bytes_scanned,
            normalized_codepoints_scanned=incomplete.normalized_codepoints_scanned,
            complete=False,
        )
        with pytest.raises(SourceOrderError, match="complete shard evidence"):
            index.commit_shard(incomplete)
        index.rollback_shard()
