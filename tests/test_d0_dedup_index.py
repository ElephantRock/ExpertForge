from __future__ import annotations

from pathlib import Path

import pytest

from expertforge.d0.dedup_index import DedupIndex
from expertforge.d0.errors import SourceOrderError


def test_first_in_source_order_is_retained_across_shards(tmp_path: Path) -> None:
    index_path = tmp_path / "dedup.sqlite3"
    with DedupIndex(index_path) as index:
        index.begin_shard("sample/000.parquet")
        first = index.register("alpha", 0)
        duplicate = index.register("alpha", 1)
        index.commit_shard()
        assert first.accepted is True
        assert duplicate.accepted is False
        assert duplicate.first_source_file_path == "sample/000.parquet"
        assert duplicate.first_physical_row_index == 0

    with DedupIndex(index_path) as index:
        assert index.last_completed_shard == "sample/000.parquet"
        index.begin_shard("sample/001.parquet")
        cross_shard_duplicate = index.register("alpha", 0)
        new_document = index.register("beta", 1)
        index.commit_shard()
        assert cross_shard_duplicate.accepted is False
        assert new_document.accepted is True
        assert index.unique_document_count == 2


def test_incomplete_shard_rolls_back_for_exact_replay(tmp_path: Path) -> None:
    index_path = tmp_path / "dedup.sqlite3"
    with DedupIndex(index_path) as index:
        index.begin_shard("sample/000.parquet")
        index.register("alpha", 0)
        index.commit_shard()
        index.begin_shard("sample/001.parquet")
        index.register("beta", 0)
        index.rollback_shard()
        assert index.last_completed_shard == "sample/000.parquet"
        assert index.unique_document_count == 1
        index.begin_shard("sample/001.parquet")
        replayed = index.register("beta", 0)
        index.commit_shard()
        assert replayed.accepted is True


def test_shard_and_row_order_are_fail_closed(tmp_path: Path) -> None:
    with DedupIndex(tmp_path / "dedup.sqlite3") as index:
        index.begin_shard("sample/001.parquet")
        index.register("alpha", 1)
        with pytest.raises(SourceOrderError, match="row order"):
            index.register("beta", 1)
        index.commit_shard()
        with pytest.raises(SourceOrderError, match="shard order"):
            index.begin_shard("sample/000.parquet")
