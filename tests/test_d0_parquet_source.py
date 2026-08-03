from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from expertforge.d0.errors import MissingOptionalDependencyError, SourceVerificationError
from expertforge.d0.parquet_source import iter_parquet_documents

pa: Any = pytest.importorskip("pyarrow")
pq: Any = pytest.importorskip("pyarrow.parquet")


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=2)


def _row(document_id: str, text: object) -> dict[str, object]:
    return {
        "id": document_id,
        "text": text,
        "url": f"https://example.invalid/{document_id}",
        "dump": "CC-MAIN-2026-30",
        "file_path": f"crawl/{document_id}.warc.gz",
        "unused": 7,
    }


def test_parquet_traversal_preserves_physical_row_order(tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    _write(path, [_row("doc-2", "second"), _row("doc-1", "first"), _row("doc-3", "third")])
    documents = tuple(
        iter_parquet_documents(
            path,
            source_file_path="sample/000.parquet",
            batch_size=1,
        )
    )
    assert tuple(document.document_id for document in documents) == (
        "doc-2",
        "doc-1",
        "doc-3",
    )
    assert tuple(document.physical_row_index for document in documents) == (0, 1, 2)
    assert all(document.source_file_path == "sample/000.parquet" for document in documents)
    assert documents[0].upstream_file_path == "crawl/doc-2.warc.gz"


def test_parquet_traversal_rejects_missing_consumed_field(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    row = _row("doc-1", "text")
    row.pop("dump")
    _write(path, [row])
    with pytest.raises(SourceVerificationError, match="missing consumed fields"):
        tuple(iter_parquet_documents(path, source_file_path="sample/000.parquet"))


def test_parquet_traversal_rejects_null_text(tmp_path: Path) -> None:
    path = tmp_path / "null.parquet"
    _write(path, [_row("doc-1", None)])
    with pytest.raises(SourceVerificationError, match="text.*non-null string"):
        tuple(iter_parquet_documents(path, source_file_path="sample/000.parquet"))


def test_parquet_traversal_requires_explicit_optional_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def missing_import(name: str) -> Any:
        if name == "pyarrow.parquet":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", missing_import)
    with pytest.raises(MissingOptionalDependencyError, match="d0-data"):
        tuple(
            iter_parquet_documents(
                tmp_path / "unused.parquet",
                source_file_path="sample/000.parquet",
            )
        )
