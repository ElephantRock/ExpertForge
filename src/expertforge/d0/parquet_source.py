"""Lazy PyArrow boundary for frozen D0 Parquet source traversal."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from expertforge.d0.errors import MissingOptionalDependencyError, SourceVerificationError

_CONSUMED_FIELDS: tuple[str, ...] = ("id", "text", "url", "dump", "file_path")
_DEFAULT_BATCH_SIZE = 4096


class _SchemaLike(Protocol):
    @property
    def names(self) -> list[str]: ...


class _RecordBatchLike(Protocol):
    @property
    def num_rows(self) -> int: ...

    def to_pylist(self) -> list[dict[str, object]]: ...


class _ParquetFileLike(Protocol):
    @property
    def schema_arrow(self) -> _SchemaLike: ...

    def iter_batches(
        self,
        *,
        batch_size: int,
        columns: Sequence[str],
        use_threads: bool,
    ) -> Iterator[_RecordBatchLike]: ...


class _ParquetModuleLike(Protocol):
    def ParquetFile(self, source: str) -> _ParquetFileLike: ...


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One consumed FineWeb-Edu row with explicit physical source position."""

    document_id: str
    text: str
    url: str
    dump: str
    upstream_file_path: str
    source_file_path: str
    physical_row_index: int


def _load_parquet_module() -> _ParquetModuleLike:
    try:
        module = importlib.import_module("pyarrow.parquet")
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependencyError(
            "D0 Parquet traversal requires the optional 'd0-data' extra; "
            "install ExpertForge with expertforge[d0-data]"
        ) from exc
    return cast(_ParquetModuleLike, module)


def _required_string(row: Mapping[str, object], field: str, row_index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise SourceVerificationError(
            f"Parquet row {row_index} field {field!r} must be a non-null string"
        )
    if field == "id" and not value:
        raise SourceVerificationError(f"Parquet row {row_index} document id must not be blank")
    return value


def iter_parquet_documents(
    path: Path,
    *,
    source_file_path: str,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Iterator[SourceDocument]:
    """Yield consumed fields in row-group and physical-row order.

    Column decoding is deliberately single-threaded. The iterator never applies
    filtering, projection reordering, dataset partition discovery, or parallel
    fragment scheduling that could obscure the frozen physical row sequence.
    """

    if not source_file_path:
        raise SourceVerificationError("source_file_path must not be blank")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive exact integer")
    parquet = _load_parquet_module()
    try:
        parquet_file = parquet.ParquetFile(str(path))
        names = tuple(parquet_file.schema_arrow.names)
    except Exception as exc:
        raise SourceVerificationError(f"cannot open Parquet source {path}: {exc}") from exc

    missing = [field for field in _CONSUMED_FIELDS if field not in names]
    if missing:
        raise SourceVerificationError(
            f"Parquet source {source_file_path} is missing consumed fields: {missing}"
        )

    physical_row_index = 0
    try:
        batches = parquet_file.iter_batches(
            batch_size=batch_size,
            columns=_CONSUMED_FIELDS,
            use_threads=False,
        )
        for batch in batches:
            rows = batch.to_pylist()
            if len(rows) != batch.num_rows:
                raise SourceVerificationError(
                    f"Parquet batch row-count mismatch in {source_file_path}"
                )
            for row in rows:
                if not isinstance(row, Mapping):
                    raise SourceVerificationError(
                        f"Parquet row {physical_row_index} is not a field mapping"
                    )
                yield SourceDocument(
                    document_id=_required_string(row, "id", physical_row_index),
                    text=_required_string(row, "text", physical_row_index),
                    url=_required_string(row, "url", physical_row_index),
                    dump=_required_string(row, "dump", physical_row_index),
                    upstream_file_path=_required_string(row, "file_path", physical_row_index),
                    source_file_path=source_file_path,
                    physical_row_index=physical_row_index,
                )
                physical_row_index += 1
    except SourceVerificationError:
        raise
    except Exception as exc:
        raise SourceVerificationError(
            f"cannot stream Parquet source {source_file_path}: {exc}"
        ) from exc
