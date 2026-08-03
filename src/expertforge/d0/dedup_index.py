"""Restartable, scan-local exact-text duplicate suppression."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from expertforge.d0.errors import SourceOrderError
from expertforge.d0.normalization import normalized_text_sha256


@dataclass(frozen=True, slots=True)
class DedupDecision:
    """Result of registering one normalized document in frozen source order."""

    digest: str
    accepted: bool
    first_source_file_path: str
    first_physical_row_index: int


class DedupIndex:
    """SQLite-backed duplicate index committed only at shard boundaries.

    A transaction spans one source shard. A crash or explicit rollback removes
    every digest first observed in the incomplete shard, allowing that shard to
    be replayed without weakening first-in-source-order semantics.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_documents (
                digest TEXT PRIMARY KEY NOT NULL,
                first_source_file_path TEXT NOT NULL,
                first_physical_row_index INTEGER NOT NULL CHECK(first_physical_row_index >= 0)
            ) WITHOUT ROWID
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_state (
                key TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        self._active_source_file_path: str | None = None
        self._last_row_index = -1
        self._closed = False

    def __enter__(self) -> DedupIndex:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._active_source_file_path is not None:
            self.rollback_shard()
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("dedup index is closed")

    @property
    def last_completed_shard(self) -> str | None:
        """Return the latest durably committed shard path, if any."""

        self._require_open()
        row = self._connection.execute(
            "SELECT value FROM scan_state WHERE key = 'last_completed_shard'"
        ).fetchone()
        if row is None:
            return None
        value = row[0]
        if not isinstance(value, str):
            raise SourceOrderError("dedup index contains malformed shard state")
        return value

    @property
    def unique_document_count(self) -> int:
        """Return the number of unique normalized-text digests durably visible."""

        self._require_open()
        row = self._connection.execute("SELECT COUNT(*) FROM seen_documents").fetchone()
        if row is None or type(row[0]) is not int:
            raise SourceOrderError("dedup index count query returned malformed state")
        return row[0]

    def begin_shard(self, source_file_path: str) -> None:
        """Begin one lexicographically ordered shard transaction."""

        self._require_open()
        if self._active_source_file_path is not None:
            raise RuntimeError("a shard transaction is already active")
        if not source_file_path:
            raise SourceOrderError("source_file_path must not be blank")
        previous = self.last_completed_shard
        if previous is not None and source_file_path <= previous:
            raise SourceOrderError(
                f"shard order is not strictly increasing: previous {previous}, "
                f"next {source_file_path}"
            )
        self._connection.execute("BEGIN IMMEDIATE")
        self._active_source_file_path = source_file_path
        self._last_row_index = -1

    def register(self, normalized_text: str, physical_row_index: int) -> DedupDecision:
        """Register one normalized document and select first-in-source-order."""

        self._require_open()
        source_file_path = self._active_source_file_path
        if source_file_path is None:
            raise RuntimeError("begin_shard must be called before register")
        if type(physical_row_index) is not int or physical_row_index < 0:
            raise SourceOrderError("physical_row_index must be a non-negative exact integer")
        if physical_row_index <= self._last_row_index:
            raise SourceOrderError(
                f"row order is not strictly increasing in {source_file_path}: "
                f"previous {self._last_row_index}, next {physical_row_index}"
            )

        digest = normalized_text_sha256(normalized_text)
        cursor = self._connection.execute(
            """
            INSERT OR IGNORE INTO seen_documents (
                digest,
                first_source_file_path,
                first_physical_row_index
            ) VALUES (?, ?, ?)
            """,
            (digest, source_file_path, physical_row_index),
        )
        accepted = cursor.rowcount == 1
        if accepted:
            first_source_file_path = source_file_path
            first_physical_row_index = physical_row_index
        else:
            first = self._connection.execute(
                """
                SELECT first_source_file_path, first_physical_row_index
                FROM seen_documents
                WHERE digest = ?
                """,
                (digest,),
            ).fetchone()
            if first is None or not isinstance(first[0], str) or type(first[1]) is not int:
                raise SourceOrderError("dedup index lost an existing digest record")
            first_source_file_path = first[0]
            first_physical_row_index = first[1]

        self._last_row_index = physical_row_index
        return DedupDecision(
            digest=digest,
            accepted=accepted,
            first_source_file_path=first_source_file_path,
            first_physical_row_index=first_physical_row_index,
        )

    def commit_shard(self) -> None:
        """Durably commit the active shard and its restart boundary."""

        self._require_open()
        source_file_path = self._active_source_file_path
        if source_file_path is None:
            raise RuntimeError("no shard transaction is active")
        self._connection.execute(
            """
            INSERT INTO scan_state (key, value)
            VALUES ('last_completed_shard', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (source_file_path,),
        )
        self._connection.execute("COMMIT")
        self._active_source_file_path = None
        self._last_row_index = -1

    def rollback_shard(self) -> None:
        """Discard every mutation from the active incomplete shard."""

        self._require_open()
        if self._active_source_file_path is None:
            raise RuntimeError("no shard transaction is active")
        self._connection.execute("ROLLBACK")
        self._active_source_file_path = None
        self._last_row_index = -1

    def close(self) -> None:
        """Close the index after rolling back no implicit work."""

        if self._closed:
            return
        if self._active_source_file_path is not None:
            raise RuntimeError("cannot close dedup index with an active shard transaction")
        self._connection.close()
        self._closed = True
