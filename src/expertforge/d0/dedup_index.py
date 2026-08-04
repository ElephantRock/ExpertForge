"""Restartable, scan-local duplicate suppression and shard evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from expertforge.d0.contamination import CheckName, HitRecord
from expertforge.d0.errors import SourceOrderError
from expertforge.d0.normalization import normalized_text_sha256
from expertforge.d0.scan_report import ShardScanStats
from expertforge.d0.source_manifest import canonical_json_bytes

_CHECK_NAMES: tuple[CheckName, ...] = (
    "exact_normalized_prompt_substring",
    "exact_normalized_probe_substring",
    "any_exact_contiguous_64_codepoint_prompt_window",
)


@dataclass(frozen=True, slots=True)
class DedupDecision:
    """Result of registering one normalized document in frozen source order."""

    digest: str
    accepted: bool
    first_source_file_path: str
    first_physical_row_index: int


def _record_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise SourceOrderError(f"stored shard evidence field {field!r} is invalid")
    return value


def _record_int(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise SourceOrderError(f"stored shard evidence field {field!r} is invalid")
    return value


def _record_bool(record: Mapping[str, Any], field: str) -> bool:
    value = record.get(field)
    if type(value) is not bool:
        raise SourceOrderError(f"stored shard evidence field {field!r} is invalid")
    return value


def _deserialize_shard_stats(payload: str) -> ShardScanStats:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SourceOrderError("stored shard evidence is malformed JSON") from exc
    if not isinstance(value, Mapping):
        raise SourceOrderError("stored shard evidence must be a JSON object")
    return ShardScanStats(
        path=_record_string(value, "path"),
        sha256=_record_string(value, "sha256"),
        size_bytes=_record_int(value, "size_bytes"),
        physical_rows_visited=_record_int(value, "physical_rows_visited"),
        accepted_normalized_documents=_record_int(value, "accepted_normalized_documents"),
        rejected_documents=_record_int(value, "rejected_documents"),
        duplicates_suppressed=_record_int(value, "duplicates_suppressed"),
        unique_documents_scanned=_record_int(value, "unique_documents_scanned"),
        normalized_utf8_bytes_scanned=_record_int(value, "normalized_utf8_bytes_scanned"),
        normalized_codepoints_scanned=_record_int(value, "normalized_codepoints_scanned"),
        complete=_record_bool(value, "complete"),
    )


class DedupIndex:
    """SQLite-backed duplicate and report-evidence journal.

    A transaction spans one source shard. Duplicate keys, shard statistics, hit
    records, and the restart marker commit together. A crash or rollback removes
    every mutation from the incomplete shard, so replay cannot retain dedup state
    while losing the report evidence needed to prove what was scanned.
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
            CREATE TABLE IF NOT EXISTS completed_shards (
                source_file_path TEXT PRIMARY KEY NOT NULL,
                evidence_json TEXT NOT NULL,
                evidence_sha256 TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS contamination_hits (
                source_file_path TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                check_name TEXT NOT NULL,
                document_id TEXT NOT NULL,
                physical_row_index INTEGER NOT NULL CHECK(physical_row_index >= 0),
                match_start_codepoint INTEGER NOT NULL CHECK(match_start_codepoint >= 0),
                match_end_codepoint INTEGER NOT NULL CHECK(match_end_codepoint >= 0),
                PRIMARY KEY (
                    source_file_path,
                    prompt_id,
                    check_name,
                    document_id,
                    physical_row_index,
                    match_start_codepoint,
                    match_end_codepoint
                ),
                FOREIGN KEY(source_file_path)
                    REFERENCES completed_shards(source_file_path)
                    ON DELETE CASCADE
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
        """Return the number of normalized-text digests durably visible."""

        self._require_open()
        row = self._connection.execute("SELECT COUNT(*) FROM seen_documents").fetchone()
        if row is None or type(row[0]) is not int:
            raise SourceOrderError("dedup index count query returned malformed state")
        return row[0]

    def completed_shard_stats(self) -> tuple[ShardScanStats, ...]:
        """Load completed shard evidence in frozen source order."""

        self._require_open()
        rows = self._connection.execute(
            "SELECT evidence_json, evidence_sha256 FROM completed_shards "
            "ORDER BY source_file_path ASC"
        ).fetchall()
        stats: list[ShardScanStats] = []
        for payload, declared_digest in rows:
            if not isinstance(payload, str) or not isinstance(declared_digest, str):
                raise SourceOrderError("stored shard evidence row has invalid types")
            actual_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if actual_digest != declared_digest:
                raise SourceOrderError("stored shard evidence digest mismatch")
            item = _deserialize_shard_stats(payload)
            item.as_dict()
            stats.append(item)
        return tuple(stats)

    def completed_hits(self) -> tuple[HitRecord, ...]:
        """Load all durably committed hit records in deterministic report order."""

        self._require_open()
        rows = self._connection.execute(
            """
            SELECT
                prompt_id,
                check_name,
                document_id,
                source_file_path,
                physical_row_index,
                match_start_codepoint,
                match_end_codepoint
            FROM contamination_hits
            ORDER BY
                prompt_id ASC,
                CASE check_name
                    WHEN 'exact_normalized_prompt_substring' THEN 0
                    WHEN 'exact_normalized_probe_substring' THEN 1
                    WHEN 'any_exact_contiguous_64_codepoint_prompt_window' THEN 2
                    ELSE 3
                END ASC,
                document_id ASC,
                match_start_codepoint ASC,
                source_file_path ASC,
                physical_row_index ASC,
                match_end_codepoint ASC
            """
        ).fetchall()
        hits: list[HitRecord] = []
        for row in rows:
            prompt_id, check_name, document_id, source_path, row_index, start, end = row
            if (
                not isinstance(prompt_id, str)
                or check_name not in _CHECK_NAMES
                or not isinstance(document_id, str)
                or not isinstance(source_path, str)
                or type(row_index) is not int
                or type(start) is not int
                or type(end) is not int
            ):
                raise SourceOrderError("stored contamination hit has invalid fields")
            hits.append(
                HitRecord(
                    prompt_id=prompt_id,
                    check=cast(CheckName, check_name),
                    document_id=document_id,
                    source_file_path=source_path,
                    physical_row_index=row_index,
                    match_start_codepoint=start,
                    match_end_codepoint=end,
                )
            )
        return tuple(hits)

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

    def commit_shard(
        self,
        stats: ShardScanStats,
        hits: Sequence[HitRecord] = (),
    ) -> None:
        """Commit duplicate state, shard statistics, hits, and restart marker together."""

        self._require_open()
        source_file_path = self._active_source_file_path
        if source_file_path is None:
            raise RuntimeError("no shard transaction is active")
        if stats.path != source_file_path:
            raise SourceOrderError("shard evidence path does not match active shard")
        if not stats.complete:
            raise SourceOrderError("only complete shard evidence may be committed")
        evidence = stats.as_dict()
        evidence_payload = canonical_json_bytes(evidence).decode("utf-8")
        evidence_digest = hashlib.sha256(evidence_payload.encode("utf-8")).hexdigest()
        self._connection.execute(
            """
            INSERT INTO completed_shards (
                source_file_path,
                evidence_json,
                evidence_sha256
            ) VALUES (?, ?, ?)
            """,
            (source_file_path, evidence_payload, evidence_digest),
        )
        for hit in hits:
            if hit.source_file_path != source_file_path:
                raise SourceOrderError("hit source path does not match active shard")
            if hit.physical_row_index >= stats.physical_rows_visited:
                raise SourceOrderError("hit row index exceeds shard rows visited")
            self._connection.execute(
                """
                INSERT INTO contamination_hits (
                    source_file_path,
                    prompt_id,
                    check_name,
                    document_id,
                    physical_row_index,
                    match_start_codepoint,
                    match_end_codepoint
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_file_path,
                    hit.prompt_id,
                    hit.check,
                    hit.document_id,
                    hit.physical_row_index,
                    hit.match_start_codepoint,
                    hit.match_end_codepoint,
                ),
            )
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
