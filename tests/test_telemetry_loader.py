"""Tests for the Issue #9 telemetry loader (design comment 5136093570 §10).

Covers:

- authoritative verified load (identity/process binding, open/close lifecycle,
  contiguous sequence, non-decreasing elapsed/progress, strict validation,
  duplicate-key rejection, invalid UTF-8, non-object JSON);
- diagnostic scan reporting accepted prefix + complete|incomplete|corrupt status;
- truncated tail, missing close, record-after-close, sequence gaps, corrupt
  interior records, identity mismatch.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from expertforge.identity.fingerprint import specification_fingerprint
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.telemetry.loader import (
    ScanResult,
    TelemetryLoadError,
    load_telemetry_stream,
    scan_telemetry_stream,
)
from expertforge.telemetry.models import (
    EVENT_SCHEMA,
    EVENT_SCHEMA_VERSION,
    STREAM_FORMAT_VERSION,
    MetricObservation,
    ProcessContext,
)
from expertforge.telemetry.writer import TelemetryWriter

_VALID_RUN = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"
_VALID_ATTEMPT = "attempt-20260101t000000z-cccccccccccccccccccc"
_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _identity() -> AttemptIdentityRecord:
    fp = specification_fingerprint(b'{"x":1}')
    return AttemptIdentityRecord(
        specification_fingerprint=fp,
        run_id=_VALID_RUN,
        attempt_id=_VALID_ATTEMPT,
        created_at_utc=_TS,
    )


def _ctx(rank: int = 0, world_size: int = 1) -> ProcessContext:
    return ProcessContext(rank=rank, world_size=world_size, local_rank=rank)


class _Clock:
    def __init__(self) -> None:
        self.t = _TS
        self.m = 0

    def wall(self) -> datetime:
        return self.t

    def mono(self) -> int:
        v = self.m
        self.m += 1000
        return v


def _make_closed_stream(
    root: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    n_events: int = 1,
) -> tuple[Path, AttemptIdentityRecord, ProcessContext]:
    ident = _identity()
    ctx = _ctx(rank, world_size)
    clk = _Clock()
    w = TelemetryWriter(
        artifact_root=root,
        identity=ident,
        process_context=ctx,
        wall_clock=clk.wall,
        monotonic_clock=clk.mono,
        console_enabled=False,
    )
    for _ in range(n_events):
        w.emit_event(component="training", severity="INFO", event_name="training.update")
    w.close(outcome="normal")
    return w.path, ident, ctx


def _write_raw(path: Path, lines: list[bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for line in lines:
            f.write(line)
    return path


def _event_dict(seq: int, **overrides: Any) -> dict[str, Any]:
    base = dict(
        schema=EVENT_SCHEMA,
        schema_version=EVENT_SCHEMA_VERSION,
        stream_format_version=STREAM_FORMAT_VERSION,
        run_id=_VALID_RUN,
        attempt_id=_VALID_ATTEMPT,
        specification_fingerprint=_identity().fingerprint_digest_str(),
        component="training",
        rank=0,
        local_rank=0,
        world_size=1,
        sequence=seq,
        timestamp_utc="2026-01-01T00:00:00.000000Z",
        elapsed_ns=seq * 1000,
        progress=None,
        severity="INFO",
        event_name="training.update",
        diagnostic_code=None,
        fields=[],
        operator_message=None,
    )
    base.update(overrides)
    return base


def _opened(seq: int = 0) -> dict[str, Any]:
    return _event_dict(seq, component="logging", event_name="logging.stream_opened")


def _closed(seq: int, outcome: str = "normal") -> dict[str, Any]:
    return _event_dict(
        seq,
        component="logging",
        event_name="logging.stream_closed",
        fields=[{"name": "outcome", "value": outcome}],
    )


def _line(d: dict[str, Any]) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


# ---------------------------------------------------------------------------
# Authoritative load — happy path
# ---------------------------------------------------------------------------


class TestAuthoritativeLoad:
    def test_load_complete_stream(self, tmp_path: Path) -> None:
        path, ident, ctx = _make_closed_stream(tmp_path, n_events=2)
        records = load_telemetry_stream(path, expected_identity=ident, expected_process_context=ctx)
        # opened + 2 events + closed
        assert len(records) == 4
        names = [r.event_name for r in records if hasattr(r, "event_name")]
        assert names[0] == "logging.stream_opened"
        assert names[-1] == "logging.stream_closed"

    def test_require_closed_default_true(self, tmp_path: Path) -> None:
        path, ident, ctx = _make_closed_stream(tmp_path)
        # Default require_closed=True loads a closed stream fine.
        records = load_telemetry_stream(path, expected_identity=ident, expected_process_context=ctx)
        assert len(records) >= 2

    def test_require_closed_false_allows_missing_close(self, tmp_path: Path) -> None:
        path, ident, ctx = _make_closed_stream(tmp_path)
        # Strip the trailing close record to simulate an unclosed-but-valid prefix.
        raw = path.read_bytes()
        lines = raw.split(b"\n")
        # drop the last close line + the empty trailing element
        lines = lines[:-2] + [b""]
        path.write_bytes(b"\n".join(lines))
        # require_closed=True should fail; require_closed=False should succeed.
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(path, expected_identity=ident, expected_process_context=ctx)
        records = load_telemetry_stream(
            path,
            expected_identity=ident,
            expected_process_context=ctx,
            require_closed=False,
        )
        assert len(records) >= 1

    def test_identity_mismatch_rejected(self, tmp_path: Path) -> None:
        path, _ident, ctx = _make_closed_stream(tmp_path)
        other = AttemptIdentityRecord(
            specification_fingerprint=specification_fingerprint(b'{"x":2}'),
            run_id="run-20260101t000000z-bbbbbbbbbbbb-cccccccccccccccccccc",
            attempt_id="attempt-20260101t000000z-dddddddddddddddddddd",
            created_at_utc=_TS,
        )
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(path, expected_identity=other, expected_process_context=ctx)

    def test_process_context_mismatch_rejected(self, tmp_path: Path) -> None:
        path, ident, _ctx = _make_closed_stream(tmp_path, rank=0, world_size=1)
        wrong = ProcessContext(rank=0, world_size=2, local_rank=0)
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(path, expected_identity=ident, expected_process_context=wrong)

    def test_sequence_interleaving_preserved(self, tmp_path: Path) -> None:
        clk = _Clock()
        w = TelemetryWriter(
            artifact_root=tmp_path,
            identity=_identity(),
            process_context=_ctx(),
            wall_clock=clk.wall,
            monotonic_clock=clk.mono,
            console_enabled=False,
        )
        w.emit_event(component="t", severity="INFO", event_name="a.b")
        w.emit_metric(
            component="t",
            observations=[
                MetricObservation(
                    namespace="t",
                    name="l",
                    unit="dimensionless",
                    aggregation="gauge",
                    window="point",
                    value_status="finite",
                    value=1.0,
                )
            ],
        )
        w.emit_event(component="t", severity="INFO", event_name="c.d")
        w.close(outcome="normal")
        records = load_telemetry_stream(
            w.path, expected_identity=_identity(), expected_process_context=_ctx()
        )
        seqs = [r.sequence for r in records]
        assert seqs == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Corrupt / malformed inputs
# ---------------------------------------------------------------------------


class TestCorruptInputs:
    def test_invalid_utf_8_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        _write_raw(path, [b'{"schema":"expertforge.telemetry-event"', b"\xff\xfe\n"])
        ident, ctx = _identity(), _ctx()
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(path, expected_identity=ident, expected_process_context=ctx)
        result = scan_telemetry_stream(path)
        assert result.status == "corrupt"

    def test_non_object_json_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        _write_raw(path, [b"123\n", b"not-an-object\n"])
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )

    def test_duplicate_json_keys_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "dup.jsonl"
        bad = b'{"sequence":0,"sequence":1}\n'
        _write_raw(path, [bad])
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )

    def test_unknown_schema_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "unk.jsonl"
        rec = _opened()
        rec["schema"] = "expertforge.bogus"
        _write_raw(path, [_line(rec)])
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )

    def test_unknown_schema_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "ver.jsonl"
        rec = _opened()
        rec["schema_version"] = 999
        _write_raw(path, [_line(rec)])
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )

    def test_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.jsonl"
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )


# ---------------------------------------------------------------------------
# Lifecycle violations
# ---------------------------------------------------------------------------


class TestLifecycleViolations:
    def test_missing_open_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "noopen.jsonl"
        # first record is NOT logging.stream_opened
        _write_raw(path, [_line(_event_dict(0)), _line(_closed(1))])
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )

    def test_missing_close_incomplete(self, tmp_path: Path) -> None:
        path = tmp_path / "noclose.jsonl"
        _write_raw(path, [_line(_opened(0)), _line(_event_dict(1))])
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )
        result = scan_telemetry_stream(path)
        assert result.status == "incomplete"

    def test_record_after_close_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "after.jsonl"
        _write_raw(
            path,
            [_line(_opened(0)), _line(_closed(1)), _line(_event_dict(2))],
        )
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )
        # scan: a record after close is corrupt.
        result = scan_telemetry_stream(path)
        assert result.status == "corrupt"


# ---------------------------------------------------------------------------
# Sequence and monotonicity
# ---------------------------------------------------------------------------


class TestSequenceAndMonotonicity:
    def test_sequence_gap_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "gap.jsonl"
        _write_raw(
            path,
            [_line(_opened(0)), _line(_event_dict(2)), _line(_closed(3))],
        )
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )
        result = scan_telemetry_stream(path)
        assert result.status == "corrupt"

    def test_elapsed_regression_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "regress.jsonl"
        a = _event_dict(1, elapsed_ns=5000)
        b = _event_dict(2, elapsed_ns=1000)  # regresses
        _write_raw(path, [_line(_opened(0)), _line(a), _line(b), _line(_closed(3))])
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )

    def test_progress_regression_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "pregress.jsonl"
        a = _event_dict(1, progress={"step": 5, "update": 5, "processed_tokens": 100})
        b = _event_dict(2, progress={"step": 3, "update": 3, "processed_tokens": 80})
        _write_raw(path, [_line(_opened(0)), _line(a), _line(b), _line(_closed(3))])
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )


# ---------------------------------------------------------------------------
# Truncated tail
# ---------------------------------------------------------------------------


class TestTruncatedTail:
    def test_truncated_tail_rejected_on_load_incomplete_on_scan(self, tmp_path: Path) -> None:
        path, ident, ctx = _make_closed_stream(tmp_path, n_events=1)
        raw = path.read_bytes()
        # Truncate the last record mid-line (drop final bytes before the newline).
        truncated = raw[: raw.rfind(b"\n")]
        path.write_bytes(truncated)
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(path, expected_identity=ident, expected_process_context=ctx)
        result = scan_telemetry_stream(path)
        # A truncated trailing fragment is reported as incomplete (accepted prefix).
        assert result.status == "incomplete"
        # The accepted prefix excludes the partial record.
        assert result.accepted_count >= 2  # opened + event

    def test_non_newline_terminated_final_record_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "noterm.jsonl"
        good = _line(_opened(0)) + _line(_event_dict(1))
        # append a final record WITHOUT a newline
        bad = json.dumps(_closed(2), sort_keys=True, separators=(",", ":")).encode("utf-8")
        path.write_bytes(good + bad)
        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(
                path, expected_identity=_identity(), expected_process_context=_ctx()
            )


# ---------------------------------------------------------------------------
# Scan status matrix
# ---------------------------------------------------------------------------


class TestScanStatusMatrix:
    def test_scan_complete(self, tmp_path: Path) -> None:
        path, _, _ = _make_closed_stream(tmp_path)
        result = scan_telemetry_stream(path)
        assert result.status == "complete"
        assert result.accepted_count >= 2

    def test_scan_corrupt_interior(self, tmp_path: Path) -> None:
        path = tmp_path / "interior.jsonl"
        _write_raw(
            path,
            [_line(_opened(0)), b"garbage-not-json\n", _line(_closed(2))],
        )
        result = scan_telemetry_stream(path)
        assert result.status == "corrupt"
        # accepted prefix is just the opened record
        assert result.accepted_count == 1

    def test_scan_result_carries_accepted_count(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.jsonl"
        _write_raw(path, [_line(_opened(0)), _line(_closed(1))])
        result = scan_telemetry_stream(path)
        assert isinstance(result, ScanResult)
        assert result.accepted_count == 2
        assert result.status == "complete"
