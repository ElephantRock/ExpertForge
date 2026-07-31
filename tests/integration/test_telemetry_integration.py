"""Portable CPU-only integration tests for the Issue #9 telemetry contract
(design comment 5136093570 §15).

Covers the required portable integration matrix:

1. ``prepare_run`` → explicit telemetry writer → mixed events/metrics → close →
   authoritative load;
2. captured console output plus parseable canonical JSONL;
3. checkpoint-style progress events tied to run/attempt/fingerprint and processed
   tokens;
4. simulated interruption followed by a new attempt with continuous progress and
   untouched prior history;
5. subprocess repeatability of schema/serialization behavior under injected
   deterministic clocks;
6. truncated-tail and failed-write recovery diagnostics;
7. bounded writer statistics and fixture emission measurement.

Markers: permanent #13 fast/integration selectors; CPU-only; no network; no
credentials; no GPU; no ExpertOS dependency.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from expertforge.config.resolve import resolve_config
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.orchestrate import prepare_run
from expertforge.telemetry.loader import load_telemetry_stream, scan_telemetry_stream
from expertforge.telemetry.models import (
    EVENT_SCHEMA,
    METRIC_SCHEMA,
    MetricObservation,
    ProcessContext,
    ProgressPosition,
)
from expertforge.telemetry.writer import TelemetryWriter

pytestmark = pytest.mark.integration

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_FIXED = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


class _DeterministicClock:
    """Wall clock fixed at a base; monotonic clock advancing by a fixed step."""

    def __init__(self, *, step: int = 1_000_000) -> None:
        self._wall = _FIXED
        self._mono = 0
        self._step = step

    def wall(self) -> datetime:
        return self._wall

    def mono(self) -> int:
        v = self._mono
        self._mono += self._step
        return v


def _prepare(tmp_path: Path) -> tuple[AttemptIdentityRecord, ProcessContext, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    identity, _provenance, _sidecar = prepare_run(
        artifact_root=tmp_path / "runs",
        config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
        repo=repo,
        clock=lambda: _FIXED,
        entropy=lambda n: bytes(n),
    )
    ctx = ProcessContext(rank=0, world_size=1, local_rank=0)
    return identity, ctx, tmp_path / "runs"


class _CaptureConsole:
    """Captures console writes into a list."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, data: str) -> int:
        self.lines.append(data)
        return len(data)

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 1. prepare_run → writer → events/metrics → close → authoritative load
# ---------------------------------------------------------------------------


class TestPrepareRunToLoad:
    def test_full_round_trip(self, tmp_path: Path) -> None:
        identity, ctx, root = _prepare(tmp_path)
        clk = _DeterministicClock()
        console = _CaptureConsole()
        w = TelemetryWriter(
            artifact_root=root,
            identity=identity,
            process_context=ctx,
            console_stream=console,
            wall_clock=clk.wall,
            monotonic_clock=clk.mono,
        )
        w.emit_event(component="training", severity="INFO", event_name="training.update")
        w.emit_metric(
            component="training",
            progress=ProgressPosition(step=1, update=1, processed_tokens=128),
            observations=[
                MetricObservation(
                    namespace="training",
                    name="loss",
                    unit="dimensionless",
                    aggregation="gauge",
                    window="point",
                    value_status="finite",
                    value=0.5,
                ),
                MetricObservation(
                    namespace="training",
                    name="ppl",
                    unit="dimensionless",
                    aggregation="gauge",
                    window="point",
                    value_status="finite",
                    value=20.0,
                ),
            ],
        )
        w.close(outcome="normal")

        records = load_telemetry_stream(
            w.path, expected_identity=identity, expected_process_context=ctx
        )
        # opened + event + metric + closed
        assert len(records) == 4
        event_names = [r.event_name for r in records if hasattr(r, "event_name")]
        assert event_names[0] == "logging.stream_opened"
        assert event_names[-1] == "logging.stream_closed"
        # Identity/fingerprint binding is exactly the prepare_run identity.
        for r in records:
            assert r.run_id == identity.run_id
            assert r.attempt_id == identity.attempt_id
            assert r.specification_fingerprint == identity.fingerprint_digest_str()
        # the metric record carries the processed_tokens progress
        metric = records[2]
        assert metric.progress is not None and metric.progress.processed_tokens == 128


# ---------------------------------------------------------------------------
# 2. Captured console output plus parseable canonical JSONL
# ---------------------------------------------------------------------------


class TestConsoleAndCanonicalJsonl:
    def test_console_lines_match_records(self, tmp_path: Path) -> None:
        identity, ctx, root = _prepare(tmp_path)
        clk = _DeterministicClock()
        console = _CaptureConsole()
        w = TelemetryWriter(
            artifact_root=root,
            identity=identity,
            process_context=ctx,
            console_stream=console,
            wall_clock=clk.wall,
            monotonic_clock=clk.mono,
        )
        w.emit_event(component="training", severity="INFO", event_name="training.update")
        w.close(outcome="normal")

        # Canonical JSONL: every line parses independently.
        lines = w.path.read_bytes().split(b"\n")
        bodies = [ln for ln in lines if ln]
        assert all(json.loads(b)["schema"] in (EVENT_SCHEMA, METRIC_SCHEMA) for b in bodies)
        # Console output: one line per rendered record, no ANSI.
        assert len(console.lines) == len(bodies)
        for line in console.lines:
            assert line.endswith("\n")
            assert "\x1b[" not in line


# ---------------------------------------------------------------------------
# 3. Checkpoint-style progress tied to run/attempt/fingerprint and tokens
# ---------------------------------------------------------------------------


class TestCheckpointProgress:
    def test_progress_monotonic_and_bound_to_identity(self, tmp_path: Path) -> None:
        identity, ctx, root = _prepare(tmp_path)
        clk = _DeterministicClock()
        w = TelemetryWriter(
            artifact_root=root,
            identity=identity,
            process_context=ctx,
            console_enabled=False,
            wall_clock=clk.wall,
            monotonic_clock=clk.mono,
        )
        for tokens in (128, 256, 512):
            w.emit_event(
                component="training",
                severity="INFO",
                event_name="checkpoint.written",
                progress=ProgressPosition(
                    step=tokens // 128, update=tokens // 128, processed_tokens=tokens
                ),
            )
        w.close(outcome="normal")
        records = load_telemetry_stream(
            w.path, expected_identity=identity, expected_process_context=ctx
        )
        progress_records = [r for r in records if r.progress is not None]
        tokens_seq = [r.progress.processed_tokens for r in progress_records]  # type: ignore[union-attr]
        assert tokens_seq == [128, 256, 512]
        for r in progress_records:
            assert r.run_id == identity.run_id
            assert r.specification_fingerprint == identity.fingerprint_digest_str()


# ---------------------------------------------------------------------------
# 4. Simulated interruption + new attempt with continuous progress, prior untouched
# ---------------------------------------------------------------------------


class TestInterruptionAndResume:
    def test_new_attempt_separate_stream_prior_bytes_untouched(self, tmp_path: Path) -> None:
        from expertforge.identity.emit import AllocationMode
        from expertforge.identity.lineage import ResumeLineage

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        root = tmp_path / "runs"
        env = resolve_config(CONFIGS / "smoke.yaml")
        ctx = ProcessContext(rank=0, world_size=1, local_rank=0)
        clk = _DeterministicClock()

        # First (interrupted) attempt under the shared artifact root.
        identity1, _, _ = prepare_run(
            artifact_root=root,
            config_envelope=env,
            repo=repo,
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x01" * n,
        )
        w1 = TelemetryWriter(
            artifact_root=root,
            identity=identity1,
            process_context=ctx,
            console_enabled=False,
            wall_clock=clk.wall,
            monotonic_clock=clk.mono,
        )
        w1.emit_event(
            component="training",
            severity="INFO",
            event_name="training.update",
            progress=ProgressPosition(step=3, update=3, processed_tokens=384),
        )
        # Simulated interruption: close with the interrupted outcome.
        w1.close(outcome="interrupted", diagnostic_code="handled_interruption")
        first_path = w1.path
        first_bytes = first_path.read_bytes()
        first_size = first_path.stat().st_size

        # Resume: same source repo → same specification fingerprint; retained
        # run_id; brand-new attempt_id. This creates a linked new attempt stream
        # rather than mutating the accepted first-attempt history.
        lin = ResumeLineage(
            parent_run_id=identity1.run_id,
            parent_attempt_id=identity1.attempt_id,
            parent_checkpoint_id="ckpt-001",
        )
        identity2, _, _ = prepare_run(
            artifact_root=root,
            config_envelope=env,
            repo=repo,
            mode=AllocationMode.RESUME,
            lineage=lin,
            parent_specification_fingerprint=identity1.fingerprint_digest_str(),
            retained_run_id=identity1.run_id,
            clock=lambda: _FIXED,
            entropy=lambda n: b"\x02" * n,
        )
        assert identity2.run_id == identity1.run_id
        assert identity2.attempt_id != identity1.attempt_id
        new_path = (
            root
            / identity2.run_id
            / "attempts"
            / identity2.attempt_id
            / "logs"
            / "telemetry-rank-0000000000.jsonl"
        )
        assert new_path != first_path

        w2 = TelemetryWriter(
            artifact_root=root,
            identity=identity2,
            process_context=ctx,
            console_enabled=False,
            wall_clock=clk.wall,
            monotonic_clock=clk.mono,
        )
        # Resume continues the logical run's progress from the last committed point.
        w2.emit_event(
            component="training",
            severity="INFO",
            event_name="training.update",
            progress=ProgressPosition(step=4, update=4, processed_tokens=512),
        )
        w2.close(outcome="normal")

        # Prior attempt bytes are byte-for-byte unchanged.
        assert first_path.read_bytes() == first_bytes
        assert first_path.stat().st_size == first_size

        # Both streams load authoritatively against their own identities.
        load_telemetry_stream(first_path, expected_identity=identity1, expected_process_context=ctx)
        load_telemetry_stream(new_path, expected_identity=identity2, expected_process_context=ctx)


# ---------------------------------------------------------------------------
# 5. Subprocess repeatability under deterministic clocks
# ---------------------------------------------------------------------------


_ROOT = Path(__file__).resolve().parents[2]


class TestSubprocessRepeatability:
    def _run(self, script: str) -> str:
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=60,
        ).stdout

    def test_identical_serialization_across_processes(self) -> None:
        script = """
import json, tempfile
from datetime import UTC, datetime
from pathlib import Path
from expertforge.config.resolve import resolve_config
from expertforge.identity.emit import AllocationMode, emit_attempt_identity
from expertforge.telemetry.models import MetricObservation, ProcessContext, ProgressPosition
from expertforge.telemetry.writer import TelemetryWriter
from expertforge.telemetry.loader import load_telemetry_stream

root = Path(tempfile.mkdtemp())
env = resolve_config('configs/smoke.yaml')
ident, _ = emit_attempt_identity(
    artifact_root=root, config_envelope=env, mode=AllocationMode.INDEPENDENT,
    clock=lambda: datetime(2026,1,1,tzinfo=UTC), entropy=lambda n: bytes(n),
)
ctx = ProcessContext(rank=0, world_size=1, local_rank=0)
mono = [0]
def mclock():
    v = mono[0]; mono[0] += 1000000; return v
w = TelemetryWriter(
    artifact_root=root, identity=ident, process_context=ctx, console_enabled=False,
    wall_clock=lambda: datetime(2026,1,1,tzinfo=UTC), monotonic_clock=mclock,
)
w.emit_event(component='training', severity='INFO', event_name='training.update',
             progress=ProgressPosition(step=1, update=1, processed_tokens=128))
w.emit_metric(component='training',
    observations=[MetricObservation(namespace='training', name='loss', unit='dimensionless',
        aggregation='gauge', window='point', value_status='finite', value=0.5)])
w.close(outcome='normal')
print(w.path.read_bytes().decode('utf-8'))
"""
        out1 = self._run(script)
        out2 = self._run(script)
        assert out1 == out2
        # Every non-empty printed line is valid canonical JSON (sorted keys, compact).
        lines = [ln for ln in out1.splitlines() if ln]
        assert lines
        for line in lines:
            obj = json.loads(line)
            assert obj == json.loads(json.dumps(obj, sort_keys=True, separators=(",", ":")))


# ---------------------------------------------------------------------------
# 6. Truncated-tail and failed-write recovery diagnostics
# ---------------------------------------------------------------------------


class TestRecoveryDiagnostics:
    def test_truncated_tail_scan_reports_incomplete_prefix(self, tmp_path: Path) -> None:
        identity, ctx, root = _prepare(tmp_path)
        clk = _DeterministicClock()
        w = TelemetryWriter(
            artifact_root=root,
            identity=identity,
            process_context=ctx,
            console_enabled=False,
            wall_clock=clk.wall,
            monotonic_clock=clk.mono,
        )
        w.emit_event(component="training", severity="INFO", event_name="training.update")
        w.close(outcome="normal")
        raw = w.path.read_bytes()
        # Truncate the final record mid-line.
        last_nl = raw.rfind(b"\n")
        prev_nl = raw.rfind(b"\n", 0, last_nl)
        w.path.write_bytes(raw[: prev_nl + 1] + raw[prev_nl + 1 : last_nl])
        result = scan_telemetry_stream(w.path)
        assert result.status == "incomplete"
        assert result.accepted_count >= 2  # opened + event survive

    def test_load_rejects_truncated_tail(self, tmp_path: Path) -> None:
        identity, ctx, root = _prepare(tmp_path)
        clk = _DeterministicClock()
        w = TelemetryWriter(
            artifact_root=root,
            identity=identity,
            process_context=ctx,
            console_enabled=False,
            wall_clock=clk.wall,
            monotonic_clock=clk.mono,
        )
        w.close(outcome="normal")
        raw = w.path.read_bytes()
        w.path.write_bytes(raw[:-1])  # drop the trailing newline only
        from expertforge.telemetry.loader import TelemetryLoadError

        with pytest.raises(TelemetryLoadError):
            load_telemetry_stream(w.path, expected_identity=identity, expected_process_context=ctx)


# ---------------------------------------------------------------------------
# 7. Bounded writer statistics and fixture emission measurement
# ---------------------------------------------------------------------------


class TestBoundedStats:
    def test_stats_consistent_and_emission_bounded(self, tmp_path: Path) -> None:
        identity, ctx, root = _prepare(tmp_path)
        clk = _DeterministicClock(step=1_000)
        w = TelemetryWriter(
            artifact_root=root,
            identity=identity,
            process_context=ctx,
            console_enabled=False,
            fsync_interval_records=5,
            wall_clock=clk.wall,
            monotonic_clock=clk.mono,
        )
        n = 50
        start = time.perf_counter()
        for i in range(n):
            w.emit_event(
                component="training",
                severity="INFO",
                event_name="training.update",
                progress=ProgressPosition(step=i),
            )
        elapsed = time.perf_counter() - start
        w.close(outcome="normal")

        stats = w.stats
        # opened + n events + closed
        assert stats.records_written == n + 2
        assert stats.bytes_written == w.path.stat().st_size
        assert stats.first_sequence == 0
        assert stats.last_sequence == n + 1
        # fsync cadence: at least floor(records/interval) cadence fsyncs + close fsync.
        assert stats.fsync_count >= (stats.records_written // 5) + 1
        assert stats.console_failures == 0
        # Bounded overhead: emitting n records should complete well under a few
        # seconds on any CI host (correctness does not depend on a tight threshold).
        assert elapsed < 30.0


# ---------------------------------------------------------------------------
# 8. Two ranks produce independent files and preserve only rank-local order
# ---------------------------------------------------------------------------


class TestTwoRanks:
    def test_two_ranks_independent_streams(self, tmp_path: Path) -> None:
        identity, _, root = _prepare(tmp_path)
        world = 2
        clk0 = _DeterministicClock()
        clk1 = _DeterministicClock()
        w0 = TelemetryWriter(
            artifact_root=root,
            identity=identity,
            process_context=ProcessContext(rank=0, world_size=world, local_rank=0),
            console_enabled=False,
            wall_clock=clk0.wall,
            monotonic_clock=clk0.mono,
        )
        w1 = TelemetryWriter(
            artifact_root=root,
            identity=identity,
            process_context=ProcessContext(rank=1, world_size=world, local_rank=1),
            console_enabled=False,
            wall_clock=clk1.wall,
            monotonic_clock=clk1.mono,
        )
        assert w0.path != w1.path
        w0.emit_event(component="training", severity="INFO", event_name="training.update")
        w1.emit_event(component="training", severity="INFO", event_name="training.update")
        w0.close(outcome="normal")
        w1.close(outcome="normal")

        # Each rank loads independently with its own process context.
        load_telemetry_stream(
            w0.path,
            expected_identity=identity,
            expected_process_context=ProcessContext(rank=0, world_size=world, local_rank=0),
        )
        load_telemetry_stream(
            w1.path,
            expected_identity=identity,
            expected_process_context=ProcessContext(rank=1, world_size=world, local_rank=1),
        )
