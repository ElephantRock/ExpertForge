"""Tests for the configuration CLI entrypoint (Issue #5).

The CLI wraps the resolver: given a config path and repeatable --set overrides,
it resolves the configuration and emits the resolution envelope and the
canonical behavioral bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from expertforge.config.cli import build_parser, run_cli

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
FIXTURE = CONFIGS_DIR / "smoke.yaml"


class TestFixtureConfig:
    def test_committed_fixture_resolves(self) -> None:
        # The fixture under configs/ must resolve cleanly (acceptance criterion).
        from expertforge.config.resolve import resolve_config

        env = resolve_config(FIXTURE)
        assert env.config.run.name == "smoke-fixture"


class TestCLIBasic:
    def test_resolve_emits_envelope_and_canonical(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = run_cli([str(FIXTURE)])
        assert rc == 0
        out = capsys.readouterr().out
        # The whole output is a JSON object.
        obj = json.loads(out)
        assert "source_path" in obj
        assert "content_hash" in obj
        # canonical_bytes is a string whose contents are themselves valid JSON.
        assert isinstance(obj["canonical_bytes"], str)
        parsed = json.loads(obj["canonical_bytes"])
        assert parsed["run"]["name"] == "smoke-fixture"

    def test_cli_applies_set_overrides(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = run_cli([str(FIXTURE), "--set", "training.seed=99"])
        assert rc == 0
        obj = json.loads(capsys.readouterr().out)
        # The canonical bytes reflect the overridden seed.
        parsed = json.loads(obj["canonical_bytes"])
        assert parsed["training"]["seed"] == 99

    def test_cli_rejects_unknown_override_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = run_cli([str(FIXTURE), "--set", "run.bogus=1"])
        assert rc != 0
        err = capsys.readouterr().err
        assert "unknown path" in err.lower() or "bogus" in err.lower()

    def test_cli_reports_invalid_config(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "run:\n  name: x\nmodel:\n  dim: 5\n  n_layers: 1\n  n_heads: 2\n  ffn_dim: 4\n"
            "training:\n  seed: 1\n  tokens: 100\n  batch_size: 1\n  lr: 0.001\n",
            encoding="utf-8",
        )
        rc = run_cli([str(bad)])
        assert rc != 0
        assert "dim" in capsys.readouterr().err


class TestParser:
    def test_repeated_set_collected(self) -> None:
        p = build_parser()
        ns = p.parse_args(["cfg.yaml", "--set", "a.b=1", "--set", "c.d=2"])
        assert ns.set == ["a.b=1", "c.d=2"]


class TestCLIInputErrorsAreCleanDiagnostics:
    """Regression: input errors must produce concise diagnostics + non-zero
    exit, NOT uncaught tracebacks. Covers: missing file, invalid UTF-8,
    malformed YAML, malformed override, duplicate override."""

    def test_missing_file_is_clean_diagnostic(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = run_cli(["definitely_missing.yaml"])
        assert rc != 0
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "error" in err.lower()

    def test_malformed_yaml_is_clean_diagnostic(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("a: [unclosed\n", encoding="utf-8")
        rc = run_cli([str(bad)])
        assert rc != 0
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "error" in err.lower()

    def test_invalid_utf8_is_clean_diagnostic(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.yaml"
        # Invalid UTF-8 byte sequence.
        bad.write_bytes(b"run:\n  name: \xff\xfe broken\n")
        rc = run_cli([str(bad)])
        assert rc != 0
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "error" in err.lower()

    def test_malformed_override_is_clean_diagnostic(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = run_cli([str(FIXTURE), "--set", "no_equals_here"])
        assert rc != 0
        err = capsys.readouterr().err
        assert "Traceback" not in err

    def test_duplicate_override_is_clean_diagnostic(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = run_cli([str(FIXTURE), "--set", "training.seed=1", "--set", "training.seed=2"])
        assert rc != 0
        err = capsys.readouterr().err
        assert "Traceback" not in err

    def test_non_finite_override_is_clean_diagnostic(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = run_cli([str(FIXTURE), "--set", "training.lr=Infinity"])
        assert rc != 0
        err = capsys.readouterr().err
        assert "Traceback" not in err

    def test_surrogate_yaml_is_clean_diagnostic(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Valid UTF-8 file whose YAML uses a \uD800 escape. Currently rejected
        # at Pydantic validation; if a future field accepted the surrogate, the
        # serialization boundary must still produce a clean diagnostic.
        bad = tmp_path / "surrogate.yaml"
        bad.write_bytes(
            b'run:\n  name: "\\uD800"\n'
            b"model:\n  dim: 64\n  n_layers: 2\n  n_heads: 2\n  ffn_dim: 128\n"
            b"training:\n  seed: 1\n  tokens: 1024\n  batch_size: 4\n  lr: 0.001\n"
        )
        rc = run_cli([str(bad)])
        assert rc != 0
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        # No partial JSON output should be written on failure.
        assert captured.out == ""

    def test_no_partial_output_on_serialization_failure(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force canonical_bytes to fail AFTER resolution succeeds, and confirm
        # the CLI writes nothing to stdout (no partial envelope) and exits non-zero.
        import expertforge.config.cli as cli_mod
        from expertforge.config.resolve import ConfigResolutionError

        def boom(envelope: object) -> bytes:
            raise ConfigResolutionError("forced serialization failure")

        monkeypatch.setattr(cli_mod, "canonical_bytes", boom)
        rc = run_cli([str(FIXTURE)])
        assert rc != 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "forced serialization failure" in captured.err
