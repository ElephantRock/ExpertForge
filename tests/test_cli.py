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
