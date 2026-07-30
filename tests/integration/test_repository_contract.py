"""Portable CPU integration checks for the repository validation foundation."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]


def test_installed_config_cli_resolves_smoke_fixture() -> None:
    executable = shutil.which("expertforge-config")
    assert executable is not None

    result = subprocess.run(
        [executable, "configs/smoke.yaml"],
        cwd=_ROOT,
        capture_output=True,
        check=True,
        text=True,
        timeout=15,
    )
    payload = json.loads(result.stdout)

    assert payload["content_hash"]
    assert payload["resolved_config"]
    assert payload["canonical_bytes"]


def test_repository_policy_checks_pass_on_tracked_tree() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_repository.py", "all"],
        cwd=_ROOT,
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )

    assert "issue templates:" in result.stdout
    assert "secret scan: 0 hits" in result.stdout
    assert "ExpertOS boundary audit: 0 hits" in result.stdout
