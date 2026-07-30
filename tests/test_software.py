"""Tests for sanitized software-environment capture (Issue #7 review item 4)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from expertforge.provenance.record import SoftwareEnvironment
from expertforge.provenance.software import (
    capture_software_environment,
    redact_token_patterns,
    sanitize_repository_url,
)


class TestRepositoryUrlSanitization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (
                "https://github.com/ElephantRock/ExpertForge.git",
                "https://github.com/ElephantRock/ExpertForge.git",
            ),
            (
                "https://user:token@github.com/ElephantRock/ExpertForge.git",
                "https://github.com/ElephantRock/ExpertForge.git",
            ),
            (
                "https://github.com/ElephantRock/ExpertForge.git?signed=xyz",
                "https://github.com/ElephantRock/ExpertForge.git",
            ),
            (
                "https://github.com/ElephantRock/ExpertForge.git#frag",
                "https://github.com/ElephantRock/ExpertForge.git",
            ),
            ("C:/Users/secret/repos/ExpertForge", None),
            ("/home/secret/repos/ExpertForge", None),
            ("file:///home/secret/repos/ExpertForge", None),
            (
                "git@github.com:ElephantRock/ExpertForge.git",
                "git@github.com:ElephantRock/ExpertForge.git",
            ),
        ],
    )
    def test_url_sanitization(self, raw: str, expected: str | None) -> None:
        assert sanitize_repository_url(raw) == expected


class TestTokenRedaction:
    @pytest.mark.parametrize(
        "raw",
        [
            "ghp_" + "a" * 36,
            "gho_" + "a" * 36,
            "AKIA" + "A" * 16,
            "value=ghs_" + "a" * 36,
        ],
    )
    def test_known_token_patterns_redacted(self, raw: str) -> None:
        out = redact_token_patterns(raw)
        assert raw not in out
        assert "[redacted]" in out

    def test_plain_text_passes_through(self) -> None:
        assert redact_token_patterns("nothing sensitive here") == "nothing sensitive here"


class TestSoftwareCaptureAllowlist:
    def test_returns_typed_model(self) -> None:
        env = capture_software_environment()
        assert isinstance(env, SoftwareEnvironment)

    def test_excludes_env_user_host_paths(self) -> None:
        env = capture_software_environment()
        serialized = repr(env)
        assert "HOME" not in serialized
        assert "PATH" not in serialized

    def test_capture_records_python_and_platform_facts(self) -> None:
        env = capture_software_environment()
        assert env.python.version
        assert env.python.implementation
        assert env.platform.status == "available"
        assert env.platform.cpu.status in ("available", "unavailable")

    def test_dependency_versions_recorded_without_local_urls(self) -> None:
        env = capture_software_environment()
        for dep in env.dependencies:
            assert not re.search(r"(file://|/Users/|/home/|C:\\)", str(dep.version))

    def test_lockfile_digest_recorded_when_present(self, tmp_path: Path) -> None:
        env = capture_software_environment(repo_root=tmp_path)
        assert env.lockfile.status in ("available", "unavailable")

    def test_cpu_count_recorded(self) -> None:
        env = capture_software_environment()
        if env.platform.cpu.status == "available":
            assert env.platform.cpu.count is not None
            assert env.platform.cpu.count >= 1
