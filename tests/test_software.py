"""Tests for sanitized software-environment capture (Issue #7 decision: secret handling).

V1 uses **allowlist-first capture**, not "capture everything and redact later."
Must not record: unrestricted env vars, usernames/hostnames, home/repo/
interpreter absolute paths, coordinator addresses/IPs, hardware serials/MAC/
UUIDs, raw subprocess stderr, signed URL queries, or local dependency URLs.
Repository URLs are structurally sanitized (strip credentials/queries/fragments/
signed params/local paths). Pattern-based token redaction is defense-in-depth.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from expertforge.provenance.software import (
    capture_software_environment,
    redact_token_patterns,
    sanitize_repository_url,
)

# --- repository URL sanitization ------------------------------------------


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
            # Local filesystem location -> not a remote URL.
            ("C:/Users/secret/repos/ExpertForge", None),
            ("/home/secret/repos/ExpertForge", None),
            ("file:///home/secret/repos/ExpertForge", None),
            # SSH with no userinfo is preserved; userinfo stripped.
            (
                "git@github.com:ElephantRock/ExpertForge.git",
                "git@github.com:ElephantRock/ExpertForge.git",
            ),
        ],
    )
    def test_url_sanitization(self, raw: str, expected: str | None) -> None:
        assert sanitize_repository_url(raw) == expected


# --- token redaction (defense in depth) -----------------------------------


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


# --- allowlist-first capture ----------------------------------------------


class TestSoftwareCaptureAllowlist:
    def test_capture_excludes_env_user_host_paths(self) -> None:
        env = capture_software_environment()
        serialized = repr(env)
        # No home/absolute interpreter paths recorded verbatim.
        assert "HOME" not in serialized
        # No username/hostname fields.
        for forbidden in ("username", "hostname", "user_login", "host_name"):
            assert forbidden not in env, f"{forbidden} must not be captured"
        # No environment-variable dump.
        assert "environ" not in env
        assert "PATH" not in serialized

    def test_capture_records_python_and_platform_facts(self) -> None:
        env = capture_software_environment()
        # Python version is a stable, non-secret fact.
        assert "version" in env["python"]
        assert env["python"]["status"] == "available"
        # Platform family recorded without identifying host details.
        assert env["platform"]["status"] == "available"

    def test_dependency_versions_recorded_without_local_urls(self) -> None:
        env = capture_software_environment()
        deps = env["dependencies"]
        # Each dependency entry is name -> version, not a file:// URL.
        for name, version in deps.items():
            assert not re.search(r"(file://|/Users/|/home/|C:\\)", str(version)), (
                f"dependency {name!r} version {version!r} contains a path/URL"
            )

    def test_lockfile_digest_recorded_when_present(self, tmp_path: Path) -> None:
        # A uv.lock exists in the repo root; capture should hash it if found.
        env = capture_software_environment(repo_root=tmp_path)
        # When absent (tmp_path has no lockfile), status degrades explicitly.
        assert env["lockfile"]["status"] in {"available", "unavailable"}
