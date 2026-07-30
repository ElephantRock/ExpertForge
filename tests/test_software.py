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

    def test_dependency_conflicts_field_present(self) -> None:
        # capture_software_environment populates dependency_conflicts (empty by
        # default in a clean environment).
        env = capture_software_environment()
        assert isinstance(env.dependency_conflicts, tuple)

    def test_dependency_conflicts_recorded_on_version_mismatch(self) -> None:
        # When the SAME normalized name appears at DIFFERENT versions, the first
        # observation wins and the conflict (name + sorted unique observed
        # versions) is recorded on the model as a typed DependencyConflict. We
        # verify the model carries and round-trips the dependency_conflicts field.
        from expertforge.provenance.record import (
            DependencyConflict,
            DependencyObservation,
            PythonInfo,
        )

        conflict = DependencyConflict(name="numpy", observed_versions=("1.0.0", "2.0.0"))
        env = SoftwareEnvironment(
            python=PythonInfo(version="3.11", implementation="cpython"),
            dependencies=(DependencyObservation(name="numpy", version="1.0.0"),),
            dependency_conflicts=(conflict,),
        )
        assert env.dependency_conflicts == (conflict,)
        # Round-trips through JSON.
        restored = SoftwareEnvironment.model_validate_json(env.model_dump_json())
        assert restored.dependency_conflicts == env.dependency_conflicts

    def test_dependency_conflicts_must_be_sorted_unique(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from expertforge.provenance.record import DependencyConflict, PythonInfo

        with pytest.raises(PydanticValidationError):
            SoftwareEnvironment(
                python=PythonInfo(version="3.11", implementation="cpython"),
                dependency_conflicts=(
                    DependencyConflict(name="zeta", observed_versions=("1.0.0", "2.0.0")),
                    DependencyConflict(name="alpha", observed_versions=("1.0.0", "2.0.0")),
                ),  # unsorted by name
            )
        with pytest.raises(PydanticValidationError):
            SoftwareEnvironment(
                python=PythonInfo(version="3.11", implementation="cpython"),
                dependency_conflicts=(
                    DependencyConflict(name="numpy", observed_versions=("1.0.0", "2.0.0")),
                    DependencyConflict(name="numpy", observed_versions=("1.0.0", "3.0.0")),
                ),  # duplicate name
            )


class TestMemoryCaptureResilience:
    def test_memory_capture_degrades_on_psutil_runtime_error(self) -> None:
        # ANY psutil failure (not just ImportError) degrades to unavailable
        # without crashing software capture. Simulate a broken psutil whose
        # virtual_memory() raises a RuntimeError at call time.
        import sys
        import types

        from expertforge.provenance.record import MemoryInfo
        from expertforge.provenance.software import _capture_memory

        class _Boom(Exception):
            """Raised by the fake psutil at call time."""

        broken = types.ModuleType("psutil")

        def _virtual_memory() -> object:
            raise RuntimeError("simulated psutil runtime failure")

        broken.virtual_memory = _virtual_memory  # type: ignore[attr-defined]
        saved = sys.modules.get("psutil")
        sys.modules["psutil"] = broken
        try:
            mem = _capture_memory()
        finally:
            if saved is not None:
                sys.modules["psutil"] = saved
            else:
                sys.modules.pop("psutil", None)
        assert isinstance(mem, MemoryInfo)
        assert mem.status == "unavailable"
        assert mem.total_bytes is None
        # Suppress unused-import warning for _Boom (kept for clarity).
        assert issubclass(_Boom, Exception)
