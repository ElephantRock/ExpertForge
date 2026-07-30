"""Final Issue #7 model-boundary regressions for PR #18."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from expertforge.config.resolve import resolve_config
from expertforge.identity.record import AttemptIdentityRecord
from expertforge.provenance.orchestrate import prepare_run
from expertforge.provenance.record import AcceleratorInfo, LockfileDigest, ProvenanceRecord
from expertforge.provenance.sidecar import ProvenanceSidecarError, load_provenance_sidecar
from expertforge.provenance.source_snapshot import (
    RemoteWarning,
    SourceCounts,
    SourceEvidence,
    SourceSnapshot,
    SubmoduleEntry,
    UntrackedEntry,
    _envelope_digest,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
_FIXED = datetime(2026, 1, 1, tzinfo=UTC)
_ZERO = "0" * 64


def _standard_limitations() -> tuple[str, ...]:
    return ("external_symlink_targets_not_followed", "ignored_files_not_included")


def _source_snapshot(
    *, remote_url: str | None, remote_warnings: tuple[RemoteWarning, ...]
) -> SourceSnapshot:
    tree = "a" * 64
    return SourceSnapshot(
        commit_sha="b" * 40,
        branch="main",
        remote_url=remote_url,
        remote_warnings=remote_warnings,
        is_clean=True,
        is_canonical=True,
        tree_digest=tree,
        input_digest=_envelope_digest(tree, None),
        evidence=None,
    )


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def _prepared_run(
    tmp_path: Path, *, dirty: bool = False
) -> tuple[AttemptIdentityRecord, ProvenanceRecord, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    if dirty:
        (repo / "untracked.txt").write_text("payload\n", encoding="utf-8")
    return prepare_run(
        artifact_root=tmp_path / "runs",
        config_envelope=resolve_config(CONFIGS / "smoke.yaml"),
        repo=repo,
        allow_dirty=dirty,
        clock=lambda: _FIXED,
        entropy=lambda n: bytes(n),
    )


class TestEvidenceFactsRequireDurableExplanation:
    def test_unreadable_fact_cannot_be_hidden_by_unrelated_warning(self) -> None:
        with pytest.raises(ValidationError):
            SourceEvidence(
                staged_digest=_ZERO,
                unstaged_digest=_ZERO,
                untracked=(
                    UntrackedEntry(path="secret.bin", kind="file", mode="0o600", digest=None),
                ),
                counts=SourceCounts(untracked=1),
                completeness="partial",
                limitations=tuple(
                    sorted((*_standard_limitations(), "submodule_inspection_failed"))
                ),
                warnings=("submodule_inspection_failed",),
            )

    def test_dirty_submodule_fact_cannot_be_hidden_by_unrelated_warning(self) -> None:
        with pytest.raises(ValidationError):
            SourceEvidence(
                staged_digest=_ZERO,
                unstaged_digest=_ZERO,
                submodule_status=(
                    SubmoduleEntry(name="vendor/model", commit="1" * 40, state="changed"),
                ),
                counts=SourceCounts(submodules=1),
                completeness="partial",
                limitations=tuple(
                    sorted((*_standard_limitations(), "submodule_inspection_failed"))
                ),
                warnings=("submodule_inspection_failed",),
            )

    def test_duplicate_limitations_are_noncanonical(self) -> None:
        with pytest.raises(ValidationError):
            SourceEvidence(
                staged_digest=_ZERO,
                unstaged_digest=_ZERO,
                completeness="complete",
                limitations=(
                    "external_symlink_targets_not_followed",
                    "ignored_files_not_included",
                    "ignored_files_not_included",
                ),
            )


class TestRemoteWarningSurvivingCorrelation:
    def test_none_url_with_only_credentials_warning_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _source_snapshot(
                remote_url=None,
                remote_warnings=(RemoteWarning(code="remote_credentials_removed"),),
            )

    def test_none_url_with_local_warning_accepted(self) -> None:
        snap = _source_snapshot(
            remote_url=None,
            remote_warnings=(RemoteWarning(code="remote_local_or_unsupported_removed"),),
        )
        assert snap.remote_url is None

    def test_none_url_without_warning_is_valid_no_origin(self) -> None:
        assert _source_snapshot(remote_url=None, remote_warnings=()).remote_warnings == ()

    def test_nonnull_url_with_local_warning_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _source_snapshot(
                remote_url="https://github.com/org/repo.git",
                remote_warnings=(RemoteWarning(code="remote_local_or_unsupported_removed"),),
            )


class TestReasonStatusSemantics:
    def test_lockfile_not_found_is_not_an_error_reason(self) -> None:
        with pytest.raises(ValidationError):
            LockfileDigest(status="error", reason="not_found")  # type: ignore[arg-type]

    def test_accelerator_not_found_is_not_an_error_reason(self) -> None:
        with pytest.raises(ValidationError):
            AcceleratorInfo(status="error", reason="not_found")  # type: ignore[arg-type]


class TestAuthoritativeLoadRejectsTamperedModels:
    def _rewrite(self, path: Path, data: dict[str, object]) -> None:
        path.write_text(
            json.dumps(data, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    def test_remote_warning_impossibility_rejected_on_verified_load(self, tmp_path: Path) -> None:
        identity, provenance, path = _prepared_run(tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["source"]["remote_url"] = None
        data["source"]["remote_warnings"] = [{"code": "remote_credentials_removed"}]
        self._rewrite(path, data)
        with pytest.raises(ProvenanceSidecarError):
            load_provenance_sidecar(
                path,
                expected_identity=identity,
                expected_source_snapshot=provenance.source,
            )

    def test_unreadable_fact_masking_rejected_on_verified_load(self, tmp_path: Path) -> None:
        identity, provenance, path = _prepared_run(tmp_path, dirty=True)
        data = json.loads(path.read_text(encoding="utf-8"))
        evidence = data["source"]["evidence"]
        evidence["untracked"][0]["digest"] = None
        evidence["completeness"] = "partial"
        evidence["warnings"] = ["submodule_inspection_failed"]
        evidence["limitations"] = sorted(
            [
                "external_symlink_targets_not_followed",
                "ignored_files_not_included",
                "submodule_inspection_failed",
            ]
        )
        self._rewrite(path, data)
        with pytest.raises(ProvenanceSidecarError):
            load_provenance_sidecar(
                path,
                expected_identity=identity,
                expected_source_snapshot=provenance.source,
            )

    def test_not_found_reason_rejected_on_verified_load(self, tmp_path: Path) -> None:
        identity, provenance, path = _prepared_run(tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["software"]["lockfile"] = {
            "status": "error",
            "algorithm": None,
            "digest": None,
            "reason": "not_found",
        }
        self._rewrite(path, data)
        with pytest.raises(ProvenanceSidecarError):
            load_provenance_sidecar(
                path,
                expected_identity=identity,
                expected_source_snapshot=provenance.source,
            )
