"""Tests for the repository-native validation checks introduced by Issue #13."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.validate_repository import (
    RepositoryValidationError,
    find_expertos_boundary_findings,
    find_secret_findings,
    validate_issue_templates,
)


def _init_tracked_repository(root: Path, files: dict[str, str]) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)


def test_secret_scan_detects_high_confidence_token(tmp_path: Path) -> None:
    token = "AKIA" + ("A" * 16)
    _init_tracked_repository(tmp_path, {"src/example.py": f'VALUE = "{token}"\n'})

    findings = find_secret_findings(tmp_path)

    assert [(finding.path, finding.rule) for finding in findings] == [
        ("src/example.py", "aws-access-key")
    ]


def test_secret_scan_accepts_benign_security_vocabulary(tmp_path: Path) -> None:
    _init_tracked_repository(
        tmp_path,
        {"src/example.py": 'TOKEN_ENVIRONMENT_VARIABLE = "EXPERTFORGE_TOKEN"\n'},
    )

    assert find_secret_findings(tmp_path) == ()


def test_expertos_boundary_detects_import_and_machine_path(tmp_path: Path) -> None:
    direct_import = "import " + "expertos\n"
    machine_path = "ROOT = " + repr("C:" + "\\" + "ExpertOS" + "\\src") + "\n"
    _init_tracked_repository(
        tmp_path,
        {
            "src/imported.py": direct_import,
            "tests/test_path.py": machine_path,
        },
    )

    findings = find_expertos_boundary_findings(tmp_path)

    assert {(finding.path, finding.rule) for finding in findings} == {
        ("src/imported.py", "expertos-python-import"),
        ("tests/test_path.py", "expertos-absolute-path"),
    }


def test_issue_template_validation_parses_mapping_documents(tmp_path: Path) -> None:
    template = tmp_path / ".github" / "ISSUE_TEMPLATE" / "implementation.yml"
    template.parent.mkdir(parents=True)
    template.write_text(
        "name: Implementation\ndescription: Work item\nbody: []\n", encoding="utf-8"
    )

    assert validate_issue_templates(tmp_path) == (".github/ISSUE_TEMPLATE/implementation.yml",)


def test_issue_template_validation_rejects_non_mapping_root(tmp_path: Path) -> None:
    template = tmp_path / ".github" / "ISSUE_TEMPLATE" / "invalid.yml"
    template.parent.mkdir(parents=True)
    template.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(RepositoryValidationError, match="issue_template_root_not_mapping"):
        validate_issue_templates(tmp_path)
