"""Deterministic repository checks used locally and by CI.

The checks are intentionally network-free and operate only on tracked repository
content. They provide stable, sanitized diagnostics suitable for pull-request
validation without depending on machine-specific paths, credentials, or GPUs.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

_REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_SCAN_SCOPES: Final[tuple[str, ...]] = (
    ".github",
    "configs",
    "schemas",
    "scripts",
    "src",
    "tests",
    "pyproject.toml",
)


@dataclass(frozen=True, order=True)
class Finding:
    """One stable repository-validation finding."""

    path: str
    rule: str


_SECRET_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{36,255}\b")),
    (
        "github-fine-grained-token",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,255}\b"),
    ),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    (
        "credential-assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|"
            r"secret[_-]?key)\b\s*[:=]\s*[\"'][^\"'\r\n]{12,}[\"']"
        ),
    ),
)

_EXPERTOS_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "expertos-absolute-path",
        re.compile(r"(?i)(?:\b[A-Z]:[\\/]+ExpertOS(?:[\\/]|$)|/ExpertOS(?:/|$))"),
    ),
    (
        "expertos-python-import",
        re.compile(
            r"(?m)^\s*(?:from\s+expertos(?:\.|\s)|import\s+expertos(?:\.|\s|$))",
            re.IGNORECASE,
        ),
    ),
)


class RepositoryValidationError(RuntimeError):
    """Raised when a deterministic repository check cannot be completed."""


def _git_environment() -> dict[str, str]:
    """Return a deterministic, credential-minimized environment for Git reads."""

    allowed = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT")
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return env


def _tracked_files(root: Path, scopes: Sequence[str] = _SCAN_SCOPES) -> tuple[Path, ...]:
    """Return sorted tracked files under ``scopes`` without following symlinks."""

    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *scopes],
        cwd=root,
        env=_git_environment(),
        capture_output=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        raise RepositoryValidationError("git_ls_files_failed")
    try:
        names = result.stdout.decode("utf-8").split("\0")
    except UnicodeDecodeError as exc:
        raise RepositoryValidationError("git_ls_files_invalid_utf8") from exc
    return tuple(root / name for name in sorted(name for name in names if name))


def _read_text_without_following(path: Path) -> str | None:
    """Read tracked text safely; return ``None`` for binary content."""

    try:
        if path.is_symlink():
            return os.readlink(path)
        data = path.read_bytes()
    except OSError as exc:
        raise RepositoryValidationError(f"tracked_file_unreadable:{path.as_posix()}") from exc
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepositoryValidationError(f"tracked_text_invalid_utf8:{path.as_posix()}") from exc


def _find_patterns(
    *, relative_path: str, text: str, patterns: Iterable[tuple[str, re.Pattern[str]]]
) -> tuple[Finding, ...]:
    return tuple(Finding(relative_path, rule) for rule, pattern in patterns if pattern.search(text))


def find_secret_findings(root: Path) -> tuple[Finding, ...]:
    """Scan tracked implementation/configuration content for high-confidence secrets."""

    findings: list[Finding] = []
    for path in _tracked_files(root):
        text = _read_text_without_following(path)
        if text is None:
            continue
        relative = path.relative_to(root).as_posix()
        findings.extend(
            _find_patterns(relative_path=relative, text=text, patterns=_SECRET_PATTERNS)
        )
    return tuple(sorted(findings))


def find_expertos_boundary_findings(root: Path) -> tuple[Finding, ...]:
    """Reject direct ExpertOS source/runtime coupling in executable repository content."""

    findings: list[Finding] = []
    for path in _tracked_files(root):
        relative = path.relative_to(root).as_posix()
        if any(part.casefold() == "expertos" for part in Path(relative).parts):
            findings.append(Finding(relative, "expertos-path-component"))
        text = _read_text_without_following(path)
        if text is None:
            continue
        findings.extend(
            _find_patterns(relative_path=relative, text=text, patterns=_EXPERTOS_PATTERNS)
        )
    return tuple(sorted(findings))


def validate_issue_templates(root: Path) -> tuple[str, ...]:
    """Parse every issue-template YAML file and require a mapping root."""

    template_dir = root / ".github" / "ISSUE_TEMPLATE"
    paths = tuple(sorted((*template_dir.glob("*.yml"), *template_dir.glob("*.yaml"))))
    if not paths:
        raise RepositoryValidationError("issue_templates_missing")
    validated: list[str] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as stream:
                document = yaml.safe_load(stream)
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise RepositoryValidationError(
                f"issue_template_invalid:{path.relative_to(root).as_posix()}"
            ) from exc
        if not isinstance(document, dict):
            raise RepositoryValidationError(
                f"issue_template_root_not_mapping:{path.relative_to(root).as_posix()}"
            )
        validated.append(path.relative_to(root).as_posix())
    return tuple(validated)


def _render_findings(name: str, findings: Sequence[Finding]) -> str:
    lines = [f"{name}: {len(findings)} finding(s)"]
    lines.extend(f"{finding.path}: {finding.rule}" for finding in findings)
    return "\n".join(lines)


def run(command: str, root: Path) -> int:
    """Run one named validation command and emit deterministic output."""

    try:
        if command in {"issue-templates", "all"}:
            validated = validate_issue_templates(root)
            print(f"issue templates: {len(validated)} parsed")
        if command in {"secrets", "all"}:
            findings = find_secret_findings(root)
            if findings:
                print(_render_findings("secret scan", findings), file=sys.stderr)
                return 1
            print("secret scan: 0 hits")
        if command in {"expertos-boundary", "all"}:
            findings = find_expertos_boundary_findings(root)
            if findings:
                print(_render_findings("ExpertOS boundary audit", findings), file=sys.stderr)
                return 1
            print("ExpertOS boundary audit: 0 hits")
    except RepositoryValidationError as exc:
        print(f"repository validation error: {exc}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("all", "issue-templates", "secrets", "expertos-boundary"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPOSITORY_ROOT,
        help="Repository root (defaults to the parent of scripts/).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args.command, args.root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
