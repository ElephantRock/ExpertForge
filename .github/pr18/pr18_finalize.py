"""Apply the final PR #18 contract corrections on the checked-out branch."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Source evidence and remote-warning invariants
# ---------------------------------------------------------------------------
source_path = "src/expertforge/provenance/source_snapshot.py"
source = read(source_path)

source = replace_once(
    source,
    '''        # Limitations: sorted.\n        if list(self.limitations) != sorted(self.limitations):\n            raise ValueError(f"limitations must be sorted; got {list(self.limitations)!r}.")\n''',
    '''        # Limitations: sorted + unique. Duplicate limitations would make the\n        # behavioral envelope non-canonical even if their set of meanings were\n        # unchanged.\n        if list(self.limitations) != sorted(self.limitations):\n            raise ValueError(f"limitations must be sorted; got {list(self.limitations)!r}.")\n        if len(set(self.limitations)) != len(self.limitations):\n            raise ValueError(f"limitations must be unique; got {list(self.limitations)!r}.")\n''',
    label="source limitation uniqueness",
)

source_evidence_marker = "        return self\n\n\nclass RemoteWarning"
source_evidence_insert = '''        # Represented incomplete facts MUST carry their exact durable\n        # explanation. Existing warning↔limitation pairing and count checks are\n        # insufficient when an unrelated warning is used to mask a missing pair.\n        unreadable_limits = [\n            lim\n            for lim in self.limitations\n            if lim.startswith("unreadable_untracked_files:")\n        ]\n        has_unreadable_warning = "unreadable_untracked_content" in self.warnings\n        if unreadable_actual > 0:\n            expected = f"unreadable_untracked_files:{unreadable_actual}"\n            if unreadable_limits != [expected] or not has_unreadable_warning:\n                raise ValueError(\n                    "unreadable untracked entries require both "\n                    "unreadable_untracked_content and exactly "\n                    f"{expected!r}; got warnings={list(self.warnings)!r}, "\n                    f"limitations={list(self.limitations)!r}."\n                )\n        elif unreadable_limits or has_unreadable_warning:\n            raise ValueError(\n                "unreadable_untracked_content and unreadable_untracked_files:N "\n                "are forbidden when no untracked entry has digest=None."\n            )\n\n        dirty_limits = [\n            lim\n            for lim in self.limitations\n            if lim.startswith("dirty_submodules_not_snapshotted:")\n        ]\n        has_dirty_warning = "dirty_submodule_content_not_captured" in self.warnings\n        if dirty_actual > 0:\n            expected = f"dirty_submodules_not_snapshotted:{dirty_actual}"\n            if dirty_limits != [expected] or not has_dirty_warning:\n                raise ValueError(\n                    "changed/conflicted submodule entries require both "\n                    "dirty_submodule_content_not_captured and exactly "\n                    f"{expected!r}; got warnings={list(self.warnings)!r}, "\n                    f"limitations={list(self.limitations)!r}."\n                )\n        elif dirty_limits or has_dirty_warning:\n            raise ValueError(\n                "dirty_submodule_content_not_captured and "\n                "dirty_submodules_not_snapshotted:N are forbidden when no "\n                "submodule entry is changed/conflicted."\n            )\n'''
if source_evidence_marker not in source:
    raise RuntimeError("SourceEvidence return marker not found")
source = source.replace(
    source_evidence_marker,
    source_evidence_insert + source_evidence_marker,
    1,
)

source_snapshot_start = source.index("class SourceSnapshot(BaseModel):")
source_digest_marker = '''        return self\n\n    @model_validator(mode="after")\n    def _verify_input_digest'''
marker_index = source.index(source_digest_marker, source_snapshot_start)
remote_insert = '''        # Enforce the remote-warning relationships that remain observable after\n        # sanitization. Historical credential/query redactions may accompany a\n        # non-null sanitized locator. Local/unsupported redaction necessarily\n        # omits the locator, and an omitted locator carrying other redaction\n        # observations must also explain that omission.\n        remote_codes = {warning.code for warning in self.remote_warnings}\n        has_local_removed = "remote_local_or_unsupported_removed" in remote_codes\n        has_supported_redaction = bool(\n            remote_codes\n            & {"remote_credentials_removed", "remote_query_fragment_removed"}\n        )\n        if self.remote_url is not None and has_local_removed:\n            raise ValueError(\n                "remote_local_or_unsupported_removed requires remote_url=None; "\n                "a non-null sanitized locator cannot carry that observation."\n            )\n        if self.remote_url is None and has_supported_redaction and not has_local_removed:\n            raise ValueError(\n                "credential/query warnings with remote_url=None require "\n                "remote_local_or_unsupported_removed to explain the omitted locator."\n            )\n'''
source = source[:marker_index] + remote_insert + source[marker_index:]
write(source_path, source)

# Replace the old regression that deliberately accepted an impossible remote.
test_source_path = "tests/test_source_snapshot.py"
test_source = read(test_source_path)
pattern = re.compile(
    r"    def test_none_url_with_warnings_accepted_by_model\(self\) -> None:\n.*?"
    r"(?=    def test_scp_style_remote_preserved_verbatim)",
    re.DOTALL,
)
replacement = '''    def test_none_url_with_credentials_warning_requires_local_removal(self) -> None:\n        from expertforge.provenance.source_snapshot import (\n            RemoteWarning,\n            SourceSnapshot,\n        )\n\n        tree = "c" * 64\n        with pytest.raises(ValidationError):\n            SourceSnapshot(\n                commit_sha="1" * 40,\n                is_clean=True,\n                is_canonical=True,\n                remote_url=None,\n                remote_warnings=(RemoteWarning(code="remote_credentials_removed"),),\n                tree_digest=tree,\n                input_digest=_envelope_digest_for(tree, None),\n            )\n\n'''
test_source, replaced = pattern.subn(replacement, test_source, count=1)
if replaced != 1:
    raise RuntimeError(f"remote acceptance regression replacement count={replaced}")
write(test_source_path, test_source)

# ---------------------------------------------------------------------------
# Status-specific error reasons
# ---------------------------------------------------------------------------
record_path = "src/expertforge/provenance/record.py"
record = read(record_path)
record = replace_once(
    record,
    '''_ACCELERATOR_REASON = Literal[\n    "io_error",\n    "timeout",\n    "decode_error",\n    "duplicate_device_ordinals",\n    "not_found",\n]\n''',
    '''_ACCELERATOR_REASON = Literal[\n    "io_error",\n    "timeout",\n    "decode_error",\n    "duplicate_device_ordinals",\n]\n''',
    label="accelerator reason domain",
)
record = replace_once(
    record,
    '    reason: Literal["io_error", "not_found"] | None = Field(default=None)\n',
    '    reason: Literal["io_error"] | None = Field(default=None)\n',
    label="lockfile reason domain",
)
record = record.replace(
    "('io_error' or 'not_found'); got None.",
    "'io_error'; got None.",
)
record = record.replace(
    "('io_error', 'timeout', 'decode_error', "
    "'duplicate_device_ordinals', or 'not_found'); got None.",
    "('io_error', 'timeout', 'decode_error', "
    "or 'duplicate_device_ordinals'); got None.",
)
write(record_path, record)

hardware_path = "src/expertforge/provenance/hardware.py"
hardware = read(hardware_path)
hardware = replace_once(
    hardware,
    '''    ) -> Literal["io_error", "timeout", "decode_error", "duplicate_device_ordinals", "not_found"]:\n''',
    '''    ) -> Literal["io_error", "timeout", "decode_error", "duplicate_device_ordinals"]:\n''',
    label="stable reason return type",
)
hardware = replace_once(
    hardware,
    '''    if isinstance(exc, FileNotFoundError):\n        return "not_found"\n''',
    "",
    label="stable reason not_found branch",
)
write(hardware_path, hardware)

# Add focused direct-model and authoritative-loader regressions.
new_test = ROOT / "tests/test_provenance_final_contracts.py"
new_test.write_text(
    (ROOT / ".github/pr18/test_provenance_final_contracts.py").read_text(encoding="utf-8"),
    encoding="utf-8",
)

# The execution harness is branch-local and must not remain in the final tree.
(ROOT / ".github/workflows/pr18-finalize.yml").unlink()
(ROOT / ".github/pr18/test_provenance_final_contracts.py").unlink()
Path(__file__).unlink()
