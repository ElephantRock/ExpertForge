from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pytest

from expertforge.d0.errors import SourceVerificationError
from expertforge.d0.scan_runner import execute_contamination_scan
from expertforge.d0.source_manifest import SourceFileIdentity, SourceManifest

pa: Any = pytest.importorskip("pyarrow")
pq: Any = pytest.importorskip("pyarrow.parquet")

_PROMPT_TEXT = "Z" * 70
_PROMPT_PROBE = "Z" * 64


def _prompt_manifest() -> dict[str, Any]:
    return {
        "contamination_policy": {
            "checks_in_precedence_order": [
                "exact_normalized_prompt_substring",
                "exact_normalized_probe_substring",
                "any_exact_contiguous_64_codepoint_prompt_window",
            ],
            "partial_window_codepoints": 64,
        },
        "prompt_payload_sha256": "4" * 64,
        "prompts": [
            {
                "id": "synthetic-prompt",
                "text": _PROMPT_TEXT,
                "contamination_probe_text": _PROMPT_PROBE,
            }
        ],
    }


def _write_parquet(path: Path, texts: list[object]) -> None:
    rows = [
        {
            "id": f"doc-{index}",
            "text": text,
            "url": f"https://example.invalid/{index}",
            "dump": "CC-MAIN-SYNTHETIC",
            "file_path": f"crawl/{index}.warc.gz",
        }
        for index, text in enumerate(texts)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)


def _identity(path: str, local_path: Path) -> SourceFileIdentity:
    payload = local_path.read_bytes()
    return SourceFileIdentity(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _manifest(
    kind: Literal["dataset", "tokenizer"],
    identity: SourceFileIdentity,
    *,
    digest: str,
) -> SourceManifest:
    return SourceManifest(
        kind=kind,
        schema_version=(
            "expertforge-dataset-source-manifest/1"
            if kind == "dataset"
            else "expertforge-tokenizer-source-manifest/1"
        ),
        manifest_id=f"synthetic-{kind}",
        repository=f"synthetic/{kind}",
        revision="a" * 40,
        license="synthetic-test-only",
        configuration="sample" if kind == "dataset" else None,
        split="train" if kind == "dataset" else None,
        source_root="sample" if kind == "dataset" else None,
        files=(identity,),
        total_size_bytes=identity.size_bytes,
        manifest_sha256=digest,
    )


def _scan(
    tmp_path: Path,
    texts: list[object],
) -> tuple[dict[str, object], Path]:
    dataset_root = tmp_path / "dataset"
    tokenizer_root = tmp_path / "tokenizer"
    parquet_path = dataset_root / "sample/000.parquet"
    tokenizer_path = tokenizer_root / "tokenizer.json"
    _write_parquet(parquet_path, texts)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.write_bytes(b"synthetic tokenizer")

    report_path = tmp_path / "report.json"
    report = execute_contamination_scan(
        dataset_root=dataset_root,
        tokenizer_root=tokenizer_root,
        dataset_manifest=_manifest(
            "dataset",
            _identity("sample/000.parquet", parquet_path),
            digest="1" * 64,
        ),
        tokenizer_manifest=_manifest(
            "tokenizer",
            _identity("tokenizer.json", tokenizer_path),
            digest="2" * 64,
        ),
        prompt_manifest=_prompt_manifest(),
        prompt_manifest_sha256="3" * 64,
        prompt_payload_sha256="4" * 64,
        state_database_path=tmp_path / "state.sqlite3",
        scan_identity_path=tmp_path / "scan-identity.json",
        report_path=report_path,
        scanner_source_commit="5" * 40,
        parquet_batch_size=1,
    )
    return report, report_path


def _counts(report: Mapping[str, object]) -> Mapping[str, object]:
    value = report.get("counts")
    assert isinstance(value, Mapping)
    return value


def test_complete_zero_hit_scan_is_restartable(tmp_path: Path) -> None:
    report, report_path = _scan(tmp_path, ["alpha", "alpha", "beta"])

    assert report["status"] == "complete_zero_hit"
    assert report["accepted"] is True
    counts = _counts(report)
    assert counts["physical_rows_visited"] == 3
    assert counts["duplicates_suppressed"] == 1
    assert counts["unique_documents_scanned"] == 2
    assert counts["hit_count"] == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == report

    second, _ = _scan(tmp_path, ["alpha", "alpha", "beta"])
    assert second == report


def test_complete_scan_with_hit_is_truthful_failure_evidence(tmp_path: Path) -> None:
    report, _ = _scan(tmp_path, [_PROMPT_TEXT, "safe"])

    assert report["status"] == "complete_hits_detected"
    assert report["accepted"] is False
    assert _counts(report)["hit_count"] == 1
    hits = report["hits"]
    assert isinstance(hits, list)
    assert hits[0]["check"] == "exact_normalized_prompt_substring"


def test_failed_shard_writes_incomplete_report_and_rolls_back(tmp_path: Path) -> None:
    with pytest.raises(SourceVerificationError, match="text.*non-null string"):
        _scan(tmp_path, ["safe", None])

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "incomplete"
    assert report["accepted"] is False
    assert report["observed_inventory"]["completed_shards"] == 0
