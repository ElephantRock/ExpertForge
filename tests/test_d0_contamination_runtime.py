from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from expertforge.d0.contamination import ContaminationMatcher
from scripts.validate_d0_generation_prompts import scan_document_for_prompt_contamination

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "experiments/d0/generation-prompts-v1.json"


def _manifest() -> Mapping[str, Any]:
    value = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _prompt(manifest: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    prompts = manifest["prompts"]
    assert isinstance(prompts, list)
    prompt = prompts[index]
    assert isinstance(prompt, dict)
    return prompt


@pytest.mark.parametrize("case", ["full", "probe", "window", "none"])
def test_optimized_matcher_matches_ratified_reference(case: str) -> None:
    manifest = _manifest()
    matcher = ContaminationMatcher.from_manifest(manifest)
    prompt = _prompt(manifest, 0)
    text = prompt["text"]
    probe = prompt["contamination_probe_text"]
    assert isinstance(text, str)
    assert isinstance(probe, str)
    if case == "full":
        document = f"prefix {text} suffix"
    elif case == "probe":
        document = f"prefix {probe} suffix"
    elif case == "window":
        document = f"prefix {text[:64]} suffix"
    else:
        document = "a document with no frozen generation-prompt material"

    expected = scan_document_for_prompt_contamination(
        manifest,
        document_id="doc-1",
        source_file_path="sample/000.parquet",
        physical_row_index=7,
        text=document,
    )
    actual = [
        hit.as_dict()
        for hit in matcher.scan_document(
            document_id="doc-1",
            source_file_path="sample/000.parquet",
            physical_row_index=7,
            text=document,
        )
    ]
    assert actual == expected


def test_matcher_applies_shared_newline_normalization() -> None:
    manifest = _manifest()
    matcher = ContaminationMatcher.from_manifest(manifest)
    prompt = _prompt(manifest, 5)
    text = prompt["text"]
    assert isinstance(text, str)
    crlf_document = text.replace("\n", "\r\n")
    hits = matcher.scan_document(
        document_id="doc-newlines",
        source_file_path="sample/000.parquet",
        physical_row_index=0,
        text=crlf_document,
    )
    selected = [hit for hit in hits if hit.prompt_id == prompt["id"]]
    assert len(selected) == 1
    assert selected[0].check == "exact_normalized_prompt_substring"
