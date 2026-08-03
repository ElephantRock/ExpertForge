"""Acquire or verify frozen D0 sources and execute the restartable contamination scan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from expertforge.d0.errors import D0PreflightError
from expertforge.d0.scan_runner import execute_contamination_scan, load_prompt_manifest
from expertforge.d0.source_acquisition import acquire_source_inventory
from expertforge.d0.source_manifest import load_source_manifest

ROOT = Path(__file__).resolve().parents[1]
DATASET_MANIFEST_PATH = ROOT / "data/manifests/d0-fineweb-edu-sample-10bt-source-v1.json"
TOKENIZER_MANIFEST_PATH = ROOT / "tokenizers/manifests/d0-gpt-neox-v1.json"
PROMPT_MANIFEST_PATH = ROOT / "experiments/d0/generation-prompts-v1.json"

DATASET_MANIFEST_SHA256 = "85fab524b13c49da78264eb03124ce113395657d6403627e53008aade7eea7d7"
TOKENIZER_MANIFEST_SHA256 = "eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c"
PROMPT_MANIFEST_SHA256 = "c748706d9706e42bc0147d62c80f8f486ebb6aaa72fb53140b491ce7706e18c1"
PROMPT_PAYLOAD_SHA256 = "c52ef9f4420ff8160fd5f212370f46cefde71033ac38431bdad2878640070e51"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Cache root containing the frozen 14-shard FineWeb-Edu inventory.",
    )
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        required=True,
        help="Cache root containing the five frozen tokenizer files.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Durable directory for restart database, identity sidecar, and report.",
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="Full 40-character lowercase Git SHA of the exact scanner source.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional report path; defaults to WORK_DIR/contamination-report.json.",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=4096,
        help="Rows decoded per single-threaded PyArrow batch.",
    )
    parser.add_argument(
        "--acquire",
        action="store_true",
        help="Download missing frozen source files before verification. Existing cache files are reverified.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root: Path = args.dataset_root
    tokenizer_root: Path = args.tokenizer_root
    work_dir: Path = args.work_dir
    report_path: Path = args.report or (work_dir / "contamination-report.json")
    try:
        dataset_manifest = load_source_manifest(
            DATASET_MANIFEST_PATH,
            expected_manifest_sha256=DATASET_MANIFEST_SHA256,
        )
        tokenizer_manifest = load_source_manifest(
            TOKENIZER_MANIFEST_PATH,
            expected_manifest_sha256=TOKENIZER_MANIFEST_SHA256,
        )
        prompt_manifest, prompt_payload_sha256 = load_prompt_manifest(
            PROMPT_MANIFEST_PATH,
            expected_sha256=PROMPT_MANIFEST_SHA256,
        )
        if prompt_payload_sha256 != PROMPT_PAYLOAD_SHA256:
            raise D0PreflightError("frozen prompt payload identity changed")

        if args.acquire:
            acquire_source_inventory(dataset_root, dataset_manifest)
            acquire_source_inventory(tokenizer_root, tokenizer_manifest)

        report = execute_contamination_scan(
            dataset_root=dataset_root,
            tokenizer_root=tokenizer_root,
            dataset_manifest=dataset_manifest,
            tokenizer_manifest=tokenizer_manifest,
            prompt_manifest=prompt_manifest,
            prompt_manifest_sha256=PROMPT_MANIFEST_SHA256,
            prompt_payload_sha256=PROMPT_PAYLOAD_SHA256,
            state_database_path=work_dir / "contamination-state.sqlite3",
            scan_identity_path=work_dir / "contamination-scan-identity.json",
            report_path=report_path,
            scanner_source_commit=args.source_commit,
            parquet_batch_size=args.parquet_batch_size,
        )
    except (D0PreflightError, OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"D0 CONTAMINATION SCAN FAILED: {exc}\n")
        return 1

    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report.get("accepted") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
