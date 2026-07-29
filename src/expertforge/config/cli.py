"""Command-line entrypoint for configuration resolution (Issue #5).

Usage::

    expertforge-config <config.yaml> [--set path=value ...]

Emits a JSON object with the resolution envelope (provenance: source path,
content hash, overrides) and the canonical behavioral bytes. Exits non-zero with
a diagnostic on stderr if the configuration cannot be loaded, overridden, or
validated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from expertforge.config.resolve import (
    ConfigResolutionError,
    canonical_bytes,
    resolve_config,
)

__all__ = ["build_parser", "main", "run_cli"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="expertforge-config",
        description="Resolve and validate an ExpertForge run configuration.",
    )
    p.add_argument("config", type=Path, help="Path to the source YAML configuration.")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="path=value",
        help="Dotted-path override (repeatable). Value is JSON-parsed if valid.",
    )
    return p


def _emit(rendered: str) -> int:
    """Write a fully-rendered, validated output string to stdout in one go.

    ``rendered`` must already be the complete output (final record serialized
    and UTF-8 validated inside the error boundary); this function performs a
    single write so no partial output can ever reach stdout.
    """
    sys.stdout.write(rendered)
    return 0


def run_cli(argv: list[str]) -> int:
    """Resolve ``argv`` and print the envelope + canonical bytes. Return exit code.

    All resolution AND serialization happen inside the error boundary; the full
    output record is rendered and UTF-8 validated before anything is written to
    stdout, so a failure never produces partial output.
    """
    args = build_parser().parse_args(argv)
    try:
        envelope = resolve_config(args.config, args.set)
        # canonical_bytes converts encoding/JSON failures to ConfigResolutionError.
        canonical = canonical_bytes(envelope).decode("utf-8")
        resolved_json = envelope.config.model_dump_json()
        record = {
            "source_path": str(envelope.source_path),
            "content_hash": envelope.content_hash,
            "overrides": [
                {
                    "raw_token": o.raw_token,
                    "path": o.path,
                    "value": o.value,
                }
                for o in envelope.overrides
            ],
            "resolved_config": json.loads(resolved_json),
            "canonical_bytes": canonical,
        }
        # Render the FINAL record to a complete UTF-8 string inside the boundary.
        # json.dumps raises TypeError on non-serializable values (e.g. a set that
        # slipped through) and ValueError on encoding issues; both are caught.
        rendered = (
            (json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            .encode("utf-8")
            .decode("utf-8")
        )
    except ConfigResolutionError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    except (ValueError, TypeError) as e:
        # Covers JSON/serialization errors not already wrapped by canonical_bytes,
        # including final-record rendering failures (non-serializable values).
        sys.stderr.write(f"error: Configuration could not be serialized: {e}\n")
        return 2
    return _emit(rendered)


def main() -> None:  # pragma: no cover - thin shim
    sys.exit(run_cli(sys.argv[1:]))
