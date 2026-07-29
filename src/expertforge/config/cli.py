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


def _emit(envelope_obj: dict[str, object]) -> int:
    json.dump(envelope_obj, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def run_cli(argv: list[str]) -> int:
    """Resolve ``argv`` and print the envelope + canonical bytes. Return exit code.

    All resolution AND serialization happen inside the error boundary; the full
    output record is built before anything is written to stdout, so a failure
    never produces partial output.
    """
    args = build_parser().parse_args(argv)
    try:
        envelope = resolve_config(args.config, args.set)
        # Serialize inside the boundary. canonical_bytes converts encoding/JSON
        # failures to ConfigResolutionError; model_dump_json may raise
        # PydanticSerializationError, caught below.
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
    except ConfigResolutionError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    except ValueError as e:
        # Covers JSON/serialization errors not already wrapped by canonical_bytes.
        sys.stderr.write(f"error: Configuration could not be serialized: {e}\n")
        return 2
    return _emit(record)


def main() -> None:  # pragma: no cover - thin shim
    sys.exit(run_cli(sys.argv[1:]))
