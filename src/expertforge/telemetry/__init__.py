"""ExpertForge structured logging and metric collection (Issue #9).

This package owns the format and write/read behavior of the canonical
per-process telemetry stream. It defines three independently versioned
contracts (stream format, event schema, metric schema) and frozen, deeply
immutable Pydantic models.

Design boundary (Issue #9 design comment 5136093570):

- It does NOT mutate Python's root logger, install handlers, open files, read
  environment variables, or emit output at import time.
- It depends only on the standard library and the existing Pydantic runtime,
  plus the frozen Issue #6 identity types.
- It does NOT depend on Issue #7 provenance or Issue #10 artifact management.
  Issue #10 later registers, hashes, retains, and locates closed telemetry
  artifacts; #9 owns their format and write/read behavior.

Modules:

- :mod:`expertforge.telemetry.models` — frozen core models and closed domains;
- :mod:`expertforge.telemetry.console` — deterministic console renderer;
- :mod:`expertforge.telemetry.writer` — exclusive-create JSONL stream writer;
- :mod:`expertforge.telemetry.loader` — authoritative load + diagnostic scan.
"""

from __future__ import annotations
