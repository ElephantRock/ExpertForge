#!/usr/bin/env python
"""Measure the Milestone 0 smoke-gate wall-clock and peak memory (Issue #14, B4).

Reproducible resource-measurement helper for the acceptance report. Runs the
full U0/R0/R1 gate in-process under ``tracemalloc`` and prints:

    GATE wall-clock: <s>
    GATE tracemalloc peak: <MiB>

``tracemalloc`` counts only Python-allocated heap; process RSS is higher due to
the interpreter and NumPy runtime overhead. Use ``--artifact-root`` for an
isolated run directory (generated artifacts are gitignored). The reported
wall-clock is the in-process gate; the one-command CLI subprocess adds
interpreter-startup overhead.

Usage:
    uv run --locked python scripts/measure_smoke_gate.py \
        --artifact-root runs/m0-smoke-measure --config configs/m0-smoke-gate.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

from expertforge.smoke.run import build_runner, run_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("runs/m0-smoke-measure"),
        help="Base directory under which identity-derived attempt paths coexist.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m0-smoke-gate.yaml"),
        help="Path to the dedicated gate configuration.",
    )
    parser.add_argument("--repo", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow a dirty source tree.")
    args = parser.parse_args(argv)

    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    runner = build_runner(
        artifact_root=artifact_root,
        config_path=args.config,
        repo=args.repo.resolve(),
        allow_dirty=args.allow_dirty,
    )
    tracemalloc.start()
    t0 = time.perf_counter()
    passed, _report, _summary = run_gate(runner)
    wall = time.perf_counter() - t0
    _curr, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"GATE wall-clock: {wall:.2f} s")
    print(f"GATE tracemalloc peak: {peak / (1024 * 1024):.1f} MiB")
    print(f"passed: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
