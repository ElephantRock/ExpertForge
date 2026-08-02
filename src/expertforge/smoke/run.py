"""One-command smoke gate CLI (Issue #14, amendment L + correction #10).

The designated gate command:

    uv run --locked python -m expertforge.smoke.run --artifact-root runs/m0-smoke

Runs U0 (baseline), R0 (interrupted), R1 (resumed), compares U0@N vs R1@N, and
publishes the comparison report as an R1 report artifact (before R1's manifest).
The command NEVER deletes or overwrites prior canonical runs: the artifact root
is a base directory under which identity-derived attempt paths coexist. Tests
use isolated temporary roots.

Exit code 0 = gate PASS (computational digests equal AND loss-movement criterion
met); non-zero = FAIL. The committed ``reports/milestone-0-smoke-gate.md`` is
produced separately from the accepted evidence run and is never rewritten by
this CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from expertforge.config.resolve import resolve_config
from expertforge.provenance.orchestrate import prepare_run
from expertforge.smoke.comparison import compare
from expertforge.smoke.report import ComparisonReport
from expertforge.smoke.runner import SmokeRunner

__all__ = ["main", "build_runner", "run_gate"]


def build_runner(
    *,
    artifact_root: Path,
    config_path: str | Path,
    repo: Path,
    allow_dirty: bool = False,
) -> SmokeRunner:
    """Resolve the gate config, validate the threshold, and build a SmokeRunner."""
    env = resolve_config(config_path)
    cfg = env.config
    threshold = cfg.evaluation.loss_improvement_threshold
    if threshold is None:
        raise SystemExit(
            "configs/m0-smoke-gate.yaml must declare evaluation.loss_improvement_threshold "
            "(amendment J)."
        )
    seq_len = cfg.training.seq_len or cfg.data.seq_len
    tokens_per_update = cfg.training.batch_size * seq_len
    if cfg.training.tokens % tokens_per_update != 0:
        raise SystemExit(
            f"training.tokens ({cfg.training.tokens}) must be divisible by "
            f"tokens_per_update (batch_size*seq_len = {tokens_per_update})."
        )
    if cfg.checkpointing.interval_tokens % tokens_per_update != 0:
        raise SystemExit(
            f"checkpointing.interval_tokens ({cfg.checkpointing.interval_tokens}) must be "
            f"divisible by tokens_per_update ({tokens_per_update})."
        )
    n_updates = cfg.training.tokens // tokens_per_update
    k_updates = cfg.checkpointing.interval_tokens // tokens_per_update
    if not (0 < k_updates < n_updates):
        raise SystemExit(f"require 0 < K_updates ({k_updates}) < N_updates ({n_updates}).")
    return SmokeRunner(
        artifact_root=artifact_root,
        config_envelope=env,
        repo=repo,
        prepare_run_fn=prepare_run,
        loss_threshold=float(threshold),
        k_updates=k_updates,
        n_updates=n_updates,
        tokens_per_update=tokens_per_update,
        allow_dirty=allow_dirty,
    )


def run_gate(
    runner: SmokeRunner, *, force_mismatch: bool = False
) -> tuple[bool, ComparisonReport | None, dict[str, object]]:
    """Run U0/R0/R1, compare, and publish the R1 comparison report.

    Returns ``(passed, report_or_None, summary)`` where ``summary`` records the
    observed losses, digests, and mismatch list for the committed report.
    ``force_mismatch`` is a fault-injection hook (defaults False) that perturbs
    R1's continuation to drive the canonical failing-comparison path.
    """
    u0 = runner.run_u0()
    r0 = runner.run_r0()
    r1 = runner.run_r1(r0, u0, force_mismatch=force_mismatch)

    comparison = compare(u0.computational_state, r1.computational_state)  # type: ignore[arg-type]
    computational_equal = bool(comparison["equal"])

    # Loss-movement criterion (amendment J): finite initial/final, final improves
    # by at least the predeclared threshold.
    initial_loss = float(u0.initial_validation_loss)  # type: ignore[arg-type]
    final_loss = float(u0.final_validation_loss)  # type: ignore[arg-type]
    import math

    loss_ok = (
        math.isfinite(initial_loss)
        and math.isfinite(final_loss)
        and (initial_loss - final_loss) >= runner.loss_threshold
    )

    passed = computational_equal and loss_ok

    # The comparison report is published by run_r1 as an R1 report artifact
    # (before R1's terminal manifest). Re-expose it here for the summary/CLI.
    report: ComparisonReport | None = r1.comparison_report

    summary: dict[str, object] = {
        "u0_run_id": u0.identity.run_id,
        "u0_attempt_id": u0.identity.attempt_id,
        "r0_run_id": r0.identity.run_id,
        "r0_attempt_id": r0.identity.attempt_id,
        "r1_run_id": r1.identity.run_id,
        "r1_attempt_id": r1.identity.attempt_id,
        "specification_fingerprint": u0.identity.fingerprint_digest_str(),
        "initial_validation_loss": initial_loss,
        "u0_final_validation_loss": final_loss,
        "r1_final_validation_loss": float(r1.final_validation_loss),  # type: ignore[arg-type]
        "loss_improvement": initial_loss - final_loss,
        "loss_threshold": runner.loss_threshold,
        "loss_movement_ok": loss_ok,
        "computational_equal": computational_equal,
        "u0_computational_digest": u0.computational_state.computational_digest,  # type: ignore[union-attr]
        "r1_computational_digest": r1.computational_state.computational_digest,  # type: ignore[union-attr]
        "comparison_mismatches": comparison["mismatches"],
        "u0_manifest_id": u0.manifest_record.artifact_id,
        "r0_manifest_id": r0.manifest_record.artifact_id,
        "r1_manifest_id": r1.manifest_record.artifact_id,
        "passed": passed,
    }
    return passed, report, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m expertforge.smoke.run",
        description="Milestone 0 end-to-end smoke & recovery gate (Issue #14).",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Base directory under which identity-derived attempt paths coexist.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m0-smoke-gate.yaml"),
        help="Path to the dedicated gate configuration.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("."),
        help="Repository root for source-snapshot provenance.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty source tree (development only; never for the evidence run).",
    )
    parser.add_argument(
        "--force-mismatch",
        action="store_true",
        help=(
            "Fault-injection: perturb R1's continuation so U0@N vs R1@N diverges, "
            "driving the canonical failing-comparison path (decision=fail, exit 1). "
            "For testing the failure-reporting machinery only; never for the evidence run."
        ),
    )
    args = parser.parse_args(argv)

    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    runner = build_runner(
        artifact_root=artifact_root,
        config_path=args.config,
        repo=args.repo.resolve(),
        allow_dirty=args.allow_dirty,
    )
    passed, report, summary = run_gate(runner, force_mismatch=args.force_mismatch)
    # Print a concise summary to stdout for the CI job and the report author.
    import json

    print(json.dumps(summary, indent=2, sort_keys=True))
    if report is not None:
        print("COMPARISON_REPORT:")
        print(report.canonical_bytes().decode("utf-8"))
    if not passed:
        print("SMOKE GATE: FAIL", file=sys.stderr)
        return 1
    print("SMOKE GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
