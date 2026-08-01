# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue and pull-request records. When this file conflicts with the
underlying issues/PRs or with `main`, those prevail.

Last updated: Milestone 0 in progress; #12 complete, #14 is the final gate, #13 active. 9 of 11 child issues closed; 10 of 11 capabilities delivered.

## Current milestone

- **Milestone 0 — Experimental substrate.** Status: **in progress**; this is the
  current priority.
- No large training run begins before Milestone 0 is complete.
- Milestone 0 is complete when the repository provides: validated configuration
  loading; run identity generation; source provenance capture; deterministic seed
  management; logging and metric collection; artifact management; checkpoint
  serialization and restoration; experiment-manifest generation; hardware /
  environment capture; automated tests; a minimal training smoke test
  (umbrella #3; charter; AGENTS; founding spec §20).

## Milestone 0 issue hierarchy

Umbrella: **#3 — Milestone 0: Build the experimental substrate.** Dependency
order (see #3 for the full graph):

- **Foundation** — #4 scaffold · #5 configuration · #6 run identity
- **Reproducibility & observability** — #7 provenance/environment · #8 seed/RNG ·
  #9 logging/metrics
- **Persistence & evidence** — #10 artifacts · #11 checkpoints · #12 manifests
- **Verification & integration** — #13 test/CI harness · #14 smoke-and-recovery gate

#13 integrates tests throughout Milestone 0; #14 is strictly last.

**Progress:** **9 of 11 child issues closed** (#4–#12). **10 of 11 capabilities
delivered** (counting the active #13 CI foundation as delivered but not yet
closed). Remaining: #13 (active, will close after #14) and #14 (final gate).

## Active baseline

- No model baseline exists yet. The first model milestone is **D0 (minimal dense
  baseline)**; its entry criteria require Milestone 0 to be complete first
  ([doctrine/model-lineage.md](doctrine/model-lineage.md) §2). D0 planning is
  conditional on acceptance of #14 and closure of the remaining Milestone 0
  issues (#13 and #14). The D0 component choices are recorded in
  [doctrine/decisions/0002-initial-d0-architecture.md](doctrine/decisions/0002-initial-d0-architecture.md).

## Active issues

- **#14 — M0.11 end-to-end training smoke and recovery gate — sole remaining
  implementation gate.** All dependencies (#4–#12) are complete. This is the
  final Milestone 0 issue.
- **#13 — M0.10 test/CI harness — in progress.** Phase A merged via PR #21,
  squash commit `aaa31187d5d6b63671ee96df14a85a75b125f068`: permanent read-only CI,
  strict fast/integration/smoke/accelerator tiers, pinned toolchain, and
  repository-native policy checks are active. #13 remains open until the full
  Milestone 0 component and integration suite — including #14's smoke gate — is
  present and all child issues are closed.
- #3 umbrella remains open.
- **#4–#12 — completed.** All nine implementation and infrastructure issues from
  scaffold through experiment manifests are closed:
  - #4 scaffold · #5 configuration · #6 run identity
  - #7 provenance · #8 seed/RNG · #9 logging/metrics
  - #10 artifacts · #11 checkpoints (PR #35, `6801184`) · #12 manifests (PR #38, `f92f66c`)

## Open pull requests

- This state synchronization PR (documentation-only).

## Known blockers

- None. #14 is unblocked and is the sole remaining implementation gate.
- #13 will close after #14 is accepted and all child issues are resolved.
- D0 planning is conditional on Milestone 0 completion (#13 + #14 closed).

## Latest accepted experiment

- None yet. No experiments run; experiments will be recorded through the
  `experiment` issue template (which requires a maturity stage and a research
  family) and the experiment doctrine
  ([doctrine/evaluation-and-experiments.md](doctrine/evaluation-and-experiments.md)).

## Next recommended action

1. Begin **#14 (end-to-end training smoke and recovery gate)** — the final
   Milestone 0 issue.
2. After #14 is accepted, close **#13** (CI foundation) once the complete
   Milestone 0 suite satisfies its issue contract.
3. Close **#3** (umbrella) once all 11 child issues are closed.
4. After Milestone 0 is complete, begin D0 (minimal dense baseline, F0
   exact-attention control).

## Critical path

```text
#14 smoke-and-recovery gate → #13 closure → #3 closure → D0
```

## Relationship to ExpertOS

ExpertOS is the external runtime/control-plane counterpart. ExpertForge builds
its own models and does not copy ExpertOS internals; shared interfaces are
governed by versioned schemas and the ExpertOS resource contract defined here
([doctrine/deployment-and-runtime-codesign.md](doctrine/deployment-and-runtime-codesign.md)).
