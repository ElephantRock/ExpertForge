# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue and pull-request records. When this file conflicts with the
underlying issues/PRs or with `main`, those prevail.

Last updated: Milestone 0 implementation complete; #14 (final gate) delivered with #13 closing through the same PR. All 11 child issues close on merge; acceptance report at reports/milestone-0-smoke-gate.md.

## Current milestone

- **Milestone 0 — Experimental substrate.** Status: **implementation complete**
  (closes on merge of the #13/#14 PR). The end-to-end smoke & recovery gate
  (#14) proves the substrate composes with exact computational-state
  reproducibility across an interrupt/resume boundary; see
  [reports/milestone-0-smoke-gate.md](reports/milestone-0-smoke-gate.md).
- No large training run begins before Milestone 0 is complete.
- Milestone 0 is complete when the repository provides: validated configuration
  loading; run identity generation; source provenance capture; deterministic seed
  management; logging and metric collection; artifact management; checkpoint
  serialization and restoration; experiment-manifest generation; hardware /
  environment capture; automated tests; a minimal training smoke test
  (umbrella #3; charter; AGENTS; founding spec §20). All delivered.

## Milestone 0 issue hierarchy

Umbrella: **#3 — Milestone 0: Build the experimental substrate.** Dependency
order (see #3 for the full graph):

- **Foundation** — #4 scaffold · #5 configuration · #6 run identity
- **Reproducibility & observability** — #7 provenance/environment · #8 seed/RNG ·
  #9 logging/metrics
- **Persistence & evidence** — #10 artifacts · #11 checkpoints · #12 manifests
- **Verification & integration** — #13 test/CI harness · #14 smoke-and-recovery gate

#13 integrates tests throughout Milestone 0; #14 is strictly last. #13 and #14
close together through the #14 PR (one PR closes two child issues); attribution
for #13's remaining responsibility (the permanent #14 smoke integration) is
posted on umbrella #3.

**Progress:** **11 of 11 child issues delivered** (#4–#14); all close on merge of
the #13/#14 PR. The permanent CI runs the quality/fast, CPU integration, and
locked smoke tiers, including the one-command gate.

## Active baseline

- No model baseline exists yet. The first model milestone is **D0 (minimal dense
  baseline)**; its entry criteria require Milestone 0 to be complete first
  ([doctrine/model-lineage.md](doctrine/model-lineage.md) §2). With Milestone 0
  implementation complete and #13/#14 closing through their PR, D0 planning is
  next. The D0 component choices are recorded in
  [doctrine/decisions/0002-initial-d0-architecture.md](doctrine/decisions/0002-initial-d0-architecture.md).

## Active issues

- **#14 — M0.11 end-to-end training smoke and recovery gate — delivered.** The
  U0/R0/R1 topology runs the full substrate with exact U0@N vs R1@N
  computational-state reproducibility. Closes with #13 through the #14 PR.
  Acceptance report: [reports/milestone-0-smoke-gate.md](reports/milestone-0-smoke-gate.md).
- **#13 — M0.10 test/CI harness — delivered; closing with #14.** Phase A merged
  via PR #21, squash commit `aaa31187d5d6b63671ee96df14a85a75b125f068`: permanent
  read-only CI, strict fast/integration/smoke/accelerator tiers, pinned
  toolchain, repository-native policy checks. The #14 PR adds the permanent
  smoke CI job (locked smoke tier + one-command gate), completing #13's
  remaining responsibility.
- #3 umbrella remains open until all 11 children close, then closes separately.
- **#4–#12 — completed.** All nine implementation and infrastructure issues from
  scaffold through experiment manifests are closed:
  - #4 scaffold · #5 configuration · #6 run identity
  - #7 provenance · #8 seed/RNG · #9 logging/metrics
  - #10 artifacts · #11 checkpoints (PR #35, `6801184`) · #12 manifests (PR #38, `f92f66c`)

## Open pull requests

- The #14 smoke-gate PR (`Closes #13, Closes #14`).

## Known blockers

- No known implementation blocker remains. Merge still requires successful
  exact-head CI and final acceptance review.

## Latest accepted experiment

- **Milestone 0 smoke & recovery gate (Issue #14).** Maturity stage Milestone 0,
  classification smoke_test. The designated clean evidence source commit E is
  `42c7de1`; CI run `30753390845` passed the quality/fast tier (1248 tests), CPU
  integration tier (114 tests), locked smoke tier (39 tests), and one-command
  gate. U0@N and R1@N have identical computational digests (`925fbd96…`) and
  loss improvement 0.438 ≥ threshold 0.1.
- Earlier evidence source commits `58acd64`/`b2abe623…`, `f0730e1`, `fc043c1`,
  `298799a`, `537da2b`, `fd6d4d1`, `71666a2`, `27b8a00`, `2890d69`, `dfdad78`,
  `2aa359f`, and `991b50a` were invalidated by blocking reviews `4835261100`,
  `4835379984`, `4835574284`, `4835657284`, `4835782418`, `4835844853`,
  `4835902575`, `4837672450`, `4837842027`, `4838345569`, `4838509896`, and
  `4838854599`, respectively, and are superseded.
- Full evidence: [reports/milestone-0-smoke-gate.md](reports/milestone-0-smoke-gate.md).

## Next recommended action

1. Complete exact-head review and merge the **#14 smoke-gate PR** (closes #13 and #14).
2. Verify all 11 children of **#3** are closed, then close **#3** separately.
3. Begin **D0** (minimal dense baseline, F0 exact-attention control).

## Critical path

```text
exact-head acceptance → merge #14 (closes #13, #14) → close #3 → D0
```

## Relationship to ExpertOS

ExpertOS is the external runtime/control-plane counterpart. ExpertForge builds
its own models and does not copy ExpertOS internals; shared interfaces are
governed by versioned schemas and the ExpertOS resource contract defined here
([doctrine/deployment-and-runtime-codesign.md](doctrine/deployment-and-runtime-codesign.md)).
