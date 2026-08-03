# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue, pull-request, decision, report, manifest, or accepted-run
records. When this file conflicts with the underlying records or with `main`,
those prevail.

Last updated: D0.0 dense-baseline contract preparation is complete on PR #43.
The prospective contract is ratification-ready with zero preparation blockers;
Issue #42 remains open and no D0.1/D0.2 implementation or material execution is
authorized until explicit acceptance, merge, and issue closure.

## Current milestone

- **D0 — Controlled dense baseline program (#41).** Status: **D0.0
  ratification-ready; not yet ratified**.
- Milestone 0 is complete and supplies the validated configuration, identity,
  provenance, seed/RNG, telemetry, artifact, checkpoint, manifest, testing, CI,
  smoke, and interruption/recovery substrate used by D0.
- D0.0 freezes the prospective scientific and execution contract. It does not
  establish a model result or authorize a qualification or canonical run.

## D0 work-package state

| Work package | State | Authorization boundary |
|---|---|---|
| D0.0 baseline contract (#42) | Ratification-ready on PR #43 | Await explicit acceptance, merge, and closure |
| D0.1 immutable data pipeline | Blocked | Authorized only after #42 closes |
| D0.2 dense model implementation | Blocked | Authorized only after #42 closes |
| D0.3 production training path | Blocked | Requires accepted D0.1 and D0.2 |
| D0.4 evaluation and profiling | Blocked | Requires production model/training integration |
| D0.5 qualification run | Blocked | First material training authorization; requires D0.1–D0.4 |
| D0.6 canonical run | Blocked | Requires accepted D0.5 |
| D0.7 baseline freeze | Blocked | Requires accepted canonical evidence |

D0.1 and D0.2 may proceed in parallel only after Issue #42 closes. No corpus
processing, source-bound contamination scan, model implementation, training,
generation, or experiment attempt is represented as complete by D0.0.

## Frozen D0.0 prospective contract

### Model profiles

| Profile | Layers | Width | Heads | Head dimension | SwiGLU width | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| Qualification | 8 | 256 | 4 | 64 | 768 | 19,685,888 |
| Canonical | 12 | 576 | 9 | 64 | 1,536 | 76,738,176 |

Both profiles are tied-embedding, bias-free, dropout-free decoder-only
Transformers with exact causal multi-head self-attention, pre-norm RMSNorm,
full-head RoPE, SwiGLU, and next-token prediction.

### Sequence, batch, and budgets

- Model context: 1,024 tokens.
- Packed source window: 1,025 tokens with one-token lookback.
- Global sequences per optimizer update: 64.
- Target tokens per optimizer update: 65,536.
- Qualification: 262,144,000 target tokens and 4,000 optimizer updates.
- Canonical: 2,097,152,000 target tokens and 32,000 optimizer updates.
- Canonical precision path: native BF16 or fail closed.
- Qualification FP32 fallback: distinct, non-equivalent attempt.
- FP16: unauthorized.

### Immutable identities

| Artifact | Identity |
|---|---|
| Tokenizer source manifest | `eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c` |
| Dataset source manifest | `d4e7108f2455a95c725fd61fcdb4423be24d1a9a5d3c6e2e322e9df6054bf0df` |
| Qualification specification | `spec-v1-sha256-4f67b477c9d36c3aa06a4e99f0509380fdc91c672f6cc8aac3dd344880ce5cbe` |
| Canonical specification | `spec-v1-sha256-2167b1f07873c3aed6112c38c7de5cdcece6ac3b94d91acde2d605ec2decfb0b` |
| Raw generation-prompt manifest | `851934e4a49210b515967fad51308038dac1e7bc449774ca603f6bd9f47ccf3e` |
| Canonical generation-prompt payload | `c52ef9f4420ff8160fd5f212370f46cefde71033ac38431bdad2878640070e51` |
| Final review report | `c41573a01f7f0568221b7e12edf87a50dd2ab54f39e9d53de37af68a088060b9` |

Normative and review artifacts:

- [prospective machine contract](experiments/d0/baseline-contract-v1.proposed.json)
- [formal experiment definition](experiments/d0/formal-experiment-definition-v1.json)
- [parameter inventory](experiments/d0/parameter-inventory-v1.json)
- [generation prompt set](experiments/d0/generation-prompts-v1.json)
- [final review identity](experiments/d0/final-review-report-v1.json)
- [final rendered review](reports/d0-baseline-contract-final-review.md)

## Permanent conformance gate

The authoritative offline, non-mutating command is:

```bash
uv run --locked python -m scripts.validate_d0_ratification
```

It validates the baseline contract, source manifests, configuration bindings,
formal experiment definition, parameter inventory, generation/contamination
contract, content-addressed final review, and synchronized project state in a
fixed order. CI runs the command as a dedicated quality step before the complete
fast suite; CPU integration and smoke/recovery depend on that quality job.

The prospective contract now records zero preparation blockers. This means the
repository evidence bundle is complete for final acceptance review. It does not
mean Issue #42 is closed or that the contract has been ratified on `main`.

## Claim and execution boundary

Current truthful statements:

- D0.0 contract preparation is complete and ratification-ready.
- The selected dataset and tokenizer sources are immutable and digest-bound.
- Exact resolved configurations and specification fingerprints exist.
- The parameter counts are independently derivable and validated.
- The prompt set and contamination protocol are frozen.
- The corpus-wide contamination scan has not run.
- No production dense model has been instantiated.
- No qualification or canonical attempt exists.
- No material D0 execution is authorized by this branch state.

After Issue #42 closes, D0.1 and D0.2 become authorized implementation work.
D0.5 remains the first authorization for material engineering training, and
D0.6 remains blocked until qualification is accepted.

## Active issues and pull requests

- **#41 — D0 controlled dense baseline program:** open umbrella.
- **#42 — D0.0 dense-baseline execution contract:** open; ratification-ready.
- **PR #43 — D0.0 contract and evidence:** prepared for final acceptance review;
  not merged.

## Known blockers

- No D0.0 contract-preparation blocker remains.
- Ratification still requires explicit acceptance, merge of PR #43, and closure
  of Issue #42.
- D0.1 and D0.2 remain authorization-blocked until that closure.

## Latest accepted experiment

- **Milestone 0 smoke and recovery gate.** This remains the latest accepted
  executed experiment. It demonstrates exact computational-state
  reproducibility across interruption/resume for the experimental substrate.
- D0.0 is a prospective contract record, not an executed experiment and not a
  model baseline result.
- Evidence: [Milestone 0 smoke-gate report](reports/milestone-0-smoke-gate.md).

## Next recommended action

1. Perform final exact-head review of PR #43.
2. Accept and merge PR #43.
3. Close Issue #42, thereby ratifying D0.0 and authorizing D0.1/D0.2.
4. Open or activate separate D0.1 and D0.2 implementation branches/issues.
5. Execute the source-bound contamination scan in D0.1 before packing or
   training.

## Critical path

```text
exact-head acceptance → merge PR #43 → close #42
    ├─→ D0.1 immutable data pipeline
    └─→ D0.2 dense model implementation
             ↓
       D0.3 production training path
             ↓
       D0.4 evaluation/profiling
             ↓
       D0.5 qualification
             ↓
       D0.6 canonical run
             ↓
       D0.7 baseline freeze
```

## Relationship to ExpertOS

ExpertOS is the external runtime/control-plane counterpart. ExpertForge builds
its own models and does not copy ExpertOS internals; shared interfaces are
governed by versioned schemas and the ExpertOS resource contract defined in
[deployment and runtime co-design](doctrine/deployment-and-runtime-codesign.md).
