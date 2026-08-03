# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue, pull-request, decision, report, manifest, or accepted-run
records. When it conflicts with those records or with `main`, those prevail.

Last updated: D0.0 acceptance-review corrections are complete on PR #43. The
prospective contract is ratification-ready with zero preparation blockers;
Issue #42 remains open. No D0.1/D0.2 implementation or material execution is
authorized until explicit acceptance, merge, and issue closure.

## Current milestone

- **D0 — Controlled dense baseline program (#41).** Status: **D0.0
  ratification-ready; not yet ratified**.
- Milestone 0 is complete and supplies the configuration, identity, provenance,
  seed/RNG, telemetry, artifact, checkpoint, manifest, CI, smoke, and recovery
  substrate used by D0.
- D0.0 freezes a prospective scientific and execution contract. It does not
  establish a model result or authorize a qualification or canonical run.

## D0 work-package state

| Work package | State | Authorization boundary |
|---|---|---|
| D0.0 baseline contract (#42) | Corrected and ratification-ready on PR #43 | Await acceptance, merge, and closure |
| D0.1 immutable data pipeline | Blocked | Authorized only after #42 closes |
| D0.2 dense model implementation | Blocked | Authorized only after #42 closes |
| D0.3 production training path | Blocked | Requires accepted D0.1 and D0.2 |
| D0.4 evaluation and profiling | Blocked | Requires model/training integration |
| D0.5 qualification run | Blocked | First material training authorization |
| D0.6 canonical run | Blocked | Requires accepted D0.5 |
| D0.7 baseline freeze | Blocked | Requires accepted canonical evidence |

D0.1 and D0.2 may proceed in parallel only after Issue #42 closes. No corpus
processing, contamination scan, model implementation, training, generation, or
experiment attempt is represented as complete by D0.0.

## Frozen D0.0 prospective contract

### Model and token accounting

| Profile | Layers | Width | Heads | SwiGLU width | Parameters | Target tokens | Updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qualification | 8 | 256 | 4 | 768 | 19,685,888 | 262,144,000 | 4,000 |
| Canonical | 12 | 576 | 9 | 1,536 | 76,738,176 | 2,097,152,000 | 32,000 |

Both profiles use tied embeddings, exact causal MHA, pre-norm RMSNorm,
full-head RoPE, SwiGLU, no biases, no dropout, and 1,024-token context.
Each optimizer update contains 64 sequences and 65,536 target tokens.

### Corrected execution semantics

Applied optimizer updates use one-based index `u`; update zero is the untrained
control with learning rate `0.0`. Warmup reaches peak LR exactly at the declared
warmup update, and cosine decay reaches final LR exactly at the last update.

Data order uses `expertforge.seed-derivation` version 1 from root seed
`2026080200`, context `component=data.order` with worker/rank/device/stream zero,
and derived value:

```text
data_seed_u64 = 4657843784274978659
```

The dataset manifest validates artifact provenance, lexicographic source-file
order, physical row order, and first-in-source-order duplicate retention.
Nested configuration overrides can traverse the optional D0 section while
section-level and unknown-path overrides remain rejected.

### Immutable identities

| Artifact | Identity |
|---|---|
| Tokenizer source manifest | `eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c` |
| Dataset source manifest | `85fab524b13c49da78264eb03124ce113395657d6403627e53008aade7eea7d7` |
| Qualification configuration | `c02acbf24ce620d2281030bff7cf5719caa56b07b1d3632d11d0c4559ea6f76d` |
| Qualification specification | `spec-v1-sha256-02093a3c05c8b5079e2c4fbb8bafaada6365c0776a3a94ca3a8143873167f6d3` |
| Canonical configuration | `1996ccd76638ae032bb5f81fb33f6a7972acd55c9dd0c4862b3729fc84155cce` |
| Canonical specification | `spec-v1-sha256-2faaa0f0beed2d1c674c7d29eb969944f9e208825a527e9f7a0024ba538815ea` |
| Raw generation-prompt manifest | `c748706d9706e42bc0147d62c80f8f486ebb6aaa72fb53140b491ce7706e18c1` |
| Canonical generation-prompt payload | `c52ef9f4420ff8160fd5f212370f46cefde71033ac38431bdad2878640070e51` |
| Corrected final review report | `a2559202fe8f76572c4f5a719285e840fdd95f23345c6da8f73f4ad9942504e1` |

Normative and review artifacts include the machine contract, resolved profile
configurations, formal experiment definition, parameter inventory, prompt set,
final review manifest, and corrected rendered review.

## Permanent conformance gate

The authoritative offline, non-mutating command is:

```bash
uv run --locked python -m scripts.validate_d0_ratification
```

It validates the contract, source manifests, configuration bindings, formal
experiment definition, parameter inventory, generation/contamination contract,
content-addressed final review, and synchronized project state in fixed order.

The contract records zero preparation blockers. This means the repository
evidence bundle is complete for exact-head acceptance; it does not mean Issue
#42 is closed or the contract is ratified on `main`.

## Claim and execution boundary

Current truthful statements:

- D0.0 preparation and acceptance-review corrections are complete.
- Dataset and tokenizer sources are immutable and digest-bound.
- Exact configurations, fingerprints, schedule formulas, and data-order seed exist.
- Parameter totals are independently derivable.
- Prompt identity and contamination mechanics are frozen.
- The corpus-wide contamination scan has not run.
- No production dense model has been instantiated.
- No qualification or canonical attempt exists.
- No material D0 execution is authorized by this branch state.

After Issue #42 closes, D0.1 and D0.2 become authorized implementation work.
D0.5 remains the first material engineering-training authorization.

## Active issues and pull requests

- **#41 — D0 controlled dense baseline program:** open umbrella.
- **#42 — D0.0 dense-baseline execution contract:** open; corrected and
  ratification-ready.
- **PR #43 — D0.0 contract and evidence:** open, unmerged, and undergoing final
  exact-head acceptance after review corrections.

## Known blockers

- No D0.0 preparation blocker remains.
- Ratification requires clean exact-head CI, resolution of all blocking review
  threads, explicit acceptance, merge of PR #43, and closure of Issue #42.
- D0.1 and D0.2 remain authorization-blocked until that closure.

## Latest accepted experiment

- **Milestone 0 smoke and recovery gate** remains the latest accepted executed
  experiment and demonstrates exact computational-state recovery for the
  experimental substrate.
- D0.0 is a prospective contract, not an executed experiment or model result.
- Evidence: [Milestone 0 smoke-gate report](reports/milestone-0-smoke-gate.md).

## Next recommended action

1. Complete exact-head CI for the acceptance-review corrections.
2. Resolve the five blocking review threads with evidence.
3. Accept and merge PR #43.
4. Close Issue #42 and update umbrella #41.
5. Activate separate D0.1 and D0.2 implementation work.
6. Execute the source-bound contamination scan before packing or training.

## Critical path

```text
corrected exact-head CI → resolve reviews → merge PR #43 → close #42
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
governed by versioned schemas and the ExpertOS resource contract in
[deployment and runtime co-design](doctrine/deployment-and-runtime-codesign.md).
