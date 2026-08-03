# PROJECT_STATE.md

**Index of current project state.** This file is an index, not a substitute for
detailed issue, pull-request, decision, report, manifest, or accepted-run
records. When it conflicts with those records or with `main`, those prevail.

Last updated: D0.0 was ratified through PR #43 and Issue #42 closure. D0.1
immutable-data-pipeline work and D0.2 dense-model implementation are authorized
to proceed in parallel. D0.3, corpus packing before contamination clearance,
qualification training, canonical training, and all material execution remain
blocked.

## Current milestone

- **D0 — Controlled dense baseline program (#41).** Status: **D0.0 ratified**.
- Accepted source head: `c1495d414d19f46d71241b9110bf9ce5e08efe17`.
- Squash merge: `9983ec62748b30e982e5879df1426fa80c900a7f`.
- Acceptance CI: run 349, completed successfully.
- D0.0 establishes an exact prospective contract. It does not establish a
  model result, corpus suitability, or training success.

## D0 work-package state

| Work package | State | Authorization boundary |
|---|---|---|
| D0.0 baseline contract (#42) | Ratified and closed | Amendments require dedicated review and new fingerprints |
| D0.1 immutable data pipeline | Authorized | No packing or training before source-bound contamination clearance |
| D0.2 dense model implementation | Authorized | Implementation and tests only; no material training |
| D0.3 production training path | Blocked | Requires accepted D0.1 and D0.2 |
| D0.4 evaluation and profiling | Blocked | Requires production model/training integration |
| D0.5 qualification run | Blocked | First material engineering-training authorization |
| D0.6 canonical run | Blocked | Requires accepted D0.5 |
| D0.7 baseline freeze | Blocked | Requires accepted canonical evidence |

D0.1 and D0.2 may proceed in parallel. Authorization is limited to their
declared implementation scopes. It does not authorize a qualification or
canonical attempt.

## Frozen D0.0 prospective contract

### Model and token accounting

| Profile | Layers | Width | Heads | SwiGLU width | Parameters | Target tokens | Updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qualification | 8 | 256 | 4 | 768 | 19,685,888 | 262,144,000 | 4,000 |
| Canonical | 12 | 576 | 9 | 1,536 | 76,738,176 | 2,097,152,000 | 32,000 |

Both profiles use tied embeddings, exact causal MHA, pre-norm RMSNorm,
full-head RoPE, SwiGLU, no biases, no dropout, and 1,024-token context.
Each optimizer update contains 64 sequences and 65,536 target tokens.

### Execution semantics

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

### Immutable identities

| Artifact | Identity |
|---|---|
| Tokenizer source manifest | `eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c` |
| Dataset source manifest | `85fab524b13c49da78264eb03124ce113395657d6403627e53008aade7eea7d7` |
| Qualification specification | `spec-v1-sha256-02093a3c05c8b5079e2c4fbb8bafaada6365c0776a3a94ca3a8143873167f6d3` |
| Canonical specification | `spec-v1-sha256-2faaa0f0beed2d1c674c7d29eb969944f9e208825a527e9f7a0024ba538815ea` |
| Raw generation-prompt manifest | `c748706d9706e42bc0147d62c80f8f486ebb6aaa72fb53140b491ce7706e18c1` |
| Canonical generation-prompt payload | `c52ef9f4420ff8160fd5f212370f46cefde71033ac38431bdad2878640070e51` |
| Corrected final review report | `a2559202fe8f76572c4f5a719285e840fdd95f23345c6da8f73f4ad9942504e1` |
| Ratification record | `d3cd5c43091d1b68bfb63ec8be896cfa663f735e74f273adbfc9765e4bab8124` |

## Permanent conformance gate

The authoritative offline, non-mutating command is:

```bash
uv run --locked python -m scripts.validate_d0_ratification
```

It validates the historical proposal, immutable source and configuration
bindings, formal experiment definition, parameter inventory,
generation/contamination contract, final review, ratification record, and
current project state in a fixed fail-closed order.

## Claim and execution boundary

Current truthful statements:

- D0.0 is ratified.
- D0.1 and D0.2 implementation work is authorized.
- The corpus-wide contamination scan has not run.
- No production dense model has been instantiated.
- No packed D0 training corpus exists.
- No qualification or canonical attempt exists.
- No material D0 execution is authorized.
- D0.3 remains blocked until D0.1 and D0.2 are accepted.
- D0.5 remains the first material engineering-training authorization.

## Active issues and pull requests

- **#41 — D0 controlled dense baseline program:** open umbrella.
- **#42 — D0.0 dense-baseline execution contract:** closed and ratified.
- **PR #43 — D0.0 contract and evidence:** merged.
- **#44 — post-ratification state transition:** open until the current-state
  record and validator update merge.

## Known blockers

- D0.1 must materialize and validate the immutable source inventory.
- The source-bound contamination scan must produce a content-addressed zero-hit
  report before packing or training.
- D0.2 must instantiate the frozen architecture and prove exact parameter,
  tensor-shape, masking, gradient, and state-dict behavior.
- D0.3 remains blocked on accepted D0.1 and D0.2 evidence.

## Latest accepted experiment

- **Milestone 0 smoke and recovery gate** remains the latest accepted executed
  experiment and demonstrates exact computational-state recovery for the
  experimental substrate.
- D0.0 is a ratified prospective contract, not an executed experiment or model
  result.
- Evidence: [Milestone 0 smoke-gate report](reports/milestone-0-smoke-gate.md).

## Next recommended action

1. Implement D0.1 and D0.2 as separate reviewable work streams.
2. In D0.1, reproduce the frozen source inventory and execute the source-bound
   zero-hit contamination scan before packing.
3. In D0.2, implement the exact architecture and verify instantiated tensor
   counts against the independent inventory.
4. Do not begin D0.3 or any material training until both packages are accepted.

## Critical path

```text
D0.0 ratified
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
