# D0.0 Dense Baseline Contract — Final Review Report

**Issue:** #42  
**Parent:** #41  
**PR:** #43  
**Review status:** Complete prospective-contract review; D0.0 remains unratified pending `PROJECT_STATE.md` synchronization  
**Execution status:** No model implementation, corpus processing, training, generation, or experiment attempt authorized

## Executive determination

The D0.0 proposal is internally coherent, content-addressed, configuration-bound,
manifest-compatible, independently parameter-accounted, prompt-fixed, and guarded
by one permanent fail-closed CI command.

The review finds the prospective dense-baseline contract suitable for final
project-state synchronization. It does not accept a scientific result and does
not authorize D0.1 or D0.2. The selected FineWeb-Edu corpus has not been scanned
for prompt contamination, no production model has been instantiated, and no
qualification or canonical training attempt exists.

Exactly one ratification blocker remains:

```text
PROJECT_STATE_synchronization
```

## Review scope

This review covers the complete D0.0 prospective contract:

1. dense model architecture and exact-attention control;
2. tokenizer and dataset source identities;
3. sequence, batch, optimizer, initialization, schedule, precision, and seed semantics;
4. qualification and canonical resolved configurations and fingerprints;
5. formal experiment definitions and production-manifest compatibility;
6. independent parameter accounting;
7. fixed generation prompts and deterministic contamination protocol;
8. acceptance, failure, and kill criteria;
9. permanent validation command and CI topology;
10. evidence boundaries and remaining authorization conditions.

The review does not cover a D0.1 data pipeline implementation, a D0.2 model or
training-loop implementation, source-bound contamination results, accelerator
qualification, or any measured model outcome.

## Frozen scientific object

### Architecture

Both profiles use the same model family:

- decoder-only Transformer;
- exact causal multi-head self-attention;
- pre-norm RMSNorm;
- full-head RoPE with base 10,000 and no scaling;
- SwiGLU feed-forward blocks;
- tied token embedding and output projection;
- no biases;
- no dropout;
- next-token prediction over all target positions.

This is a prospective baseline definition rather than an implementation claim.

### Model profiles

| Profile | Layers | Width | Heads | Head dimension | SwiGLU width | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| Qualification | 8 | 256 | 4 | 64 | 768 | 19,685,888 |
| Canonical | 12 | 576 | 9 | 64 | 1,536 | 76,738,176 |

Both profiles declare zero non-trainable parameters.

### Sequence and update semantics

| Field | Frozen value |
|---|---:|
| Model context tokens | 1,024 |
| Packed source window tokens | 1,025 |
| Input tokens per sequence | 1,024 |
| Target tokens per sequence | 1,024 |
| Cursor advance | 1,024 |
| Lookback overlap | 1 token |
| Global sequences per update | 64 |
| Target tokens per update | 65,536 |

The one-token lookback makes every packed 1,025-token source window contribute
1,024 input tokens and 1,024 next-token targets while advancing the source cursor
by exactly 1,024 tokens.

### Training schedules

| Profile | Target tokens | Optimizer updates | Warmup updates | Peak LR | Final LR |
|---|---:|---:|---:|---:|---:|
| Qualification | 262,144,000 | 4,000 | 200 | 0.0006 | 0.00006 |
| Canonical | 2,097,152,000 | 32,000 | 1,600 | 0.0004 | 0.00004 |

Both schedules are exactly aligned to 65,536 target tokens per optimizer update.

## Immutable source identities

### Tokenizer

| Field | Frozen value |
|---|---|
| Repository | `EleutherAI/gpt-neox-20b` |
| Revision | `364ae95407723fadd1d47b023c1efb92a4d891c3` |
| Vocabulary | 50,257 |
| Source files | 5 |
| Aggregate manifest SHA-256 | `eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c` |

### Dataset

| Field | Frozen value |
|---|---|
| Repository | `HuggingFaceFW/fineweb-edu` |
| Revision | `84e8104e779e409e2267ac60609138e3dda2cbd2` |
| Configuration | `sample-10BT` |
| Source files | 14 |
| Aggregate bytes | 28,518,193,415 |
| Aggregate manifest SHA-256 | `d4e7108f2455a95c725fd61fcdb4423be24d1a9a5d3c6e2e322e9df6054bf0df` |

The source manifests bind exact paths, byte sizes, file digests, and consumed
semantics. They do not assert that local source bytes have already been processed.

## Configuration and specification identity

| Profile | Canonical configuration SHA-256 | Specification fingerprint |
|---|---|---|
| Qualification | `4e1bdd0ad30bf8b2e1b61f83ff9e387acdab1c0e0d1cb975e7627db95273a960` | `spec-v1-sha256-4f67b477c9d36c3aa06a4e99f0509380fdc91c672f6cc8aac3dd344880ce5cbe` |
| Canonical | `fbc699757b91ad0883fc4d30295a938df86d60f784a9fb91fb39385bbe3e6a4e` | `spec-v1-sha256-2167b1f07873c3aed6112c38c7de5cdcece6ac3b94d91acde2d605ec2decfb0b` |

The fingerprints bind the resolved configuration to the immutable dataset and
tokenizer manifest identities. Qualification and canonical identities are
distinct and profile crossover is rejected.

## Formal experiment contract

Each profile has a separate prospective formal experiment definition accepted by
the existing version-1 `ExperimentManifest` model.

| Profile | Control | Minimum useful effect |
|---|---|---:|
| Qualification | Same frozen model at update zero | 0.50 nat validation-loss improvement |
| Canonical | Same frozen model at update zero | 1.00 nat validation-loss improvement |

The independent variable is optimizer-update count and equivalent accumulated
target-token exposure. Dependent variables include validation loss, perplexity,
non-finite events, skipped updates, recovery equality, throughput, memory,
checkpoint latency, and generation completion.

Compatibility validation uses an interrupted, partial, unpublished synthetic
manifest. It creates no run directory, attempt artifact, result, or decision.

## Independent parameter accounting

The declarative inventory expands eleven trainable tensor families:

- one token embedding;
- two RMSNorm vectors per block;
- four attention matrices per block;
- three SwiGLU matrices per block;
- one final RMSNorm vector.

The output head is a shared-storage alias of the token embedding and contributes
zero additional elements. Biases, trainable RoPE state, dropout state, and a
separate output head are explicitly zero-parameter classes.

| Profile | Tensor families | Tensor instances | Inventory total | Closed-form total | Contract total |
|---|---:|---:|---:|---:|---:|
| Qualification | 11 | 74 | 19,685,888 | 19,685,888 | 19,685,888 |
| Canonical | 11 | 110 | 76,738,176 | 76,738,176 | 76,738,176 |

The inventory expansion and compact formula are independent executable paths.

## Generation and contamination contract

The prompt manifest freezes eight ordered prompts, exact probes, prompt and probe
hashes, decoding settings, and generation seeds.

| Identity | SHA-256 |
|---|---|
| Raw prompt manifest | `851934e4a49210b515967fad51308038dac1e7bc449774ca603f6bd9f47ccf3e` |
| Canonical prompt payload | `c52ef9f4420ff8160fd5f212370f46cefde71033ac38431bdad2878640070e51` |

The eventual D0.1 scan must cover every accepted normalized,
exact-text-deduplicated document before splitting. It checks, in precedence
order:

1. exact normalized full-prompt substring;
2. exact normalized probe substring;
3. any exact contiguous 64-code-point prompt window.

The maximum permitted hit count is zero. Any hit requires an amendment rather
than silent prompt replacement.

The corpus-wide scan has not run. This is a mandatory D0.1 preflight before
packing or training.

## Acceptance, failure, and kill boundaries

Qualification acceptance requires at least 0.50 nat fixed-validation loss
improvement, no skipped updates, bounded peak device memory, exact checkpoint
round-trip, and locked-environment resume equality.

Canonical acceptance requires at least 1.00 nat fixed-validation loss
improvement, bounded regression from the best prior validation result, bounded
consecutive regressing boundaries, throughput retention, no skipped updates, and
bounded peak device memory.

Failure and kill criteria are frozen for checkpoint latency, host memory,
insufficient loss improvement, non-finite values, skipped updates, checkpoint or
resume mismatch, out-of-memory recurrence, and failed recovery attempts.

These are prospective decision rules. No result has been evaluated against them.

## Permanent validation and CI

The authoritative command is:

```bash
uv run --locked python -m scripts.validate_d0_ratification
```

The aggregate gate executes the authoritative validators in fixed order and
fails closed on omission, reordering, exceptions, invalid report types,
nondeterministic report identity, content-addressed review-report drift, or stale
blocker state.

CI places the gate after formatting, Ruff, and strict mypy and before the
complete fast CPU suite. CPU integration and the locked smoke/recovery job depend
on the quality job.

The exact review head must pass:

- formatting and Ruff;
- strict mypy;
- aggregate D0 ratification validation;
- complete fast CPU tests;
- package, configuration, lockfile, and repository-policy checks;
- portable CPU integration;
- locked smoke tests;
- one-command interruption and recovery.

## Evidence-to-claim matrix

| Claim | Evidence | Review determination |
|---|---|---|
| Source identities are immutable | tokenizer and dataset manifests | Supported |
| Configurations match the contract | resolved YAML validator and fingerprints | Supported |
| Formal definitions fit the production manifest schema | in-memory manifest compatibility validator | Supported |
| Parameter totals are independently derived | declarative inventory and closed-form counter | Supported |
| Prompt identity and scan mechanics are fixed | prompt manifest and contamination fixtures | Supported |
| All D0.0 validators compose under CI | permanent aggregate command and CI dependency chain | Supported |
| Selected corpus is contamination-free | no source-bound scan report exists | Not claimed |
| Production model matches the contract | no production model exists | Not claimed |
| Qualification or canonical outcome meets thresholds | no training attempt exists | Not claimed |
| D0.1 or D0.2 is authorized | final project-state synchronization is incomplete | Not authorized |

## Review findings

### Satisfied

1. The contract contains no approximate model, token-budget, batch, or schedule values.
2. Source revisions and aggregate manifests are exact and content-addressed.
3. Qualification and canonical configurations resolve through the existing strict schema.
4. Specification fingerprints are distinct and bind immutable source identities.
5. Formal experiment definitions are explicit and production-manifest compatible.
6. Parameter accounting has an independent executable comparison.
7. Generation prompts and contamination mechanics are fixed and mutation-tested.
8. Acceptance, failure, and kill semantics are machine-readable.
9. One permanent fail-closed command composes the full D0.0 evidence bundle.
10. CI enforces the command before downstream integration and smoke/recovery checks.

### Deliberately unresolved outside D0.0

1. The selected corpus must be scanned before packing or training.
2. The production model and data pipeline must be implemented and independently accepted.
3. Accelerator precision and resource preflight must be demonstrated.
4. Qualification must precede canonical execution.
5. Any accepted scientific claim requires complete attempt evidence and a terminal manifest.

## Alignment with the ExpertForge operating model

The proposal preserves the project’s separation between experimental substrate
and model hypothesis. D0.0 defines the scientific object and its prospective
decision rules while relying on the existing configuration, identity,
provenance, checkpoint, telemetry, manifest, artifact, recovery, and CI
contracts.

This review therefore treats a future D0 result as acceptable only when it is
traceable to immutable specification identity, source state, environment,
lineage, telemetry, checkpoints, generated outputs, and terminal manifest
evidence. Training completion alone will not constitute acceptance.

## Ratification state

Closed by this review artifact:

```text
rendered_review_report
```

Remaining:

```text
PROJECT_STATE_synchronization
```

PR #43 must remain draft until the final synchronization tranche updates
`PROJECT_STATE.md`, the machine-readable blocker state, the permanent gate, and
the exact-head CI evidence together.

## Final review disposition

**Disposition: accept the D0.0 prospective contract for project-state
synchronization.**

This disposition means the contract is ready for the final administrative and
repository-state binding step. It does not ratify a model result, authorize
material execution, or waive the D0.1 contamination preflight and D0.2
implementation acceptance requirements.
