# D0.0 Dense Baseline Contract — Final Review Report

**Issue:** #42  
**Parent:** #41  
**PR:** #43  
**Review status:** Corrected prospective-contract review complete after acceptance-review findings  
**Execution status:** No model implementation, corpus processing, training, generation, or experiment attempt authorized

## Executive determination

The corrected D0.0 proposal is internally coherent, content-addressed,
configuration-bound, manifest-compatible, independently parameter-accounted,
prompt-fixed, and protected by one permanent fail-closed validation command.

The acceptance review identified five blocking defects: underspecified learning-rate
indexing, an undefined data-order seed, incomplete dataset-manifest provenance,
unvalidated ordering and deduplication semantics, and configuration overrides that
could not descend through an optional D0 section. All five are now resolved and
mutation-tested.

The review disposition remains to accept the D0.0 prospective contract for the
project-state synchronization transition represented by
`PROJECT_STATE_synchronization`. The synchronized terminal state is validated
separately. This review does not accept a scientific result and does not authorize
D0.1 or D0.2.

## Review scope

This review covers the complete prospective execution contract:

1. exact dense architecture and parameter totals;
2. tokenizer and dataset source identities;
3. normalization, source ordering, duplicate selection, splitting, and training order;
4. sequence, batch, optimizer, initialization, schedule, precision, and seed semantics;
5. qualification and canonical resolved configurations and fingerprints;
6. formal experiment definitions and manifest compatibility;
7. generation prompts and contamination protocol;
8. acceptance, failure, and kill criteria;
9. permanent validation and CI;
10. claim and authorization boundaries.

It excludes implementation of the D0.1 data pipeline, the D0.2 model, the
production training path, corpus-wide contamination results, and measured model
outcomes.

## Frozen scientific object

### Architecture and profiles

Both profiles are tied-embedding, bias-free, dropout-free decoder-only
Transformers with exact causal multi-head self-attention, pre-norm RMSNorm,
full-head RoPE using base 10,000 without scaling, SwiGLU, and next-token
prediction.

| Profile | Layers | Width | Heads | Head dimension | SwiGLU width | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| Qualification | 8 | 256 | 4 | 64 | 768 | 19,685,888 |
| Canonical | 12 | 576 | 9 | 64 | 1,536 | 76,738,176 |

Both profiles declare zero non-trainable parameters.

### Sequence, batch, and budgets

| Field | Frozen value |
|---|---:|
| Model context tokens | 1,024 |
| Packed source window tokens | 1,025 |
| Cursor advance tokens | 1,024 |
| Global sequences per optimizer update | 64 |
| Target tokens per optimizer update | 65,536 |
| Qualification target tokens / updates | 262,144,000 / 4,000 |
| Canonical target tokens / updates | 2,097,152,000 / 32,000 |

Each packed source window contributes 1,024 inputs and 1,024 next-token targets
with a one-token lookback and no padding.

### Exact learning-rate indexing

Applied optimizer updates use one-based index `u` over
`1 <= u <= optimizer_updates`. Update zero is the untrained control and has
learning rate `0.0`.

Warmup is exact:

```text
lr(u) = peak_learning_rate * u / warmup_updates
for 1 <= u <= warmup_updates
```

Cosine decay is exact:

```text
lr(u) = final_learning_rate
      + 0.5 * (peak_learning_rate - final_learning_rate)
      * (1 + cos(pi * (u - warmup_updates)
                       / (optimizer_updates - warmup_updates)))
for warmup_updates < u <= optimizer_updates
```

Qualification uses 200 warmup updates, peak LR `0.0006`, and final LR
`0.00006`. Canonical uses 1,600 warmup updates, peak LR `0.0004`, and final LR
`0.00004`.

### Exact data-order seed

The data-order seed is derived using the existing
`expertforge.seed-derivation` version 1 contract from root seed `2026080200`
and context `component=data.order`, with worker, rank, device, and stream all
zero. The first eight SHA-256 bytes interpreted as an unsigned big-endian
integer produce:

```text
data_seed_u64 = 4657843784274978659
```

The training-order hash is explicitly bound to
`seeds.data_order_seed.data_seed_u64`.

## Immutable source identities

### Tokenizer

| Field | Frozen value |
|---|---|
| Repository | `EleutherAI/gpt-neox-20b` |
| Revision | `364ae95407723fadd1d47b023c1efb92a4d891c3` |
| Vocabulary | 50,257 |
| Source files | 5 |
| Manifest SHA-256 | `eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c` |

### Dataset

| Field | Frozen value |
|---|---|
| Repository | `HuggingFaceFW/fineweb-edu` |
| Revision | `84e8104e779e409e2267ac60609138e3dda2cbd2` |
| Configuration | `sample-10BT` |
| Source files | 14 |
| Aggregate bytes | 28,518,193,415 |
| Inventory content SHA-256 | `6851532c2387688d6d5bcff2f3da0983b962f42630f6b6f67bd1f966ef664db1` |
| Manifest SHA-256 | `85fab524b13c49da78264eb03124ce113395657d6403627e53008aade7eea7d7` |

The dataset manifest now includes artifact kind, format version, source
identity, parent experiment, generation method and command, inventory content
hash, permanent retention status, durable location, and reproduction path.

Source files are ordered lexicographically by path and rows by physical row
index. Exact-text duplicates use
`sha256(normalized_utf8_text)` and retain the first document in source order.
The validator rejects drift in provenance, ordering, or duplicate selection.

## Configuration and specification identity

| Profile | Canonical configuration SHA-256 | Specification fingerprint |
|---|---|---|
| Qualification | `c02acbf24ce620d2281030bff7cf5719caa56b07b1d3632d11d0c4559ea6f76d` | `spec-v1-sha256-02093a3c05c8b5079e2c4fbb8bafaada6365c0776a3a94ca3a8143873167f6d3` |
| Canonical | `1996ccd76638ae032bb5f81fb33f6a7972acd55c9dd0c4862b3729fc84155cce` | `spec-v1-sha256-2faaa0f0beed2d1c674c7d29eb969944f9e208825a527e9f7a0024ba538815ea` |

The fingerprints bind the corrected resolved configurations to the immutable
dataset and tokenizer identities. The configuration override resolver now
traverses direct and optional nested Pydantic sections while continuing to
reject section-level and unknown-path overrides.

## Formal experiment contract

Each profile has a separate prospective formal experiment definition accepted
by the version-1 production `ExperimentManifest` model. The qualification
minimum useful effect is a 0.50 nat fixed-validation loss improvement; the
canonical minimum useful effect is 1.00 nat.

The fixed constraints now reference the corrected dataset manifest and
specification fingerprints. Compatibility validation uses unpublished,
synthetic interrupted identities and creates no run directory, attempt
artifact, or result.

## Independent parameter accounting

The declarative inventory expands eleven trainable tensor families and agrees
with an independent closed-form counter:

| Profile | Tensor instances | Inventory total | Closed-form total | Contract total |
|---|---:|---:|---:|---:|
| Qualification | 74 | 19,685,888 | 19,685,888 | 19,685,888 |
| Canonical | 110 | 76,738,176 | 76,738,176 | 76,738,176 |

The tied output projection is a shared-storage alias and contributes no
additional parameters. Biases, trainable RoPE state, dropout state, and a
separate output head contribute zero.

## Generation and contamination contract

The prompt set freezes eight ordered prompts, probes, prompt/probe hashes,
decoding settings, and generation seeds.

| Identity | SHA-256 |
|---|---|
| Raw prompt manifest | `c748706d9706e42bc0147d62c80f8f486ebb6aaa72fb53140b491ce7706e18c1` |
| Canonical prompt payload | `c52ef9f4420ff8160fd5f212370f46cefde71033ac38431bdad2878640070e51` |

The eventual scan covers every accepted normalized and deduplicated document
before splitting. It checks exact normalized full prompts, exact probes, and
all exact contiguous 64-code-point prompt windows. The allowed hit count is
zero.

The corpus-wide scan has not run. It remains a mandatory D0.1 preflight before
packing or training.

## Acceptance, failure, and kill boundaries

Qualification acceptance requires at least 0.50 nat fixed-validation loss
improvement, zero skipped updates, bounded device memory, exact checkpoint
round-trip, and locked-environment resume equality.

Canonical acceptance requires at least 1.00 nat improvement, bounded loss
regression, bounded consecutive regressing validation boundaries, throughput
retention, zero skipped updates, and bounded device memory.

Failure and kill rules cover checkpoint latency, host memory, insufficient
improvement, non-finite values, skipped updates, checkpoint/resume mismatch,
out-of-memory recurrence, and failed recovery. No result has been evaluated
against these prospective rules.

## Permanent validation and CI

The authoritative command is:

```bash
uv run --locked python -m scripts.validate_d0_ratification
```

The fixed validator chain checks the baseline contract, source manifests,
configuration bindings, formal experiment definition, parameter inventory,
generation/contamination contract, final review, and project state. It fails
closed on registry drift, identity crossover, report mutation, stale blocker
state, or unauthorized execution claims.

Exact-head acceptance additionally requires formatting, Ruff, strict mypy, the
complete fast suite, package/configuration/lockfile/repository-policy checks,
portable CPU integration, locked smoke tests, and the one-command
interruption/recovery gate.

## Evidence-to-claim matrix

| Claim | Evidence | Determination |
|---|---|---|
| LR schedule is exactly reproducible | one-based indexing, formulas, boundary tests | Supported |
| Data order has an exact seed | versioned derivation record and computed `data_seed_u64` | Supported |
| Dataset source artifact is provenance-bound | content-addressed manifest provenance | Supported |
| Source order and duplicate retention are exact | manifest fields, validator, mutation tests | Supported |
| Optional D0 config leaves are overrideable | resolver schema traversal and regression tests | Supported |
| Corrected configs match the contract | resolved YAML and refreshed fingerprints | Supported |
| Formal definitions fit the production manifest | in-memory compatibility validator | Supported |
| Selected corpus is contamination-free | no source-bound scan exists | Not claimed |
| Production model matches the contract | no production model exists | Not claimed |
| Qualification or canonical thresholds are met | no training attempt exists | Not claimed |

## Review findings

### Satisfied

1. Learning-rate update indexing and every warmup/decay boundary are exact.
2. The data-order seed is a computed, versioned 64-bit value.
3. Dataset provenance is explicit and content-addressed.
4. Source ordering and duplicate retention are validated rather than descriptive.
5. Optional-section configuration override traversal is functional and tested.
6. Source, configuration, fingerprint, formal-definition, and prompt identities are mutually consistent.
7. Parameter accounting remains independently closed.
8. Acceptance, failure, and kill semantics remain mechanically decidable.
9. Permanent validation fails closed on all corrected contract boundaries.

### Deliberately unresolved outside D0.0

1. The selected corpus must be scanned before packing or training.
2. D0.1 and D0.2 implementations must be independently reviewed.
3. Accelerator precision and resource preflight must be demonstrated.
4. Qualification must precede canonical execution.
5. Scientific claims require complete attempt evidence and a terminal manifest.

## Alignment with the ExpertForge operating model

The corrected proposal preserves the separation between experimental substrate,
prospective scientific contract, implementation, and accepted evidence. A
future D0 result is acceptable only when bound to immutable specification,
source state, environment, lineage, telemetry, checkpoints, generated outputs,
and a terminal manifest. Training completion alone is insufficient.

The acceptance-review corrections strengthen exact reproducibility, artifact
lifecycle provenance, fail-closed configuration behavior, and the distinction
between a reviewed specification and an executed experiment.

## Ratification state

This reissued review preserves the historical review-stage transition:

```text
PROJECT_STATE_synchronization
```

That transition is now represented by a separate content-addressed project-state
artifact. The machine contract has zero preparation blockers, but ratification
still requires explicit acceptance, merge of PR #43, and closure of Issue #42.

No material execution is authorized by this review.

## Final review disposition

**Disposition: accept the D0.0 prospective contract for the project-state
synchronization transition, subject to exact-head acceptance and merge.**

This disposition incorporates all five acceptance-review corrections. It does
not authorize corpus processing, model implementation, D0.1 or D0.2 work,
qualification training, or canonical training before the governance transition
is complete.
