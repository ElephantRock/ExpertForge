# Decision Record 0005 — D0 Configuration Binding

**Status:** Proposed binding amendment within D0.0  
**Date:** 2026-08-02  
**Issue:** #42  
**Parent:** #41  
**Amends:** Decision Record 0003 configuration representation only

## Context

The D0 execution contract freezes architecture, sources, sequence semantics,
model sizes, batch arithmetic, optimization, schedules, precision, seeds,
evaluation, and decision thresholds. Before D0.1 implementation begins, those
values must be representable by the repository's existing strict configuration
resolver and included in specification identity.

The existing format-version-1 configuration schema supports Milestone 0 and is
already part of accepted run identity. Changing existing defaults or adding a
new always-serialized section would alter canonical bytes and specification
fingerprints for legacy configurations even when their behavior did not change.
That would violate the distinction between additive capability and behavioral
change.

## Decision

### Additive schema boundary

`format_version` remains `1`. A single optional `d0` section is added to
`ConfigRoot`. It is represented by strict, frozen Pydantic models in:

```text
src/expertforge/config/d0_models.py
```

Unknown fields remain forbidden. D0-specific lists authorable in YAML resolve to
immutable tuples. Cross-field validators enforce sequence packing, global batch,
target-token, optimizer-update, token-budget, warmup, and learning-rate
arithmetic before a configuration can be accepted.

When `d0` is absent, it is excluded from behavioral serialization. Consequently,
existing format-version-1 configurations retain their prior canonical bytes.
The accepted regression identity for `configs/smoke.yaml` remains:

```text
canonical configuration SHA-256:
f6cf719aab809aaaf0d59b79cfba15bda7138c9138089bc7bbd4495cca087217
```

### Exact D0 configurations

The two D0 profiles are authored explicitly in:

```text
configs/d0/qualification.yaml
configs/d0/canonical.yaml
```

Neither file relies on defaults for D0 execution semantics. Each states the
complete source, architecture, sequence, model, batch, optimizer,
initialization, schedule, precision, seed, evaluation, and threshold contract.
The legacy top-level fields used by the existing runtime substrate are also
bound to the corresponding D0 values.

### Specification fingerprints

Each resolved configuration is canonicalized by the existing configuration
layer. The dataset and tokenizer manifest digests are then added as immutable
fingerprint inputs with stable names:

```text
dataset.manifest
tokenizer.manifest
```

The committed qualification identity is:

```text
canonical configuration SHA-256:
4e1bdd0ad30bf8b2e1b61f83ff9e387acdab1c0e0d1cb975e7627db95273a960

specification fingerprint:
spec-v1-sha256-4f67b477c9d36c3aa06a4e99f0509380fdc91c672f6cc8aac3dd344880ce5cbe
```

The committed canonical identity is:

```text
canonical configuration SHA-256:
fbc699757b91ad0883fc4d30295a938df86d60f784a9fb91fb39385bbe3e6a4e

specification fingerprint:
spec-v1-sha256-2167b1f07873c3aed6112c38c7de5cdcece6ac3b94d91acde2d605ec2decfb0b
```

Records are stored in:

```text
experiments/d0/qualification-fingerprint.json
experiments/d0/canonical-fingerprint.json
```

### Validation boundary

`python scripts/validate_d0_config_binding.py` resolves both YAML files through
the production configuration layer and independently compares them with the
machine-readable D0 contract. It verifies:

- source repositories, revisions, configurations, manifest paths, and digests;
- every D0 architecture and sequence value;
- model dimensions, head geometry, and declared parameter totals;
- microbatch, accumulation, global-batch, update, and token-budget arithmetic;
- optimizer, initialization, learning-rate, cadence, and precision policy;
- seed, recovery, evaluation, generation, throughput, inference, acceptance,
  failure, and kill semantics;
- agreement between D0 fields and legacy runtime-facing fields;
- the committed specification-fingerprint records and immutable inputs;
- distinct identities for qualification and canonical profiles.

Regression tests additionally require that YAML key ordering does not affect the
fingerprint, meaningful configuration or immutable-input changes do affect it,
and the legacy smoke canonical digest remains unchanged.

CI run 192 passed formatting, Ruff, strict mypy, the full fast CPU suite,
portable CPU integration, the locked smoke tier, and the one-command
interruption/recovery gate on the exact configuration-binding decision-record
head.

## Alternatives rejected

### Add all D0 fields to existing sections

Rejected because it would spread one experiment-family contract across generic
Milestone 0 sections, increase the chance of unintended default behavior, and
make exact contract comparison less auditable.

### Bump the configuration format version

Rejected because no existing source file becomes invalid and no existing field
changes semantics. The extension is additive and absent from legacy behavioral
serialization.

### Allow D0 semantic defaults

Rejected because an execution contract must be reviewable from the authored
configuration. Every D0-relevant value is explicit in both profile files.

### Fingerprint configuration bytes without source manifests

Rejected because dataset and tokenizer byte identities are material inputs. A
configuration with changed source manifests must not retain the same
specification fingerprint.

## Consequences

- Qualification and canonical D0 profiles have stable, distinct specification
  identities.
- Existing format-version-1 configuration fingerprints remain compatible when
  no D0 section is present.
- Any material D0 configuration or source-manifest change produces a new
  specification fingerprint.
- The resolved-configuration ratification blocker is closed; six D0.0 blockers
  remain.
- No model implementation, qualification run, or canonical run is authorized by
  this decision.

## Reversal conditions

This decision may be superseded only through a dedicated amendment that:

1. identifies the field or identity rule that cannot represent the required D0
   behavior;
2. defines migration and compatibility effects for existing configuration and
   fingerprint records;
3. regenerates both D0 fingerprints from independently validated inputs;
4. reruns the complete repository evidence gates before implementation or
   execution proceeds.
