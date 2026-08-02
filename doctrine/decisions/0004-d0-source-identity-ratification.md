# Decision Record 0004 — D0 Source Identity Ratification

**Status:** Proposed binding amendment within D0.0  
**Date:** 2026-08-02  
**Issue:** #42  
**Parent:** #41  
**Amends:** Decision Record 0003 source-identity fields only

## Context

The initial D0 contract proposal selected FineWeb-Edu `sample-10BT` while pinning
repository revision `21974026070c0d94eb843d8eba56d02550f4c0b5`. Inspection of the
upstream history showed that this revision predates the commit that introduced
the `sample/10BT` file inventory. Keeping that combination would produce an
internally well-formed but non-resolvable dataset identity.

D0.0 requires immutable, content-addressed tokenizer and dataset identities
before model implementation may merge. Repository revisions alone are not
sufficient: every consumed source object must also have an exact path, byte size,
and SHA-256 digest, and the aggregate manifest must itself be digest-bound.

## Decision

### Dataset revision

The D0 FineWeb-Edu source revision is changed to:

```text
84e8104e779e409e2267ac60609138e3dda2cbd2
```

This is the exact upstream commit that adds the fourteen
`sample/10BT/*.parquet` objects selected by the contract. The superseded revision
`21974026070c0d94eb843d8eba56d02550f4c0b5` is not a valid D0 dataset identity
because it does not contain those objects.

The source inventory is frozen in:

```text
data/manifests/d0-fineweb-edu-sample-10bt-source-v1.json
```

Its declared identity is:

```text
files:                14
total source bytes:   28,518,193,415
manifest SHA-256:     d4e7108f2455a95c725fd61fcdb4423be24d1a9a5d3c6e2e322e9df6054bf0df
license declaration:  ODC-By-1.0
```

The manifest also freezes the consumed schema fields, normalization, source
ordering, deterministic split algorithm, and deduplication keep rule.

### Tokenizer identity

The D0 tokenizer remains the GPT-NeoX byte-level BPE tokenizer at revision:

```text
364ae95407723fadd1d47b023c1efb92a4d891c3
```

The five required source files were fetched once from that immutable revision,
hashed as raw bytes, and recorded in:

```text
tokenizers/manifests/d0-gpt-neox-v1.json
```

Its declared identity is:

```text
files:                5
total source bytes:   3,647,931
manifest SHA-256:     eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c
license declaration:  Apache-2.0
vocabulary size:      50,257
```

The special-token contract remains one `<|endoftext|>` token at ID 0 serving the
BOS, EOS, unknown-token, and document-boundary roles. Training defines no padding
token. Normalization is absent and byte-level pre-tokenization uses
`add_prefix_space=false`.

### Aggregate digest rule

Both manifests use:

```text
sha256(canonical_json_without_manifest_sha256)
```

Canonical JSON means UTF-8 JSON with keys sorted, no insignificant whitespace,
non-ASCII characters preserved, and non-finite numbers prohibited. The contract
records both the repository path and declared aggregate digest for each
manifest.

### Validation boundary

`python scripts/validate_d0_source_manifests.py` operates entirely from committed
files. It performs no network access and rejects:

- absent, duplicate, reordered, or unexpected paths;
- invalid or altered file SHA-256 values;
- non-positive or inconsistent byte sizes;
- aggregate manifest-digest mismatch;
- tokenizer family, vocabulary, padding, or document-boundary drift;
- dataset repository, revision, configuration, license, schema, normalization,
  or split drift;
- disagreement between the contract and either manifest;
- reintroduction of the superseded pre-upload dataset revision.

The main D0 contract validator invokes this source-identity validation before
model, batch, schedule, or threshold validation.

## Alternatives rejected

### Retain the earlier dataset revision

Rejected because the selected `sample-10BT` objects are not present at that
revision. A revision/configuration pair that cannot resolve is not an immutable
identity.

### Pin only repository revisions

Rejected because upstream repository metadata does not independently protect the
individual consumed byte streams against substitution, incomplete retrieval, or
inventory drift.

### Verify upstream files on every CI run

Rejected because permanent validation must be deterministic, offline-capable,
and independent of external service availability. Network retrieval belongs to
controlled acquisition; committed identities are the CI verification input.

### Commit the dataset shards or tokenizer payloads into this repository

Rejected for D0.0. The contract freezes identities and acquisition requirements;
it does not duplicate multi-gigabyte source assets into the source repository.

## Consequences

- The tokenizer-file and dataset-source-manifest ratification blockers are
  closed in the proposed contract.
- Seven D0.0 blockers remain; D0.0 is not ratified.
- D0.1 must acquire exactly these objects and verify raw bytes before processing.
- Any future source-object, tokenizer-semantic, or dataset-revision change is a
  material contract amendment that changes the D0 specification fingerprint.
- No model implementation, qualification run, or canonical run is authorized by
  this decision.

## Reversal conditions

This decision may be superseded only if a dedicated amendment issue:

1. identifies an unavailable, legally unusable, corrupt, or scientifically
   unsuitable source object;
2. freezes a replacement inventory and aggregate digest;
3. explains the effect on comparability and existing evidence;
4. updates every dependent D0 artifact before implementation or execution
   proceeds.
