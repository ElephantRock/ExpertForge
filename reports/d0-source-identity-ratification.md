# D0 Source Identity Ratification Report

**Issue:** #42  
**Parent:** #41  
**PR:** #43  
**Status:** D0.0 draft evidence; source-identity tranche implemented

## Result

The D0 proposal now has content-addressed source identities for both selected
upstream inputs. The tokenizer-file and dataset-source-manifest blockers are
removed from the machine-readable contract. Seven ratification blockers remain.

No model, data pipeline, training loop, qualification run, or canonical run is
authorized by this report.

## Dataset correction

The original proposal paired FineWeb-Edu `sample-10BT` with revision
`21974026070c0d94eb843d8eba56d02550f4c0b5`. That revision predates the selected
sample inventory and is therefore superseded.

The corrected dataset identity is:

| Field | Value |
|---|---|
| Repository | `HuggingFaceFW/fineweb-edu` |
| Revision | `84e8104e779e409e2267ac60609138e3dda2cbd2` |
| Configuration | `sample-10BT` |
| Split | `train` |
| License declaration | `ODC-By-1.0` |
| Source files | 14 |
| Total bytes | 28,518,193,415 |
| Manifest | `data/manifests/d0-fineweb-edu-sample-10bt-source-v1.json` |
| Manifest SHA-256 | `d4e7108f2455a95c725fd61fcdb4423be24d1a9a5d3c6e2e322e9df6054bf0df` |

The manifest records every `sample/10BT/*.parquet` path, raw-file SHA-256,
byte size, consumed schema, normalization, ordering, split, and deduplication
semantics.

## Tokenizer identity

| Field | Value |
|---|---|
| Repository | `EleutherAI/gpt-neox-20b` |
| Revision | `364ae95407723fadd1d47b023c1efb92a4d891c3` |
| Family | byte-level BPE |
| Vocabulary | 50,257 |
| Source files | 5 |
| Total bytes | 3,647,931 |
| Manifest | `tokenizers/manifests/d0-gpt-neox-v1.json` |
| Manifest SHA-256 | `eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c` |

The five frozen files are:

| Path | Bytes | SHA-256 |
|---|---:|---|
| `merges.txt` | 456,583 | `0a1f07b1d32153dcdf48f57189b0c217a69b11e11a3bee8a053861987e1640e4` |
| `special_tokens_map.json` | 90 | `c0b3c279b6ecdb71996a86ffb4d4ab94dfdb5df95f00bac9515688faef2ff5dd` |
| `tokenizer.json` | 2,113,710 | `c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624` |
| `tokenizer_config.json` | 156 | `6c50f7b37042a059c71a347be27bc53cc4fcb0f6c8166b00712f3937c91e6bc7` |
| `vocab.json` | 1,077,392 | `4529c7dbac8210536fdda3b82c499bd4fba0a1bf529dd571fef92448ff3f440c` |

## Contract bindings

The proposed contract now records, for each source:

- exact upstream repository and revision;
- repository-local manifest path;
- aggregate manifest SHA-256;
- closure of the corresponding local-identity requirement.

The dataset revision in the contract is constrained to the exact sample-upload
commit. The tokenizer contract remains pinned to its original immutable commit.

## Offline validation

Run:

```bash
uv run python scripts/validate_d0_source_manifests.py
uv run python scripts/validate_d0_contract.py
uv run pytest -q tests/test_d0_source_manifests.py tests/test_d0_contract_proposal.py
```

The source validator rejects:

- missing, duplicate, reordered, or unexpected paths;
- malformed or altered file digests;
- inconsistent byte totals or file counts;
- semantic edits without an updated aggregate manifest digest;
- contract/manifest repository, revision, configuration, license, schema,
  normalization, split, tokenizer, or special-token disagreement;
- reintroduction of the superseded dataset revision.

Permanent tests exercise valid manifests plus file-size drift, duplicate paths,
contract-binding drift, semantic edits, and the superseded dataset revision.

## Closed blockers

1. `tokenizer_file_sha256_and_byte_sizes`
2. `content_addressed_dataset_source_manifest`

## Remaining blockers

1. `formal_experiment_definition_accepted_by_existing_manifest_contract`
2. `resolved_yaml_configs_accepted_by_existing_configuration_layer`
3. `independent_parameter_accounting_executable_and_tests`
4. `committed_generation_prompts_and_contamination_checks`
5. `contract_validation_command_and_CI_gate`
6. `rendered_review_report`
7. `PROJECT_STATE_synchronization`

## Next dependency-ordered action

Add backward-compatible D0 configuration fields and exact qualification and
canonical YAML files, then prove that both resolve through the existing
configuration layer with stable behavioral fingerprints. Formal experiment
manifest binding can proceed alongside that work where schema dependencies
permit.
