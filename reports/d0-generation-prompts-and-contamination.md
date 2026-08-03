# D0 Generation Prompts and Contamination Report

**Issue:** #42  
**Parent:** #41  
**PR:** #43  
**Status:** D0.0 draft evidence; prompt/contamination tranche complete

## Result

D0 now has an ordered set of eight fixed generation prompts and a deterministic,
zero-hit contamination-check contract. Prompt identity is bound independently of
the frozen qualification and canonical YAML configurations, so the existing
configuration SHA-256 values and specification fingerprints remain unchanged.

The validator and tests establish the contract and scanner mechanics only. The
selected FineWeb-Edu corpus has not been scanned in this tranche, and no model
outputs have been generated.

## Immutable identities

| Artifact | SHA-256 |
|---|---|
| Raw `experiments/d0/generation-prompts-v1.json` bytes | `851934e4a49210b515967fad51308038dac1e7bc449774ca603f6bd9f47ccf3e` |
| Canonical prompt payload | `c52ef9f4420ff8160fd5f212370f46cefde71033ac38431bdad2878640070e51` |
| FineWeb-Edu source manifest | `d4e7108f2455a95c725fd61fcdb4423be24d1a9a5d3c6e2e322e9df6054bf0df` |
| GPT-NeoX tokenizer manifest | `eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c` |

## Prompt inventory

| ID | Category |
|---|---|
| `d0-gen-01` | narrative continuation |
| `d0-gen-02` | mechanistic explanation |
| `d0-gen-03` | risk-aware planning |
| `d0-gen-04` | systems comparison |
| `d0-gen-05` | causal hypothesis |
| `d0-gen-06` | diagnostic dialogue |
| `d0-gen-07` | technical inventory |
| `d0-gen-08` | constrained prioritization |

Every record contains exact UTF-8 prompt text, one exact probe substring of at
least 64 code points, and independent prompt/probe SHA-256 values.

## Generation protocol

| Setting | Value |
|---|---:|
| Maximum new tokens | 128 |
| Temperature | 0.8 |
| Top-p | 0.95 |
| Top-k | none |
| Repetition penalty | 1.0 |
| Stop token | 0 |
| Seeds | 17, 29, 43 |

These values are validated against the already-frozen D0 evaluation and seed
contracts.

## Contamination protocol

The eventual D0.1 scan covers all accepted normalized, exact-text-deduplicated
documents before train/validation splitting. For each prompt/document pair, the
scanner selects the first applicable check in this order:

1. exact normalized full-prompt substring;
2. exact normalized probe substring;
3. any exact contiguous 64-code-point prompt window.

The permitted hit count is zero. A hit records the prompt ID, check, document
ID, source file path, physical row index, and code-point span. Any hit fails
closed and requires a prompt-set amendment.

## Validation coverage

The committed tests verify:

- exact raw-manifest and canonical-payload identities;
- prompt ordering, categories, uniqueness, canonicalization, and text hashes;
- decoding and generation-seed agreement with the D0 contract;
- exact-prompt precedence over probe and partial-window checks;
- probe-only and 64-code-point-window detection;
- CRLF/CR normalization and NUL rejection;
- exact-integer row-index enforcement;
- rejection of digest, decoding, prompt-text, probe-length, duplicate-text, and
  payload drift;
- preservation of the existing configuration-binding and formal-experiment
  evidence.

Commands:

```bash
uv run python scripts/validate_d0_generation_prompts.py
uv run pytest -q tests/test_d0_generation_prompts.py
```

CI run 252 passed on the implementation head. Final synchronized CI run 259
passed formatting, Ruff, strict mypy, the complete fast CPU suite, portable CPU
integration, the locked smoke tier, and the one-command interruption/recovery
gate on exact head `bee4fa55904bc6dc1ea0dd983f43f0e59c3fe723`.

The documentation-inclusive head `ea5316b3b4fa1eb10e3c795ac389d02131ced832`
subsequently passed the same complete gate set in CI run 260.

## Evidence boundary

The validator report explicitly states:

```text
actual_corpus_scan_completed: false
actual_corpus_scan_stage: D0.1_preflight_before_packing_or_training
```

This tranche does not claim:

- that the selected source corpus is contamination-free;
- that data normalization, deduplication, splitting, or packing has executed;
- that any model has been instantiated or trained;
- that any generated-output artifact exists.

D0.1 must execute the full source-bound scan and produce a content-addressed
zero-hit report before packing or training. A non-zero result requires an
amendment rather than silent prompt replacement.

## Closed blocker

```text
committed_generation_prompts_and_contamination_checks
```

The permanent aggregate validation command and dedicated CI gate closed in the
subsequent tranche.

## Current remaining dependency order

1. final rendered review report;
2. `PROJECT_STATE.md` synchronization.

The permanent validation command is:

```bash
uv run --locked python -m scripts.validate_d0_ratification
```

No model implementation, data processing, qualification run, or canonical run
is authorized.
