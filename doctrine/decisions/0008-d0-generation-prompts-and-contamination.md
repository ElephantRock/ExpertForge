# Decision Record 0008 — D0 Generation Prompts and Contamination Contract

**Status:** Proposed binding amendment within D0.0  
**Date:** 2026-08-03  
**Issue:** #42  
**Parent:** #41  
**Amends:** Decision Record 0003 generation-evaluation evidence only

## Context

The D0 baseline contract already fixes the number of generation prompts, decoding
parameters, generation seeds, cadence, and requirement for generated-output
evidence. It did not yet identify the eight prompt texts or define a mechanical
rule for detecting whether those prompts are present in the selected corpus.

A prompt set chosen after data processing or training would permit outcome-aware
selection. A contamination rule that depends on subjective review would not meet
the D0.0 requirement for mechanically decidable evidence. At the same time,
D0.0 does not authorize downloading and processing the complete source corpus,
packing data, generating model outputs, or claiming that an actual corpus scan
has completed.

## Decision

### Fixed prompt set

The prospective prompt set is committed as:

```text
experiments/d0/generation-prompts-v1.json
```

It contains eight ordered prompts covering narrative continuation, mechanistic
explanation, risk-aware planning, systems comparison, causal hypothesis,
diagnostic dialogue, technical inventory, and constrained prioritization.
Each prompt has:

- a stable `d0-gen-NN` identifier;
- an exact category;
- canonical UTF-8 text;
- an exact contamination-probe substring of at least 64 code points;
- SHA-256 values for the prompt and probe text.

The immutable identities are:

| Identity | SHA-256 |
|---|---|
| Raw prompt manifest bytes | `851934e4a49210b515967fad51308038dac1e7bc449774ca603f6bd9f47ccf3e` |
| Canonical prompt payload | `c52ef9f4420ff8160fd5f212370f46cefde71033ac38431bdad2878640070e51` |

The canonical payload contains the prompt-set identifier, canonicalization,
decoding protocol, contamination policy, and ordered prompt records. The raw
manifest identity additionally protects formatting and all envelope metadata.

### Canonicalization and decoding

Prompt and document comparison uses the frozen D0 text normalization:

1. UTF-8 is required;
2. NUL is rejected;
3. CRLF and CR are mapped to LF;
4. all other code points and whitespace are preserved exactly.

Generation uses 128 maximum new tokens, temperature 0.8, top-p 0.95, no top-k
limit, repetition penalty 1.0, stop token 0, and seeds 17, 29, and 43. These
values are checked against the existing D0 evaluation and seed contracts rather
than redefined by the prompt artifact.

### Contamination checks

Every accepted normalized, exact-text-deduplicated document must be scanned
before the deterministic train/validation split. Checks run in this precedence
order for each prompt and document:

1. exact normalized full-prompt substring;
2. exact normalized contamination-probe substring;
3. any exact contiguous 64-code-point prompt window.

At most one precedence-selected hit is recorded per prompt/document. Every hit
records prompt identifier, check type, document identity, source path, physical
row index, and code-point span. Reports have a deterministic ordering.

The maximum permitted hit count is zero. Any hit fails closed and requires a
new prompt-set amendment before packing or training.

### Execution boundary

`scripts/validate_d0_generation_prompts.py` validates prompt identity, decoding,
normalization, contamination mechanics, deterministic hit precedence, and drift
rejection. It can scan supplied normalized document text, but this D0.0 tranche
does not execute the full FineWeb-Edu corpus scan.

The actual corpus-wide scan is a mandatory D0.1 preflight before packing or
training. Its absence is explicit in the validator report:

```text
actual_corpus_scan_completed: false
actual_corpus_scan_stage: D0.1_preflight_before_packing_or_training
```

No generated output, corpus-scan result, data-processing artifact, or training
evidence is claimed by this decision.

## Alternatives rejected

### Choose prompts immediately before evaluation

Rejected because post-training prompt selection permits outcome-aware curation
and weakens comparability across attempts.

### Search only the validation split

Rejected because prompt text in training documents is the primary contamination
risk. The scan therefore covers all accepted deduplicated documents before
splitting.

### Require only full-prompt exact matches

Rejected because a corpus may contain a distinctive prompt fragment or probe
without the complete surrounding text. The probe and 64-code-point window checks
provide mechanically defined partial-match coverage.

### Use fuzzy or semantic similarity

Rejected for D0 because thresholds and implementation choices would be
subjective and difficult to reproduce exactly. Later programs may add separate
semantic-contamination evidence without weakening this exact zero-hit gate.

### Claim a clean corpus from fixture tests

Rejected because unit fixtures validate scanner mechanics, not the selected
28.5-billion-byte source corpus. The actual D0.1 preflight must produce its own
content-addressed report.

## Consequences

- The eight generation prompts are frozen before model implementation or
  training.
- Prompt text, probes, decoding, and contamination thresholds are protected by
  raw and canonical SHA-256 identities.
- Exact, probe, and contiguous-window contamination cases are mechanically
  detectable with deterministic precedence and reporting.
- Existing qualification and canonical configuration fingerprints remain
  unchanged because prompt identity is bound in a separate top-level contract
  section rather than altering the frozen configuration schema.
- The generation-prompt blocker may close after the synchronized repository head
  passes all CI, integration, locked smoke, and recovery gates.
- D0.1 must execute the corpus-wide zero-hit preflight before any packing or
  training begins.
- No model implementation, data processing, qualification run, or canonical run
  is authorized.

## Reversal conditions

This decision may be superseded only by a dedicated amendment that:

1. identifies every changed prompt, probe, canonicalization rule, decoding
   setting, or contamination threshold;
2. produces new raw-manifest and canonical-payload identities;
3. updates the machine-readable D0 contract and all dependent evidence;
4. invalidates any prior contamination reports or generated-output comparisons
   that used the superseded prompt set.
