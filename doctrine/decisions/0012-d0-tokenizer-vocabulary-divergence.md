# Decision Record 0012 — D0 Tokenizer Vocabulary Divergence (Amendment)

**Status:** Proposed — escalation; **blocks** D0.1b tokenizer binding (#53) and
downstream D0.1b/D0.3 work until a coherent state is ratified  
**Date:** 2026-08-03  
**Parent:** #41 (D0 program), #53 (D0.1b deterministic training-data pipeline)  
**Amends:** Decisions 0002 (initial D0 architecture), 0003 (D0 baseline contract),
0007 (parameter inventory), 0008 (generation prompts); the ratified D0.0
contract (`experiments/d0/baseline-contract-v1.proposed.json`); the frozen
source manifest `tokenizers/manifests/d0-gpt-neox-v1.json`

## Context

While implementing the D0.1b tokenizer binding (`src/expertforge/d0/data/tokenizer.py`,
Issue #53), the binding's fail-closed vocabulary cross-check discovered a
**pre-existing contradiction between two frozen D0.0 artifacts**. The
contradiction was not introduced by D0.1b; it has been latent since the
tokenizer source manifest was committed and ratified. The binding is designed
to fail closed on exactly this class of mismatch, and it did.

The binding has been **reverted** (working tree returned to commit `ccf65e7`).
No partial tokenizer code is committed. This record documents the divergence so
the amendment can choose one coherent state before the binding re-lands.

## The divergence

### Artifact A — frozen model/contract vocabulary size: **50,257**

The value `vocabulary_size = 50257` is frozen across every artifact that defines
the D0 model's embedding table:

- `configs/d0/canonical-primitive-v2.yaml:12` → `vocab_size: 50257`
- `configs/d0/qualification-primitive-v2.yaml:12` → `vocab_size: 50257`
- `experiments/d0/baseline-contract-v1.proposed.json` →
  `tokenizer.vocabulary_size = 50257`
- `experiments/d0/parameter-inventory-v1.json` (consumed by
  `scripts/validate_d0_parameter_inventory.py:332`)
- `tokenizers/manifests/d0-gpt-neox-v1.json:13` →
  `tokenizer.vocabulary_size = 50257`

This is the **number of embedding rows** and feeds the frozen parameter counts
(Decision 0007):

- canonical 12L/576d/9h/1536ff → **76,738,176** parameters
- qualification 8L/256d/4h/768ff → **19,685,888** parameters

### Artifact B — verified `tokenizer.json` reachable token space: **50,277**

The committed manifest binds `tokenizer.json` with
`sha256 = c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624`,
`size_bytes = 2113710`. Loading those exact verified bytes via
`Tokenizer.from_buffer` (Hugging Face `tokenizers==0.22.2`) yields:

- base BPE vocab (`model.vocab`): **50,254** entries, ids `0..50253`
- added tokens: **25** entries
- effective distinct token space: **50,277** ids, range `0..50276`
- `get_vocab_size(with_added_tokens=True)` → **50,277**
- `get_vocab_size(with_added_tokens=False)` → **50,254**

The 25 added tokens are:

| id | content | special |
|----|---------|---------|
| 0 | `<|endoftext|>` | yes |
| 1 | `<|padding|>` | yes |
| 50254 | 24 spaces | no |
| 50255 | 23 spaces | no |
| 50256 | 22 spaces | no |
| … | (descending space runs) | no |
| 50276 | 2 spaces | no |

That is: 2 special tokens (`<|endoftext|>`, `<|padding|>`) and 23
non-special whitespace-run tokens (lengths 2 through 24).

### Why this is a contradiction

Ids `50257..50276` (20 of the 23 whitespace-run tokens) are **out of range of a
50,257-row embedding table**. A model with `vocab_size = 50257` cannot consume
those ids without an index error or silent truncation. The manifest's
`padding_token: null` is also inconsistent with the file's `<|padding|>` token
at id 1.

### Reachability proof (out-of-range ids are producible from ordinary text)

The following were produced by `Tokenizer.from_buffer(verified_bytes)` with
`add_special_tokens=False` — i.e. without any post-processor, from plain input:

```text
"  hello"            -> ids=[50276, 25521]      out_of_range=[50276]
"   x"               -> ids=[50275, 89]         out_of_range=[50275]
"    y"              -> ids=[50274, 90]         out_of_range=[50274]
"          tab"      -> ids=[50268, 8476]       out_of_range=[50268]
"a                    b" -> ids=[66, 50258, 67] out_of_range=[50258]
```

These are not adversarial inputs; runs of spaces occur in natural text and in
code. Whether the D0 FineWeb-Edu `sample/10BT` corpus actually produces such
ids at materialization time has **not yet been measured** (the 28 GB dataset is
not present in the implementation environment); the amendment must include that
scan as evidence (see Required evidence below).

## Impact if the contradiction is not resolved

- **Parameter counts (Decision 0007):** invalid for either the true token space
  or the embedding row count, depending on resolution.
- **Tokenizer identity (Decision 0004):** the manifest's
  `vocabulary_size = 50257` does not describe the verified artifact.
- **Dataset materialization (D0.1b, #53):** a packer that emits ids `50257..50276`
  feeds the model ids it cannot embed.
- **D0.2 architecture evidence:** the token embedding tensor shape and parameter
  identity (74/110 tensor-identity checks in `tests/test_d0_model_*`) assume
  50,257 rows.
- **Specification fingerprints:** every run's spec fingerprint embeds the
  resolved config, which carries `vocab_size`; changing it changes all spec
  fingerprints and invalidates the U0/R0/R1 reproducibility evidence
  (Milestone 0, #14) if re-derived.
- **Generation prompts (Decision 0008):** prompts containing space runs would
  tokenize to out-of-range ids.

## Parameter-count delta under each resolution

Expanding the model vocabulary from 50,257 to 50,277 (tied embeddings, counted
once) changes only the token-embedding row contribution:

- canonical (width 576): `+20 × 576 = +11,520` → **76,749,696**
- qualification (width 256): `+20 × 256 = +5,120` → **19,691,008**

Stripping the 20 out-of-range added tokens (ids 50257..50276) so the tokenizer's
reachable space is exactly `0..50256` leaves both parameter counts unchanged at
**76,738,176** / **19,685,888** but produces a new `tokenizer.json`
(new `sha256`, new manifest, new golden vectors).

## Candidate resolutions

The binding must fail closed until the amendment selects exactly one coherent
state. Two resolutions are coherent; option 2 is **not** coherent on its own.

1. **Tokenizer-side correction (revise the artifact).** Produce a derived
   `tokenizer.json` whose reachable ids are exactly `0..50256`. This requires:
   a new tokenizer artifact with a new `sha256`; a new source manifest
   (`d0-gpt-neox-v1` → e.g. `d0-gpt-neox-v2` or an explicit derivation record);
   regeneration of all tokenizer golden vectors; a record of *why* the
   GPT-NeoX upstream's 25 added tokens were reduced (provenance). Parameter
   counts and D0.2 embedding evidence are unchanged. Given the volume of
   accepted model evidence tied to 50,257 rows, this is likely the less
   disruptive correction — but that judgment belongs to the amendment, not to
   the binding implementation.

2. **~~Accept a 50,277-token encode surface alongside a 50,257-row model.~~**
   Rejected: a 50,277-token encode surface is incompatible with a 50,257-row
   model. Do not adopt.

3. **Model-side correction (expand the vocabulary).** Set
   `vocabulary_size = 50277` across configs, contract, inventory, and manifest;
   regenerate parameter identities (76,749,696 / 19,691,008), all D0.2
   embedding-shape and parameter-identity evidence, and every spec fingerprint
   / U0/R0/R1 reproducibility record derived from a resolved config.

## Required evidence for the amendment

Before the amendment is ratified, the following must be recorded:

1. **Divergent artifacts and exact digests** (this record, §"The divergence").
2. **All 25 added tokens** with ids, content, and special flag (this record,
   table above).
3. **Corpus reachability scan:** tokenize a sample of the verified FineWeb-Edu
   `sample/10BT` corpus and report whether any document produces ids
   `>= 50257`, with counts. This determines whether the divergence is material
   to training or only theoretical.
4. **Golden-vector outputs** demonstrating out-of-range ids (this record,
   reachability proof above).
5. **Impact list** (this record, §"Impact if the contradiction is not resolved").
6. **Selected resolution** and the explicit list of which prior evidence must be
   regenerated (parameter inventory; D0.2 tensor-identity / numerical-reference
   tests; spec fingerprints; U0/R0/R1 reproducibility if re-derived;
   generation-prompt tokenization if affected).

## Decision

**No decision yet — this is an escalation record.** The D0.1b tokenizer binding
(#53) remains reverted at `ccf65e7` and must not re-land until the amendment
selects resolution 1 or 3 above and the required regeneration is complete.

The binding's fail-closed vocabulary check is correct as written and must be
retained verbatim when the binding re-lands; it is what surfaced this latent
contradiction.

## Provenance of the evidence in this record

- `tokenizers == 0.22.2` (PyPI; would be added as
  `tokenizers>=0.21,<0.23` to the `d0-data` extra — also reverted pending
  amendment)
- `tokenizer.json` loaded via `Tokenizer.from_buffer(verified_bytes)` over the
  descriptor-verified file (`sha256` above), `add_special_tokens=False`
- `manifest_sha256` of `tokenizers/manifests/d0-gpt-neox-v1.json` =
  `eedbff0dbc0af3dc89ebff34155c0c00e73b53a7c82b1611507bd7a5390bd58c`
- Acquisition via `expertforge.d0.source_acquisition.acquire_source_inventory`
  into an untracked `.cache/d0-tokenizer-golden/` (not committed; external
  artifact per doctrine §9)
