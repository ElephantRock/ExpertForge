# Data, Tokenizer, and Training Doctrine

**Status:** Normative
**Source:** Issue #1 founding technical specification §9, §10, §11.

## 1. Data doctrine (spec §9)

Data is a versioned experimental input. Canonical datasets record:

- source inventory;
- acquisition / revision information;
- licensing metadata where applicable;
- normalization;
- filtering;
- deduplication;
- language identification;
- quality selection;
- domain weighting;
- contamination controls;
- splits;
- tokenizer version;
- packing rules.

A dataset used by a canonical experiment is **immutable**. Every canonical
example must be attributable to source data, processing revision, and
configuration to the practical limit defined by the data system.

## 2. Tokenizer doctrine (spec §10)

The tokenizer is a versioned model component. Each version records:

- family;
- vocabulary size;
- normalization;
- pre-tokenization;
- special tokens;
- training corpus and sample policy;
- implementation version;
- serialized hash;
- fertility / compression measurements;
- domain behavior;
- compatibility constraints.

Changing the tokenizer creates a new baseline unless the tokenizer itself is the
controlled independent variable.

## 3. Training and checkpoint doctrine (spec §11)

### 3.1 Run declaration

Canonical runs declare: model, tokenizer, dataset, token budget, sequence
length, effective batch size, optimizer, schedule, warmup, weight decay,
clipping, precision, loss scaling, accumulation, distributed strategy,
checkpoint interval, validation interval, seed policy, and hardware.

Progress is primarily measured in **processed tokens**.

### 3.2 Failure recording

The system records: non-finite losses/gradients, abnormal gradient norms,
optimizer instability, data failures, checkpoint failures, throughput
degradation, OOMs, and distributed failures. **Silent skipped updates or
corrupted data invalidate a run.**

### 3.3 Checkpoint contract

A valid checkpoint includes or references:

- model parameters;
- optimizer / scheduler state;
- precision state;
- RNG states;
- sampler / data position;
- global token / update counts;
- resolved configuration;
- source revision;
- tokenizer identity;
- dataset identity;
- format version.

### 3.4 Resume integrity

Resume tests must demonstrate **no** unintended:

- data repeat / skip;
- schedule reset;
- optimizer reset;
- graph change;
- undocumented randomization change.

## 4. Provenance

Data, tokenizer, and training provenance are inseparable from a canonical run.
Provenance records are themselves canonical artifacts and follow the same
source-of-truth and immutability rules as the rest of the repository
(collaboration §1, §14).
