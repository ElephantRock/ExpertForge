# Founding Charter

**Status:** Normative
**Applies to:** ExpertForge repository and all contributors
**Source:** Founding technical specification, originally recorded in Issue #1
and its comments (provenance and implementation source for the founding PR). On
merge, this committed charter on `main` is authoritative; the issue is retained
as provenance and no longer has precedence over `main`.

## 1. Project identity

ExpertForge is a **complete language-model research and engineering project**. It
will construct, train, evaluate, instrument, and deploy its own dense and sparse
language models from random initialization.

Its purpose is to create a **reproducible experimental platform for
model–runtime co-design**, with deployment cost, expert locality, memory
hierarchy, bandwidth, precision, and constrained hardware treated as
first-class architectural objectives.

The primary early product is **experimental control and cumulative technical
knowledge**. Models are versioned artifacts produced by that platform.

ExpertForge is *not* a documentation/schema-only repository, and *not* a copy or
fork of ExpertOS. It owns and executes its own model and inference code.

## 2. Mission

> Build a reproducible language-model system from first principles, use
> controlled experiments to understand each major modeling and training
> decision, and progressively develop architectures whose deployment behavior is
> shaped during training rather than repaired only after training.

## 3. Central research question

> How should a language model be designed and trained when deployment cost,
> expert locality, memory hierarchy, and hardware constraints are treated as
> first-class architectural objectives?

Supporting questions:

- Which dense-model design choices provide the strongest quality, stability, and
  efficiency baseline under constrained compute?
- Under comparable active computation, when does a sparse MoE outperform a dense
  model?
- Which expert properties predict runtime value?
- Can routing locality and transition predictability improve without
  unacceptable quality or load-balance regressions?
- Can expert structure be designed for placement, movement, compression,
  substitution, or partial materialization?
- Can ExpertOS deployment measurements become useful training signals?
- What machine-readable model contract is required for ExpertOS to operate
  without reverse-engineering architecture internals?
- When is hardware-aware training superior to post-training runtime optimization
  alone?

## 4. Repository boundary

### ExpertForge owns

- raw-data ingestion, filtering, deduplication, partitioning, and packing;
- tokenizer training and evaluation;
- dense Transformer implementations;
- sparse MoE implementations;
- optimizers, schedulers, mixed precision, distributed training, checkpointing,
  and recovery;
- evaluation, generation, benchmarking, profiling, and experiment tracking;
- routing and expert instrumentation;
- controlled architectural and training experiments;
- model lineage and experiment evidence;
- reference inference/runtime code needed to understand and test ExpertForge
  models;
- deployment-aware objectives and hardware-cost modeling;
- versioned interoperability schemas and resource contracts.

### ExpertOS owns

- profiling and controlling existing or ExpertForge-produced MoEs through its
  runtime/control plane;
- expert intervention and quality-sensitivity evaluation;
- placement, residency, movement, prefetch, compression, and runtime policy
  selection;
- execution policies for constrained hardware;
- runtime-side evidence and deployment evaluation.

The repositories remain separate. **ExpertForge must not copy ExpertOS
internals.** They interoperate through versioned schemas, evaluators, routing
traces, resource contracts, and measured deployment evidence.

**Prohibited statement:** "model execution happens in ExpertOS, not ExpertForge."
This is false. ExpertForge may and must execute its own models for training,
evaluation, generation, profiling, and reference inference. ExpertOS is the
external specialized runtime/control-plane counterpart.

## 5. Initial objectives

### First major objective

Train a reproducible dense decoder-only model of approximately 50–100 million
parameters from random initialization using a project-owned tokenizer, data
pipeline, model implementation, training loop, evaluation system, and checkpoint
format.

It must demonstrate:

- stable training;
- decreasing validation loss;
- checkpoint creation and restoration;
- deliberate interruption and correct recovery;
- deterministic evaluation under documented conditions;
- text generation;
- measured memory use;
- measured training and inference throughput;
- complete experiment provenance;
- repeatable execution from a declared configuration.

### Second major objective

Train a small sparse MoE with approximately comparable active computation and
compare it honestly with the dense baseline.

No deployment-aware routing objective is a core result until both objectives are
satisfied.

## 6. Non-goals

- ExpertForge is **not** a fork or reimplementation of ExpertOS.
- ExpertForge does **not** reproduce or ship ExpertOS internals; the boundary is
  the versioned contract, not copied source.
- No large training run begins before Milestone 0 (the experimental substrate)
  is complete.
- No deployment-aware objective is treated as a core result before a stable
  ordinary MoE exists (see deployment-aware training policy,
  [deployment-and-runtime-codesign.md](deployment-and-runtime-codesign.md) §1).

## 7. Canonical state

The `main` branch of `ElephantRock/ExpertForge` is the **sole canonical accepted
state** of the project. Chat, drafts, local working-tree changes, unpushed
commits, and assistant memory are provisional until merged into `main`. When
`main` conflicts with any other statement, `main` prevails. See
[collaboration.md](collaboration.md).

After the bootstrap exception, normal work requires: a GitHub issue; a working
branch; validation; a pull request; review; and merge into `main`.

## 8. Founding hypotheses

The initial research program includes:

- **H1:** activation frequency alone is insufficient to predict residency value;
- **H2:** routing locality can be trained without unacceptable quality or
  balance loss;
- **H3:** transition predictability can reduce effective transfer latency enough
  to justify its training cost;
- **H4:** shared-base-plus-delta or related structured experts can reduce
  resident memory while preserving useful specialization;
- **H5:** hardware-aware routing/expert structure can outperform runtime-only
  optimization under constrained deployment;
- **H6:** expert importance is multidimensional;
- **H7:** model-native resource contracts reduce runtime ambiguity.

Each hypothesis requires a separate experiment specification before activation.

## 9. Founding constraint

```text
The platform is continuous.
The evidence is cumulative.
The models are versioned variants.
The baseline is frozen.
The final architecture is not predetermined.
```

No technique is entitled to inclusion. Every technique must earn promotion
through controlled evidence (see promotion criteria,
[evaluation-and-experiments.md](evaluation-and-experiments.md) §3).

## 10. Founding rule

```text
Discussion proposes.
Issues authorize.
Branches implement.
Pull requests demonstrate.
Reviews challenge.
Main decides.
GitHub remembers.
```

## 11. Controlling principle: maturity stages vs. research families

ExpertForge separates two orthogonal lineage axes (see
[model-lineage.md](model-lineage.md)):

- **maturity stages (D0–M5)** describe platform capability and integration
  level;
- **research families (F0–F10)** isolate individual architectural mechanisms —
  exact attention, linear/delta/gated/fine-grained recurrent memory, hybrid
  sequence memory, depth-selective residual memory, conventional MoE, latent
  and structured experts, deployment-aware routing, and integrated
  memory-system interactions.

**No monolithic "K3-like" architecture is implemented as a single lineage
step.** Each mechanism must first exist as a separately controlled research
family. No compound model combining novel families may be promoted before each
constituent mechanism passes independently and the combined experiment has
attribution-preserving interaction ablations (model-lineage §4).

## Doctrine map

| Founding-spec section(s) | File |
|---|---|
| §1–7, §21, §22 | `doctrine/charter.md` (this file) |
| §6 dense architecture policy, §7 D0–M5 maturity stages, §8 scientific baseline rules, F0–F10 research families | `doctrine/model-lineage.md` |
| §9 data, §10 tokenizer, §11 training/checkpoint | `doctrine/data-and-training.md` |
| §12 evaluation, §13 experiment, §14 promotion, §15 kill | `doctrine/evaluation-and-experiments.md` |
| §16 deployment-aware training, §17 ExpertOS contract, §18 evidence loop | `doctrine/deployment-and-runtime-codesign.md` |
| §19 repository layout, §20 Milestone 0 | `AGENTS.md`, `PROJECT_STATE.md` |
| Source-of-truth + workflow | `doctrine/collaboration.md`, `CONTRIBUTING.md` |
| D0 architecture decisions | `doctrine/decisions/0002-initial-d0-architecture.md` |
