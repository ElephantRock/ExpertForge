# Deployment-Aware Training and Model–Runtime Co-design

**Status:** Normative
**Source:** Issue #1 founding technical specification §16, §17, §18.

## 1. Deployment-aware training policy (spec §16)

Deployment-aware objectives begin **only after** an ordinary stable MoE exists.

Candidate terms include: load balance, routing stability, locality, working-set
size, transition predictability, switching, placement / bandwidth / residency
costs, precision sensitivity, contribution sparsity, and expert-family
consistency.

Each term requires:

- precise definition;
- coefficient-selection procedure;
- causal hypothesis;
- quality-risk hypothesis;
- independent ablation;
- kill criterion.

**Multiple penalties must not be enabled together and reported as one
undifferentiated "hardware-aware" result.**

## 2. ExpertOS resource contract (spec §17)

The mature contract should include:

- model and architecture identity;
- layers;
- experts;
- routers;
- tensor boundaries / shapes / dtypes;
- expert byte sizes / grouping / dependencies;
- routing semantics;
- capacity;
- supported precisions / kernels;
- placement constraints;
- substitution relationships;
- fallback paths;
- prefetch hints;
- routing traces;
- quality-risk metadata;
- intervention compatibility.

The contract is **versioned**. ExpertOS should not infer information ExpertForge
can expose directly and reliably. ExpertForge owns the contract; ExpertOS
consumes it. The contract lives under `schemas/` as a versioned artifact, not as
copied ExpertOS source.

## 3. Evidence-exchange loop (spec §18)

```text
train ExpertForge model
→ evaluate model quality
→ profile through ExpertOS
→ identify deployment bottleneck
→ formulate architectural / training hypothesis
→ train controlled variant
→ execute through ExpertOS
→ measure quality and systems effects
→ retain, revise, or reject
```

**ExpertOS feedback is evidence, not an automatic training command.** Decisions
to revise architecture or training follow the promotion criteria in
[evaluation-and-experiments.md](evaluation-and-experiments.md) §3, not a single
runtime measurement.
