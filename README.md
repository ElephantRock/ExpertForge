# ExpertForge
ExpertForge project repository.

## What ExpertForge is

ExpertForge is a **complete language-model research and engineering project**.
It constructs, trains, evaluates, instruments, and deploys its own dense and
sparse language models from random initialization.

> **Mission:** Build a reproducible language-model system from first principles,
> use controlled experiments to understand each major modeling and training
> decision, and progressively develop architectures whose deployment behavior is
> shaped during training rather than repaired only after training.

> **Central research question:** How should a language model be designed and
> trained when deployment cost, expert locality, memory hierarchy, and hardware
> constraints are treated as first-class architectural objectives?

The primary early product is **experimental control and cumulative technical
knowledge**. Models are versioned artifacts produced by that platform.

## Relationship to ExpertOS

ExpertForge and [ExpertOS](https://github.com/ElephantRock/ExpertOS) are
**separate repositories with distinct responsibilities**.

- **ExpertForge** builds, trains, evaluates, instruments, and deploys its own
  models. It owns data, tokenizer, dense and MoE implementations, training
  systems, evaluation, instrumentation, experiment records, model lineage,
  reference inference, deployment-aware objectives, and versioned interoperability
  contracts.
- **ExpertOS** profiles and controls existing or ExpertForge-produced MoEs
  through its runtime/control plane — placement, residency, movement, prefetch,
  compression, and runtime policy selection.

The repositories remain separate. **ExpertForge must not copy ExpertOS
internals.** They interoperate through versioned schemas, evaluators, routing
traces, resource contracts, and measured deployment evidence.

ExpertForge executes its own models for training, evaluation, generation,
profiling, and reference inference. The statement "model execution happens in
ExpertOS, not ExpertForge" is false and prohibited.

## Canonical state

The `main` branch is the **sole canonical accepted state** of the project. Chat,
drafts, local working-tree changes, unpushed commits, and assistant memory are
provisional until merged into `main`. When `main` conflicts with any other
statement, `main` prevails. See [doctrine/collaboration.md](doctrine/collaboration.md).

## Working on ExpertForge

After bootstrap, all substantive work follows:

```text
GitHub issue
→ working branch
→ implementation or document change
→ validation
→ pull request
→ review
→ merge into main
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).

## Repository layout (planned)

```text
doctrine/        Normative project documents (charter, lineage, doctrine)
configs/         Validated, serialized, fully resolved run configurations
src/             Executable source: data, tokenizer, model, training,
                 evaluation, instrumentation, runtime/reference inference
schemas/         Versioned interoperability schemas + ExpertOS resource contract
tests/           Automated tests
experiments/     Experiment manifests and lightweight evidence
reports/         Experiment reports and decision records
scripts/         Operational and reproduction scripts
```

Core model logic must not live only in notebooks. Canonical behavior is
controlled by validated, serialized, immutable, fully resolved configuration
embedded in or referenced by checkpoints and experiment records.

## Model lineage

The platform is continuous; models are controlled variants. The lineage runs
Dense (D0 → D1 → D2) then MoE (M0 → M1 → M2 → M3 → M4 → M5). Each milestone has
entry and exit criteria. Full definitions:
[doctrine/model-lineage.md](doctrine/model-lineage.md).

| Milestone | Purpose |
|---|---|
| D0 | Minimal dense baseline — validate pipeline, first canonical scientific baseline |
| D1 | Optimized dense baseline — systems efficiency, semantics unchanged |
| D2 | Efficient dense variants — architecture-level techniques |
| M0 | Minimal MoE — correct sparse implementation, honest dense comparison |
| M1 | Balanced-routing MoE — utilization and stability |
| M2 | Instrumented MoE — expert-level scientific instrument |
| M3 | Locality-trained MoE — shape expert working sets |
| M4 | Structured-expert MoE — memory/sharing/compression structures |
| M5 | Native ExpertOS-aware MoE — genuine model–runtime co-design |

## Current state

Milestone 0 (experimental substrate) is the next milestone. See
[PROJECT_STATE.md](PROJECT_STATE.md).
<!-- check-lifecycle-proof -->
