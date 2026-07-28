# Model Lineage Policy

**Status:** Normative
**Applies to:** Any model, baseline, checkpoint, or derived artifact referenced by ExpertForge

## 1. Scope

ExpertForge is a specification/doctrine/schema repository. It does not, at
bootstrap, train, host, or ship model weights. This policy governs how any
model or lineage is *referenced* from ExpertForge once such references are
introduced — for example, when versioned schemas describe ExpertOS runtimes that
consume specific models.

## 2. Lineage must be explicit

Every reference to a model, checkpoint, or derived artifact must record:

* durable identifier (e.g., HuggingFace repo + revision, or org-internal ID);
* version or commit pin the reference targets;
* quantization / precision / configuration, if applicable;
* parent model or parent experiment, if derived;
* the schema version under which the reference is made;
* content hash where one is available;
* location of any large artifact that lives outside GitHub.

A model reference without this provenance is not a canonical ExpertForge result.

## 3. ExpertOS is the runtime counterpart

Model execution happens in ExpertOS, not ExpertForge. ExpertForge records
*which* model and *which* schema version a piece of doctrine or schema targets;
it does not reproduce ExpertOS internals or runtime behavior. The boundary is
the schema contract (see [charter.md](charter.md) §3).

## 4. No weights in this repository

Model weights, checkpoints, datasets, and other large artifacts are never
committed to ExpertForge. They live outside GitHub, and ExpertForge records their
provenance per doctrine §14 and this policy §2.

## 5. Baseline changes require review

Any change to the referenced baseline model or lineage is a baseline change and
requires explicit review before merge (doctrine §10), plus a decision record
under `doctrine/decisions/` when the change is durable.
