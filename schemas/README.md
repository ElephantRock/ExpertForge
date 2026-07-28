# `schemas/` — Versioned interoperability and evidence schemas

ExpertForge owns versioned schemas that define its **boundary with ExpertOS**
and its internal evidence formats (doctrine/charter.md §4; AGENTS.md §3).

Anticipated schemas (each versioned):

- the ExpertOS resource contract (doctrine/deployment-and-runtime-codesign.md §2);
- routing traces and expert-instrumentation artifacts (model-lineage.md F7/F8);
- experiment manifests (#12) and checkpoint format (#11);
- run-configuration and provenance records.

ExpertForge must not copy ExpertOS internals; schemas are the contract, not
source code. At scaffold time this directory contains only this purpose note.
Concrete schemas arrive with the issues that define them (#5, #11, #12, and
later MoE instrumentation).
