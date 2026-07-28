# `experiments/` — Experiment manifests and lightweight evidence

Every formal experiment defines question, hypothesis, independent/dependent
variables, control, fixed constraints, metrics, thresholds, and a decision
(doctrine/evaluation-and-experiments.md §2). Experiment manifests are versioned
(#12, M0.9) and tracked here.

Large artifacts (datasets, checkpoints, traces, profiler outputs) do **not** live
in GitHub; the repository records their provenance (doctrine/collaboration.md
§14): durable artifact identifier, location, content hash, generation command,
parent experiment, format version, retention status.

At scaffold time this directory contains only this purpose note.
