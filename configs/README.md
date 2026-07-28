# `configs/` — Run configurations

Canonical behavior is controlled by **validated, serialized, immutable, fully
resolved configuration** embedded in or referenced by checkpoints and experiment
records (AGENTS.md §4; doctrine/data-and-training.md §3.1).

This directory holds the source configuration files (YAML/TOML) for runs.
Issue #5 (M0.2 validated configuration resolution) defines the loader, schema,
and validation. At scaffold time this directory contains only this purpose note.

Convention to be established by #5:

- human-authored source configs live here;
- the config loader resolves and validates them;
- the *resolved* (frozen) form is recorded with each run/checkpoint/manifest.
