"""ExpertForge validated configuration system (Issue #5).

This package implements typed, validated, serializable run configuration:

- :mod:`expertforge.config.yaml_loader` — restricted YAML authoring format;
- :mod:`expertforge.config.models` — frozen Pydantic configuration sections;
- :mod:`expertforge.config.overrides` — dotted-path ``--set`` overrides;
- :mod:`expertforge.config.resolve` — resolution envelope and canonical bytes;
- :mod:`expertforge.config.cli` — command-line entrypoint.

YAML is the authoring format only. The canonical serialized representation is
deterministic JSON (see :mod:`expertforge.config.resolve`).
"""
