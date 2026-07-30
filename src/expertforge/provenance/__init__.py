"""ExpertForge source provenance and execution environment (Issue #7).

Captures source, dependency, platform, and hardware state to attribute and
reproduce every canonical run — without recording secrets.

- :mod:`expertforge.provenance.source_snapshot` — source-code *content* snapshot
  (behavioral identity; feeds Issue #6 ``ImmutableInput``);
- :mod:`expertforge.provenance.record` — identity-bound ``ProvenanceRecord``;
- :mod:`expertforge.provenance.software` — sanitized software environment
  capture;
- :mod:`expertforge.provenance.hardware` — optional hardware/topology providers;
- :mod:`expertforge.provenance.sidecar` — durable provenance sidecar;
- :mod:`expertforge.provenance.summary` — human-readable formatter.

The source-code content snapshot is behavioral identity; the rest of the
provenance record is not and is never hashed into the specification fingerprint.
"""
