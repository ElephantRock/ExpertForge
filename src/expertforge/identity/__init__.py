"""ExpertForge run identity and configuration fingerprints (Issue #6).

This package defines durable identities that distinguish a reproducible run
*specification* from an individual execution *attempt*:

- :mod:`expertforge.identity.fingerprint` — versioned specification fingerprint
  envelope and ``spec-v1-sha256-<hex>`` digest;
- :mod:`expertforge.identity.ids` — run and attempt IDs (80 cryptographically
  secure random bits, injectable clock/entropy, collision detection);
- :mod:`expertforge.identity.lineage` — frozen resume-lineage record;
- :mod:`expertforge.identity.record` — per-attempt identity record (frozen
  Pydantic v2);
- :mod:`expertforge.identity.sidecar` — deterministic identity sidecar
  write/read with no-overwrite creation and version/fingerprint verification.
"""
