"""Checkpoint serialization and exact restoration (Issue #11).

This package owns checkpoint-safe state-tree encoding, deterministic POSIX
ustar tar packaging, manifest construction/validation, compatibility checking,
and exact-restoration ordering.

It may import frozen identity types (:class:`AttemptIdentityRecord`, #6), the
RNG manager and state bundle (:mod:`expertforge.rng`, #8), resolved
configuration bytes (:mod:`expertforge.config.resolve`, #5), and the public
:class:`ArtifactStore` API (#10) for publication only.

It must not import telemetry internals, artifact-private modules, or arbitrary
training-framework serialization. Training-framework adapters are explicit
consumer/provider interfaces defined in this package; the framework implements
them, not the other way around. A pure-Python reference adapter is provided for
testing; the deferred training-framework adapter is out of scope for #11.

No import-time side effects: importing this package does not touch the
filesystem, mutate process-global RNG state, or execute deserialization.

Normative references:
- Design comment ``5145501414`` (ratified proposal).
- Binding amendment ``5145649404`` (amendments A–N). Where the two conflict,
  the amendment prevails.
"""

from __future__ import annotations

__all__: list[str] = []
