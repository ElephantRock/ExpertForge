"""Deterministic seed derivation and RNG-state management (Issue #8).

Importing this package has no random-state or backend side effects. Callers must
construct and explicitly initialize :class:`RngManager`.
"""

from expertforge.rng.adapters import (
    DeterminismUnavailableError,
    FrameworkAdapterError,
    FrameworkRngAdapter,
    FrameworkSeedResult,
    TorchRngAdapter,
)
from expertforge.rng.derivation import (
    SEED_DERIVATION_SCHEMA,
    SEED_DERIVATION_VERSION,
    DerivedSeed,
    SeedContext,
    derive_seed,
    derive_substream_seed,
    seed_derivation_bytes,
)
from expertforge.rng.manager import RngInitialization, RngManager, RngManagerError
from expertforge.rng.state import (
    RNG_STATE_SCHEMA_VERSION,
    FrameworkRngState,
    NumpyGeneratorState,
    NumpyLegacyState,
    PythonRandomState,
    RngStateBundle,
)

__all__ = [
    "RNG_STATE_SCHEMA_VERSION",
    "SEED_DERIVATION_SCHEMA",
    "SEED_DERIVATION_VERSION",
    "DerivedSeed",
    "DeterminismUnavailableError",
    "FrameworkAdapterError",
    "FrameworkRngAdapter",
    "FrameworkRngState",
    "FrameworkSeedResult",
    "NumpyGeneratorState",
    "NumpyLegacyState",
    "PythonRandomState",
    "RngInitialization",
    "RngManager",
    "RngManagerError",
    "RngStateBundle",
    "SeedContext",
    "TorchRngAdapter",
    "derive_seed",
    "derive_substream_seed",
    "seed_derivation_bytes",
]
