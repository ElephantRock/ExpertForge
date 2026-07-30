"""Direct-construction boundary tests for RNG manager policy domains."""

from __future__ import annotations

from typing import cast

import pytest

from expertforge.rng import RngManager, RngManagerError, SeedContext
from expertforge.rng.state import DeterminismMode, UnsupportedDeterminismPolicy


def test_manager_rejects_unknown_determinism_mode() -> None:
    with pytest.raises(RngManagerError, match="invalid_determinism_mode"):
        RngManager(
            root_seed=7,
            context=SeedContext(component="run"),
            determinism_mode=cast(DeterminismMode, "invalid"),
        )


def test_manager_rejects_unknown_unsupported_policy() -> None:
    with pytest.raises(RngManagerError, match="invalid_unsupported_determinism_policy"):
        RngManager(
            root_seed=7,
            context=SeedContext(component="run"),
            unsupported_determinism=cast(UnsupportedDeterminismPolicy, "ignore"),
        )
