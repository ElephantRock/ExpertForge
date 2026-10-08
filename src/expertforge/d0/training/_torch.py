"""Lazy :mod:`torch` import helper for the D0 training surface.

Mirrors :mod:`expertforge.d0.model._torch`: the base :mod:`expertforge`
installation must not require :mod:`torch`, and a missing optional
dependency produces one actionable error naming the ``d0-model`` extra.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from expertforge.d0.errors import MissingOptionalDependencyError

__all__ = ["require_torch"]


def require_torch() -> ModuleType:
    """Import and return :mod:`torch`, raising a typed error if it is absent."""
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via lazy path
        raise MissingOptionalDependencyError(
            "The D0 training surface requires the optional 'd0-model' extra; "
            "install ExpertForge with expertforge[d0-model]"
        ) from exc
