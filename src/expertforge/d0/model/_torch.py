"""Lazy :mod:`torch` import helper for the D0 model surface.

The base :mod:`expertforge` installation must not require :mod:`torch`. Every
D0 model submodule imports :mod:`torch` through this helper so that a missing
optional dependency produces a single, actionable error message pointing at the
``d0-model`` extra rather than a bare ``ModuleNotFoundError`` deep inside a
primitive.
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
            "The D0 model surface requires the optional 'd0-model' extra; "
            "install ExpertForge with expertforge[d0-model]"
        ) from exc
