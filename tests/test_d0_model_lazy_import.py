"""Base-environment import contract for the D0 model surface.

These tests intentionally do **not** ``pytest.importorskip("torch")``. They must
pass in both the base environment (no ``d0-model`` extra) and the model
environment (torch installed), so they simulate a torch-less base environment by
hiding :mod:`torch` (and any already-imported D0 model submodules) from
``sys.modules`` for the duration of each check.

The contract being pinned:

- ``import expertforge`` (the base package) succeeds without :mod:`torch`.
- Importing any D0 model surface that requires :mod:`torch` raises
  :class:`expertforge.d0.errors.MissingOptionalDependencyError` whose message
  names the ``d0-model`` extra, instead of leaking a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator
from typing import Any

import pytest

# Each D0 model submodule routes :mod:`torch` through the lazy helper at import
# time. Picking any one of them exercises the typed error path; the full set is
# enumerated so a regression that bypasses the helper for one surface is still
# caught by ``test_all_model_submodules_raise_without_torch``.
_MODEL_SUBMODULES = (
    "expertforge.d0.model.rmsnorm",
    "expertforge.d0.model.rope",
    "expertforge.d0.model.embedding",
    "expertforge.d0.model.swiglu",
    "expertforge.d0.model.attention",
    "expertforge.d0.model.initialization",
    "expertforge.d0.model.transformer",
)


def _is_torch_installed() -> bool:
    """Return True iff :mod:`torch` is importable in the real environment."""

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class _TorchImportBlocker:
    """Context manager that makes ``import torch`` raise ``ModuleNotFoundError``.

    The blocker:

    - Saves and clears any cached ``torch`` / ``torch.*`` modules from
      ``sys.modules`` so the import machinery actually invokes ``__import__``.
    - Saves and clears cached ``expertforge.d0.model.*`` submodules so the next
      access re-runs their top-level ``require_torch()`` guard.
    - Replaces ``builtins.__import__`` with a shim that raises
      ``ModuleNotFoundError("No module named 'torch'")`` for ``torch`` and any
      subpackage, while delegating every other import to the real importer.
    - Restores the original ``__import__`` and ``sys.modules`` state on exit.
    """

    def __init__(self) -> None:
        self._real_import: Any = None
        self._saved_torch_modules: dict[str, Any] = {}
        self._saved_model_modules: dict[str, Any] = {}

    def _save_and_clear(self, prefix: str, target: dict[str, Any]) -> None:
        for key in list(sys.modules):
            if key == prefix or key.startswith(prefix + "."):
                target[key] = sys.modules.pop(key)

    def __enter__(self) -> _TorchImportBlocker:
        self._real_import = builtins.__import__
        self._save_and_clear("torch", self._saved_torch_modules)
        self._save_and_clear("expertforge.d0.model", self._saved_model_modules)
        builtins.__import__ = self._blocked_import  # type: ignore[assignment]
        return self

    def _blocked_import(
        self,
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level == 0 and (name == "torch" or name.startswith("torch.")):
            raise ModuleNotFoundError(f"No module named '{name.split('.')[0]}'")
        return self._real_import(name, globals, locals, fromlist, level)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        builtins.__import__ = self._real_import
        # Restore cached modules so other tests see the previously-imported
        # (real) torch and D0 submodules unchanged.
        sys.modules.update(self._saved_torch_modules)
        sys.modules.update(self._saved_model_modules)


@pytest.fixture
def without_torch() -> Iterator[None]:
    """Run the test body with :mod:`torch` unimportable."""

    with _TorchImportBlocker():
        yield


def test_base_package_imports_without_torch(without_torch: None) -> None:
    # ``import expertforge`` must succeed even when torch is missing.
    importlib.import_module("expertforge")


def test_model_surface_raises_typed_error_without_torch(without_torch: None) -> None:
    from expertforge.d0.errors import MissingOptionalDependencyError

    with pytest.raises(MissingOptionalDependencyError, match="d0-model"):
        importlib.import_module("expertforge.d0.model.transformer")


def test_missing_dependency_error_is_import_error_subclass() -> None:
    # The typed error subclasses ``ImportError`` so ``import`` statements and
    # generic ``except ImportError`` handlers treat it uniformly.
    from expertforge.d0.errors import MissingOptionalDependencyError

    assert issubclass(MissingOptionalDependencyError, ImportError)


def test_all_model_submodules_raise_without_torch(without_torch: None) -> None:
    from expertforge.d0.errors import MissingOptionalDependencyError

    for submodule in _MODEL_SUBMODULES:
        # Drop any cached copy inside the blocker's scope too, in case an
        # earlier iteration in this loop populated ``sys.modules``.
        sys.modules.pop(submodule, None)
        with pytest.raises(MissingOptionalDependencyError, match="d0-model"):
            importlib.import_module(submodule)


def test_model_surface_imports_succeed_when_torch_present() -> None:
    # When torch really is installed, the model surface imports cleanly. This
    # branch is skipped automatically in the base environment.
    if not _is_torch_installed():
        pytest.skip("torch is not installed in this environment")
    importlib.import_module("expertforge.d0.model.transformer")
