"""D0 dense-decoder model architecture primitives.

The package intentionally performs no eager imports of :mod:`torch`. Importing
the base :mod:`expertforge` package must not acquire the model framework
dependency. Each submodule performs a lazy import of :mod:`torch` and raises
:class:`expertforge.d0.errors.MissingOptionalDependencyError` with an actionable
message when the optional ``d0-model`` extra is not installed.

Install the model surface with::

    expertforge[d0-model]
"""

from __future__ import annotations
