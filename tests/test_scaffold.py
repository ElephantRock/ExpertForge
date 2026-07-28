"""Scaffold smoke test (Issue #4).

Confirms the package is importable and reports its declared version. This is
the minimal test that lets `pytest` succeed from a clean checkout. Richer
coverage arrives with later Milestone 0 issues.
"""

import expertforge


def test_package_importable() -> None:
    """The package must be importable from a clean, synced environment."""
    assert expertforge is not None


def test_version_present() -> None:
    """The package exposes a declared version string."""
    assert isinstance(expertforge.__version__, str)
    assert expertforge.__version__  # non-empty
