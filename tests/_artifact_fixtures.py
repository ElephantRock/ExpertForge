"""Shared construction helpers for Issue #10 artifact tests.

Pure-unit helpers that build a valid :class:`AttemptIdentityRecord` and
:class:`ArtifactStore` without touching git or the network. Importing this
module has no side effects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from expertforge.artifacts import ArtifactStore
from expertforge.identity.fingerprint import specification_fingerprint
from expertforge.identity.record import AttemptIdentityRecord

VALID_RUN = "run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb"
VALID_ATTEMPT = "attempt-20260101t000000z-cccccccccccccccccccc"
VALID_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
VALID_FP = "spec-v1-sha256-" + "0" * 64


def make_identity(
    *,
    run_id: str = VALID_RUN,
    attempt_id: str = VALID_ATTEMPT,
    created_at: datetime = VALID_TS,
) -> AttemptIdentityRecord:
    """A valid identity record with a reconstructable specification fingerprint."""
    return AttemptIdentityRecord(
        specification_fingerprint=specification_fingerprint(b'{"x":1}'),
        run_id=run_id,
        attempt_id=attempt_id,
        created_at_utc=created_at,
    )


def make_store(tmp_path: Path, *, identity: AttemptIdentityRecord | None = None) -> ArtifactStore:
    """An ArtifactStore rooted at ``tmp_path`` for the given identity."""
    ident = identity or make_identity()
    return ArtifactStore(artifact_root=tmp_path / "runs", identity=ident)
