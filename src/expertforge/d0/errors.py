"""Typed failures for the D0 immutable-data preflight."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expertforge.d0.training.thresholds import ThresholdDecision


class D0DataError(Exception):
    """Base class for D0 data-preflight failures."""


class SourceManifestError(D0DataError, ValueError):
    """A frozen source manifest is malformed or identity-inconsistent."""


class SourceVerificationError(D0DataError):
    """A local source object cannot be verified against its frozen identity."""


class SourceAcquisitionError(D0DataError):
    """A source object cannot be acquired and atomically published."""


class NormalizationError(D0DataError, ValueError):
    """Document text violates the frozen D0 normalization contract."""


class SourceOrderError(D0DataError, ValueError):
    """Documents or shards were presented outside the frozen source order."""


class ContaminationError(D0DataError, ValueError):
    """The contamination contract or scan input is invalid."""


class ScanReportError(D0DataError, ValueError):
    """A contamination scan report is incomplete or internally inconsistent."""


class MissingOptionalDependencyError(D0DataError, ImportError):
    """An explicitly requested D0 surface is missing its optional dependency."""


class D0PreflightError(D0DataError):
    """Raised when the D0 preflight (source verification + contamination scan) fails."""


class D0TrainingError(Exception):
    """Base class for D0 production training-path failures."""


class ScheduleError(D0TrainingError, ValueError):
    """The frozen D0 learning-rate schedule contract was violated."""


class OptimizerStateError(D0TrainingError, ValueError):
    """D0 optimizer parameter grouping or state capture/restore failed."""


class NonFiniteGradientError(D0TrainingError, ArithmeticError):
    """A non-finite gradient was observed during the D0 training path."""


class PrecisionPreflightError(D0TrainingError):
    """The runtime environment fails the frozen D0 precision preflight."""


class TrainingKilledError(D0TrainingError):
    """A kill-condition threshold fired; the training attempt must terminate.

    Carries the :class:`~expertforge.d0.training.thresholds.ThresholdDecision`
    (typed only — this module must not import the training package at runtime).
    """

    def __init__(self, decision: ThresholdDecision, *, diagnostic_code: str) -> None:
        super().__init__(f"training killed: {decision.reason} [{diagnostic_code}]")
        self.decision = decision
        self.diagnostic_code = diagnostic_code


class RunFailedError(D0TrainingError):
    """A run-failure threshold fired; the attempt stops but is recorded.

    Carries the :class:`~expertforge.d0.training.thresholds.ThresholdDecision`.
    """

    def __init__(self, decision: ThresholdDecision, *, diagnostic_code: str) -> None:
        super().__init__(f"run failed: {decision.reason} [{diagnostic_code}]")
        self.decision = decision
        self.diagnostic_code = diagnostic_code
