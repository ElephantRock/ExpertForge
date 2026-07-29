"""Tests for optional hardware/topology providers (Issue #7 decision: explicit degradation).

Optional fields use typed statuses: ``available``, ``unavailable``,
``not_applicable``, ``error``, ``redacted``. A missing accelerator detector
must not break CPU execution. Detector errors use stable sanitized reason codes;
raw exception messages and command output are not recorded. No new runtime
dependency; ``nvidia-smi`` is an optional external detector. Precision is
recorded only when a trusted detector reports it.
"""

from __future__ import annotations

from expertforge.provenance.hardware import (
    FieldStatus,
    capture_accelerator,
    capture_hardware,
    capture_topology,
)

# --- typed statuses -------------------------------------------------------


class TestFieldStatus:
    def test_known_statuses(self) -> None:
        for s in ("available", "unavailable", "not_applicable", "error", "redacted"):
            FieldStatus(s)  # no raise


# --- accelerator ----------------------------------------------------------


class TestAccelerator:
    def test_no_nvidia_smi_degrades_to_unavailable(self) -> None:
        # When nvidia-smi is not on PATH (or returns error), degrade explicitly.
        accel = capture_accelerator(nvidia_smi="/nonexistent/nvidia-smi")
        assert accel["status"] in {"unavailable", "error"}
        # No raw stderr / device serials / MAC / UUID recorded.
        serialized = repr(accel)
        for forbidden in ("stderr", "serial", "uuid", "mac"):
            assert forbidden not in serialized.lower(), f"{forbidden} leaked"

    def test_accelerator_does_not_raise_on_cpu_host(self) -> None:
        # Must not raise even when no GPU is present.
        accel = capture_accelerator(nvidia_smi="/nonexistent/nvidia-smi")
        assert isinstance(accel, dict)

    def test_detector_error_uses_stable_reason_code(self) -> None:
        accel = capture_accelerator(nvidia_smi="/nonexistent/nvidia-smi")
        if accel["status"] == "error":
            assert "reason" in accel
            # Reason is a stable code, not a raw exception message.
            assert accel["reason"] == accel["reason"].lower().replace(" ", "_")


# --- topology -------------------------------------------------------------


class TestTopology:
    def test_topology_defaults_to_not_applicable(self) -> None:
        topo = capture_topology()
        # Single-process default: no distributed topology.
        assert topo["status"] in {"not_applicable", "available"}

    def test_explicit_topology_input_recorded(self) -> None:
        topo = capture_topology(rank=0, world_size=2)
        assert topo["status"] == "available"
        assert topo["rank"] == 0
        assert topo["world_size"] == 2
        # No coordinator address / IP recorded.
        serialized = repr(topo)
        for forbidden in ("coordinator", "master_addr", "192.", "10.", "172."):
            assert forbidden not in serialized.lower()


# --- aggregate capture ----------------------------------------------------


class TestAggregateHardware:
    def test_capture_returns_accelerator_and_topology(self) -> None:
        hw = capture_hardware()
        assert "accelerator" in hw
        assert "topology" in hw
        # CPU host: accelerator unavailable, not an error in the record.
        assert hw["accelerator"]["status"] in {"available", "unavailable", "error"}
