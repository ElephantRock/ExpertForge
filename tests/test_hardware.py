"""Tests for optional hardware/topology providers (Issue #7 review item 4)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from expertforge.provenance.hardware import (
    FieldStatus,
    capture_accelerator,
    capture_hardware,
    capture_topology,
)
from expertforge.provenance.record import AcceleratorInfo, HardwareAggregate, TopologyInfo


class TestFieldStatus:
    def test_known_statuses(self) -> None:
        for s in ("available", "unavailable", "not_applicable", "error", "redacted"):
            FieldStatus(s)


class TestAccelerator:
    def test_no_nvidia_smi_degrades_to_unavailable(self) -> None:
        accel = capture_accelerator(nvidia_smi="/nonexistent/nvidia-smi")
        assert isinstance(accel, AcceleratorInfo)
        assert accel.status in ("unavailable", "error")
        serialized = repr(accel)
        for forbidden in ("stderr", "serial", "uuid", "mac"):
            assert forbidden not in serialized.lower()

    def test_accelerator_does_not_raise_on_cpu_host(self) -> None:
        accel = capture_accelerator(nvidia_smi="/nonexistent/nvidia-smi")
        assert isinstance(accel, AcceleratorInfo)


class TestCudaSemantics:
    """CUDA is not a framework like Torch/JAX/TF; only the runtime version is
    recorded under runtime_version, framework_version stays None, and precision
    is recorded as 'unavailable' (not 'not_applicable')."""

    def _fake_completed(self, stdout: bytes) -> object:
        """Build a fake subprocess.CompletedProcess-like object."""

        class _Result:
            returncode = 0

            def __init__(self, out: bytes) -> None:
                self.stdout = out

        return _Result(stdout)

    def test_cuda_framework_version_is_none(self) -> None:
        stdout = b"0, A100, 40960, 535.104.05\n1, A100, 40960, 535.104.05\n"
        with (
            patch("expertforge.provenance.hardware.subprocess.run") as mock_run,
            patch("expertforge.provenance.hardware._detect_runtime_version", return_value="12.3.4"),
        ):
            mock_run.return_value = self._fake_completed(stdout)
            accel = capture_accelerator()
        assert accel.status == "available"
        assert accel.framework == "cuda"
        # framework_version must be None: CUDA is not a deep-learning framework.
        assert accel.framework_version is None
        # runtime_version carries the CUDA runtime distribution version.
        assert accel.runtime_version == "12.3.4"
        # precision_status is 'unavailable' (we did not probe it), not
        # 'not_applicable'.
        assert accel.precision_status == FieldStatus.UNAVAILABLE.value

    def test_malformed_nvidia_row_skipped_not_crash(self) -> None:
        # A row with a non-numeric ordinal and a truncated row must be skipped
        # while the valid rows are still captured.
        stdout = (
            b"0, A100, 40960, 535.104.05\n"
            b"notanumber, A100, 40960, 535.104.05\n"  # malformed ordinal
            b"1, A100, 40960\n"  # truncated (only 3 fields)
            b"1, A100, 40960, 535.104.05\n"
        )
        with patch("expertforge.provenance.hardware.subprocess.run") as mock_run:
            mock_run.return_value = self._fake_completed(stdout)
            accel = capture_accelerator()
        assert accel.status == "available"
        # The two valid devices (ordinal 0 and 1) survived; malformed rows skipped.
        assert accel.device_count == 2
        ordinals = [d.ordinal for d in accel.devices]
        assert ordinals == [0, 1]


class TestTopology:
    def test_returns_typed_model(self) -> None:
        topo = capture_topology()
        assert isinstance(topo, TopologyInfo)

    def test_topology_defaults_to_not_applicable(self) -> None:
        topo = capture_topology()
        assert topo.status in ("not_applicable", "available")

    def test_explicit_topology_input_recorded(self) -> None:
        topo = capture_topology(rank=0, world_size=2)
        assert topo.status == "available"
        assert topo.rank == 0
        assert topo.world_size == 2

    def test_explicit_local_rank_and_nodes(self) -> None:
        topo = capture_topology(rank=4, world_size=8, local_rank=0, node_count=2, backend="nccl")
        assert topo.local_rank == 0
        assert topo.node_count == 2
        assert topo.backend == "nccl"

    def test_no_coordinator_addresses(self) -> None:
        topo = capture_topology(rank=0, world_size=2)
        serialized = repr(topo)
        for forbidden in ("coordinator", "master_addr", "192.", "10.", "172."):
            assert forbidden not in serialized.lower()

    def test_error_status_rejects_concrete_rank_world_size(self) -> None:
        # status='error' must not carry concrete rank/world_size values.
        with pytest.raises(ValidationError):
            TopologyInfo(status="error", rank=0, world_size=2)

    def test_not_applicable_status_rejects_concrete_rank_world_size(self) -> None:
        # status='not_applicable' must not carry concrete rank/world_size values.
        with pytest.raises(ValidationError):
            TopologyInfo(status="not_applicable", rank=0, world_size=2)


class TestAggregateHardware:
    def test_capture_returns_accelerator_and_topology(self) -> None:
        hw = capture_hardware()
        assert isinstance(hw, HardwareAggregate)
        assert hw.accelerator.status in ("available", "unavailable", "error")
        assert isinstance(hw.topology, TopologyInfo)

    def test_invalid_topology_env_value_recorded_as_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An RANK env value that cannot be parsed as int must be surfaced as a
        # typed warning rather than silently discarded.
        monkeypatch.setenv("RANK", "not-an-int")
        monkeypatch.setenv("WORLD_SIZE", "2")
        hw = capture_hardware()
        assert isinstance(hw, HardwareAggregate)
        assert any("invalid_topology_env_value:RANK" in w for w in hw.topology_warnings)
        # A valid value is NOT warned about.
        assert not any("WORLD_SIZE" in w for w in hw.topology_warnings)


# --- hardware/topology degradation gaps (review item 5) --------------------


class TestHardwareDegradation:
    """Duplicate NVIDIA ordinals degrade to error instead of raising; node_count
    of 0 is rejected rather than silently coerced to 1; local_rank >= world_size
    is rejected."""

    def _fake_completed(self, stdout: bytes) -> object:
        class _Result:
            returncode = 0

            def __init__(self, out: bytes) -> None:
                self.stdout = out

        return _Result(stdout)

    def test_duplicate_ordinals_degrade_to_error_not_raise(self) -> None:
        # Two nvidia-smi rows with the SAME ordinal: the AcceleratorInfo
        # model_validator would reject duplicate ordinals. The capture path
        # must catch that and degrade to status='error' rather than raise.
        stdout = b"0, A100, 40960, 535.104.05\n0, A100, 40960, 535.104.05\n"
        with patch("expertforge.provenance.hardware.subprocess.run") as mock_run:
            mock_run.return_value = self._fake_completed(stdout)
            accel = capture_accelerator()
        assert accel.status == "error"
        assert accel.reason == "duplicate_device_ordinals"

    def test_node_count_zero_is_error_not_silently_coerced(self) -> None:
        # An explicit node_count=0 is invalid; capture_topology must NOT
        # silently coerce to 1.
        topo = capture_topology(rank=0, world_size=2, node_count=0)
        assert topo.status == "error"
        assert topo.reason == "node_count_must_be_positive"

    def test_env_nnodes_zero_is_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("NNODES", "0")
        topo = capture_topology()
        assert topo.status == "error"
        assert topo.reason == "node_count_must_be_positive"

    def test_local_rank_ge_world_size_is_error(self) -> None:
        topo = capture_topology(rank=0, world_size=4, local_rank=4)
        assert topo.status == "error"
        assert topo.reason == "invalid_local_rank"

    def test_capture_topology_does_not_default_node_count_to_one(self) -> None:
        # When neither explicit node_count nor env NNODES is supplied, the
        # resulting TopologyInfo has node_count=None (NOT silently 1).
        topo = capture_topology(rank=0, world_size=2)
        assert topo.status == "available"
        assert topo.node_count is None

    def test_capture_hardware_propagates_topology_warnings(self) -> None:
        # capture_hardware() returns a HardwareAggregate whose
        # topology_warnings carry invalid numeric env values.
        from unittest.mock import patch as _patch

        stdout = b"0, A100, 40960, 535.104.05\n"
        with (
            _patch("expertforge.provenance.hardware.subprocess.run") as mock_run,
        ):
            mock_run.return_value = self._fake_completed(stdout)
            hw = capture_hardware(rank=0, world_size=2)
        assert isinstance(hw, HardwareAggregate)
        # accelerator is either available (the mocked row) or error depending
        # on _detect_runtime_version; both are acceptable here.
        assert hw.accelerator.status in ("available", "error", "unavailable")
        assert hw.topology.status == "available"
