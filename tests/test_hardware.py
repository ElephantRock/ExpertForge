"""Tests for optional hardware/topology providers (Issue #7 review item 4)."""

from __future__ import annotations

from expertforge.provenance.hardware import (
    FieldStatus,
    capture_accelerator,
    capture_hardware,
    capture_topology,
)
from expertforge.provenance.record import AcceleratorInfo, TopologyInfo


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


class TestAggregateHardware:
    def test_capture_returns_accelerator_and_topology(self) -> None:
        hw = capture_hardware()
        assert "accelerator" in hw
        assert "topology" in hw
        assert hw["accelerator"].status in ("available", "unavailable", "error")
