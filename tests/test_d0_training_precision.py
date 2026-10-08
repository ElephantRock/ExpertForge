"""Tests for the D0 precision preflight (torch).

The CPU path is verified here: on a machine without CUDA the preflight must
fail closed with the missing-device and missing-BF16 checks recorded. The
positive CUDA path is ``accelerator``-marked and skips without hardware.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from expertforge.config.d0_models import D0PrecisionConfig  # noqa: E402
from expertforge.d0.errors import PrecisionPreflightError  # noqa: E402
from expertforge.d0.training.precision import (  # noqa: E402
    DTYPE_STRATEGY,
    run_precision_preflight,
)


def _config() -> D0PrecisionConfig:
    return D0PrecisionConfig(
        primary_accelerator="single_CUDA_accelerator_with_native_BF16",
        minimum_device_memory_bytes=25769803776,
        minimum_host_memory_bytes=68719476736,
        minimum_free_local_storage_bytes=1099511627776,
        parameter_and_compute_dtype="bfloat16",
        gradient_accumulation_dtype="float32",
        loss_reduction_dtype="float32",
        optimizer_state_dtype="float32",
        master_parameter_dtype="float32",
        canonical_fallback="none_fail_closed",
        qualification_fallback="float32_as_distinct_non_equivalent_attempt",
        fp16_authorized=False,
    )


class TestCpuFailClosed:
    """On a CUDA-less machine the preflight must fail closed, honestly."""

    def test_report_records_the_environment(self, tmp_path: Path) -> None:
        report = run_precision_preflight(_config(), artifact_root=tmp_path)
        assert report.dtype_strategy == DTYPE_STRATEGY
        assert report.canonical_fallback == "none_fail_closed"
        assert report.qualification_fallback == "float32_as_distinct_non_equivalent_attempt"
        assert report.torch_version
        check_names = {c.name for c in report.checks}
        assert {
            "cuda_device_present",
            "native_bf16",
            "minimum_device_memory",
            "minimum_host_memory",
            "minimum_free_storage",
            "fp16_not_authorized",
        } <= check_names

    def test_no_cuda_fails_the_device_checks(self, tmp_path: Path) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA present; the CPU fail-closed path is covered by CI runners")
        report = run_precision_preflight(_config(), artifact_root=tmp_path)
        assert report.cuda_available is False
        by_name = {c.name: c for c in report.checks}
        assert by_name["cuda_device_present"].passed is False
        assert by_name["native_bf16"].passed is False
        assert by_name["minimum_device_memory"].passed is False

    def test_failed_report_raises_preflight_error(self, tmp_path: Path) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA present; the CPU fail-closed path is covered by CI runners")
        report = run_precision_preflight(_config(), artifact_root=tmp_path)
        with pytest.raises(PrecisionPreflightError, match="precision preflight failed"):
            report.raise_if_failed()

    def test_fp16_is_never_authorized(self, tmp_path: Path) -> None:
        report = run_precision_preflight(_config(), artifact_root=tmp_path)
        by_name = {c.name: c for c in report.checks}
        assert by_name["fp16_not_authorized"].passed is True

    def test_storage_check_runs_against_artifact_root(self, tmp_path: Path) -> None:
        report = run_precision_preflight(_config(), artifact_root=tmp_path)
        by_name = {c.name: c for c in report.checks}
        # The check must produce a concrete verdict (pass or a real measure).
        assert by_name["minimum_free_storage"].detail


@pytest.mark.accelerator
class TestCudaPositivePath:
    def test_preflight_passes_on_compliant_cuda_host(self, tmp_path: Path) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA runtime/device unavailable")
        if not torch.cuda.is_bf16_supported():
            pytest.skip("device lacks native BF16 support")
        config = D0PrecisionConfig(
            primary_accelerator="single_CUDA_accelerator_with_native_BF16",
            minimum_device_memory_bytes=1,
            minimum_host_memory_bytes=1,
            minimum_free_local_storage_bytes=1,
            parameter_and_compute_dtype="bfloat16",
            gradient_accumulation_dtype="float32",
            loss_reduction_dtype="float32",
            optimizer_state_dtype="float32",
            master_parameter_dtype="float32",
            canonical_fallback="none_fail_closed",
            qualification_fallback="float32_as_distinct_non_equivalent_attempt",
            fp16_authorized=False,
        )
        report = run_precision_preflight(config, artifact_root=tmp_path)
        assert report.cuda_available is True
        assert report.native_bf16_supported is True
        assert report.device_name is not None
        report.raise_if_failed()  # must not raise
