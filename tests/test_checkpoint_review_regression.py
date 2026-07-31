"""Focused regression tests for PR #35 review items 1-16 (Issue #11).

Each test class targets one numbered review finding so a regression in a single
fix is immediately localized. The unit-only tests live here (no integration
mark); the save/load/restore-boundary regressions live in the integration test
module.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from expertforge import checkpoints as cp
from expertforge.checkpoints.models import (
    CHECKPOINT_ARCHIVE_FORMAT_VERSION,
    CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    COMPATIBILITY_SCHEMA_VERSION,
    SAFE_STATE_FORMAT_VERSION,
    CapturedTensor,
    CompatibilityDescriptor,
    CompatibilityResult,
    OptimizerDescriptor,
    OptimizerParamGroup,
    OptimizerStateSlot,
    RngDescriptor,
    SafeValue,
    SchedulerDescriptor,
)
from expertforge.checkpoints.restore import (
    OptimizerState,
    RestoreError,
    RestoreIncompatibleError,
    TensorComponentRef,
)
from expertforge.checkpoints.tar_reader import TarParseError, parse_ustar_archive
from expertforge.checkpoints.tar_writer import build_ustar_archive

# ---------------------------------------------------------------------------
# Item 6: manifest version validators require exact constants
# ---------------------------------------------------------------------------


def _compat_kwargs() -> dict[str, Any]:
    """Full constructor kwargs for a valid CompatibilityDescriptor."""
    from tests._checkpoint_fixtures import make_captured_state

    captured = make_captured_state()
    return dict(
        specification_fingerprint="spec-v1-sha256-" + "0" * 64,
        model_descriptor=captured.model_descriptor,
        optimizer_descriptor=captured.optimizer_descriptor,
        scheduler_descriptor=captured.scheduler_descriptor,
        scaler_descriptor=captured.scaler_descriptor,
        rng_descriptor=captured.rng_descriptor,
        data_descriptor=captured.data_descriptor,
        topology_descriptor=captured.topology_descriptor,
    )


def _compat_descriptor(**overrides: Any) -> CompatibilityDescriptor:
    """A valid descriptor; overrides applied via construction (re-validated)."""
    kw = _compat_kwargs()
    kw.update(overrides)
    return CompatibilityDescriptor(**kw)


class TestItem6ManifestVersionValidators:
    def test_compatibility_descriptor_rejects_unsupported_archive_version(self) -> None:
        with pytest.raises(ValidationError):
            _compat_descriptor(archive_format_version=CHECKPOINT_ARCHIVE_FORMAT_VERSION + 1)

    def test_compatibility_descriptor_rejects_unsupported_manifest_version(self) -> None:
        with pytest.raises(ValidationError):
            _compat_descriptor(manifest_schema_version=CHECKPOINT_MANIFEST_SCHEMA_VERSION + 1)

    def test_compatibility_descriptor_rejects_unsupported_state_version(self) -> None:
        with pytest.raises(ValidationError):
            _compat_descriptor(state_format_version=SAFE_STATE_FORMAT_VERSION + 1)

    def test_compatibility_descriptor_rejects_unsupported_compat_version(self) -> None:
        with pytest.raises(ValidationError):
            _compat_descriptor(compatibility_schema_version=COMPATIBILITY_SCHEMA_VERSION + 1)


# ---------------------------------------------------------------------------
# Item 13: captured snapshot internal consistency + multi-slot optimizer model
# ---------------------------------------------------------------------------


class TestItem13CapturedSnapshotConsistency:
    def test_multi_slot_optimizer_round_trips(self) -> None:
        from tests._checkpoint_fixtures import make_captured_state

        captured = make_captured_state()
        # Build a multi-slot optimizer (two slots per the same param shape).
        slot_a = CapturedTensor(
            logical_name="optimizer.momentum.layer.weight",
            dtype="float32",
            shape=(2, 3),
            raw_bytes=b"\x00" * 24,
        )
        slot_b = CapturedTensor(
            logical_name="optimizer.velocity.layer.weight",
            dtype="float32",
            shape=(2, 3),
            raw_bytes=b"\x00" * 24,
        )
        opt_desc = OptimizerDescriptor(
            optimizer_type="SGD",
            param_groups=(
                OptimizerParamGroup(group_index=0, param_names=("layer.weight",), options=()),
            ),
            state_slots=(
                OptimizerStateSlot(
                    group_index=0,
                    param_name="layer.weight",
                    slot_name="momentum",
                    shape=(2, 3),
                    dtype="float32",
                ),
                OptimizerStateSlot(
                    group_index=0,
                    param_name="layer.weight",
                    slot_name="velocity",
                    shape=(2, 3),
                    dtype="float32",
                ),
            ),
        )
        updated = captured.model_copy(
            update={
                "optimizer": None,
                "optimizer_slots": (slot_a, slot_b),
                "optimizer_descriptor": opt_desc,
            }
        )
        assert len(updated.optimizer_slots) == 2
        assert updated.optimizer is None

    def test_internal_consistency_rejects_descriptor_param_shape_mismatch(self) -> None:
        from tests._checkpoint_fixtures import make_captured_state

        captured = make_captured_state()
        wrong = captured.model_descriptor.model_copy(
            update={
                "parameters": (
                    captured.model_descriptor.parameters[0].model_copy(update={"shape": (3, 3)}),
                )
            }
        )
        dump = captured.model_dump(mode="python")
        dump["model_descriptor"] = wrong.model_dump(mode="python")
        with pytest.raises(ValidationError):
            type(captured).model_validate(dump, strict=False)

    def test_internal_consistency_rejects_empty_rng_bytes(self) -> None:
        from tests._checkpoint_fixtures import make_captured_state

        captured = make_captured_state()
        dump = captured.model_dump(mode="python")
        dump["rng_bundle_bytes"] = b""
        with pytest.raises(ValidationError):
            type(captured).model_validate(dump, strict=False)

    def test_internal_consistency_rejects_data_cursor_sampler_mismatch(self) -> None:
        from tests._checkpoint_fixtures import make_captured_state

        captured = make_captured_state()
        shuffled_cursor = captured.data_cursor.model_copy(
            update={"sampler_type": "shuffled", "permutation_seed": 1}
        )
        dump = captured.model_dump(mode="python")
        dump["data_cursor"] = shuffled_cursor.model_dump(mode="python")
        with pytest.raises(ValidationError):
            type(captured).model_validate(dump, strict=False)


# ---------------------------------------------------------------------------
# Item 12: strict component payload schemas + tensor cross-binding
# ---------------------------------------------------------------------------


class TestItem12StrictSchemas:
    def test_tensor_component_ref_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValidationError):
            TensorComponentRef(  # type: ignore[call-arg]
                member_name="tensors/0.bin",
                logical_names=("a",),
                dtype="float32",
                shape=(1,),
                extra="bad",
            )

    def test_optimizer_state_rejects_unknown_keys(self) -> None:
        from tests._checkpoint_fixtures import make_captured_state

        captured = make_captured_state()
        with pytest.raises(ValidationError):
            OptimizerState.model_validate(
                {
                    "schema": "expertforge.checkpoint-optimizer-state",
                    "version": 1,
                    "descriptor": captured.optimizer_descriptor.model_dump(mode="json"),
                    "has_state": True,
                    "bogus": True,
                },
                strict=False,
            )

    def test_model_state_requires_descriptor_fields(self) -> None:
        # Missing required 'descriptor' on optimizer raises even when stateless.
        from tests._checkpoint_fixtures import make_captured_state

        captured = make_captured_state()
        with pytest.raises(ValidationError):
            OptimizerState.model_validate(
                {
                    "schema": "expertforge.checkpoint-optimizer-state",
                    "version": 1,
                    "has_state": False,
                },
                strict=False,
            )
        # A stateless descriptor-only optimizer is valid.
        st = OptimizerState.model_validate(
            {
                "schema": "expertforge.checkpoint-optimizer-state",
                "version": 1,
                "descriptor": captured.optimizer_descriptor.model_dump(mode="json"),
                "has_state": False,
            },
            strict=False,
        )
        assert st.has_state is False


# ---------------------------------------------------------------------------
# Item 11: canonical tar parser (order, padding, octal, limits)
# ---------------------------------------------------------------------------


def _build_tar(members: list[tuple[str, bytes]]) -> bytes:
    from expertforge.checkpoints.tar_writer import TarMember

    return build_ustar_archive([TarMember(name=n, data=d) for n, d in members])


class TestItem11CanonicalTarParser:
    def test_member_order_enforced(self) -> None:
        # manifest, then state, then tensors is accepted.
        good = _build_tar(
            [
                ("manifest.json", b"{}"),
                ("state/rng.json", b"{}"),
                ("tensors/0.bin", b"x"),
            ]
        )
        members = parse_ustar_archive(good)
        assert [m.name for m in members] == ["manifest.json", "state/rng.json", "tensors/0.bin"]

    def test_missing_manifest_rejected(self) -> None:
        bad = _build_tar([("state/rng.json", b"{}"), ("tensors/0.bin", b"x")])
        with pytest.raises(TarParseError, match="manifest.json"):
            parse_ustar_archive(bad)

    def test_state_after_tensor_rejected(self) -> None:
        bad = _build_tar(
            [("manifest.json", b"{}"), ("tensors/0.bin", b"x"), ("state/rng.json", b"{}")]
        )
        with pytest.raises(TarParseError, match="state component"):
            parse_ustar_archive(bad)

    def test_unrecognized_member_rejected(self) -> None:
        bad = _build_tar([("manifest.json", b"{}"), ("unknown.bin", b"x")])
        with pytest.raises(TarParseError, match="not a recognized checkpoint component"):
            parse_ustar_archive(bad)

    def test_non_contiguous_tensor_indices_rejected(self) -> None:
        bad = _build_tar(
            [("manifest.json", b"{}"), ("tensors/0.bin", b"x"), ("tensors/2.bin", b"x")]
        )
        with pytest.raises(TarParseError, match="contiguous"):
            parse_ustar_archive(bad)

    def test_nonzero_padding_rejected(self) -> None:
        # Build a valid archive then corrupt a padding byte.
        good = _build_tar(
            [("manifest.json", b"{}"), ("state/rng.json", b"ab"), ("tensors/0.bin", b"cd")]
        )
        corrupted = bytearray(good)
        # Find the padding region of the manifest member (data "{}" = 2 bytes,
        # padded to 512: header at 0, data at 512..514, padding 514..1024).
        corrupted[514] = 0x01
        with pytest.raises(TarParseError, match="non-zero padding"):
            parse_ustar_archive(bytes(corrupted))

    def test_member_count_limit_enforced(self) -> None:
        # A manifest + one tensor is fine; force a tiny limit to trigger it.
        good = _build_tar([("manifest.json", b"{}"), ("tensors/0.bin", b"x")])
        with pytest.raises(TarParseError, match="member count"):
            parse_ustar_archive(good, max_member_count=1, validate_member_order=False)

    def test_archive_byte_limit_enforced(self) -> None:
        good = _build_tar([("manifest.json", b"{}"), ("tensors/0.bin", b"x")])
        with pytest.raises(TarParseError, match="archive size"):
            parse_ustar_archive(good, max_archive_bytes=10)


# ---------------------------------------------------------------------------
# Item 8: RestoreTransaction mandatory compatibility gate
# ---------------------------------------------------------------------------


class TestItem8CompatibilityGate:
    def test_require_exact_compatibility_raises_on_mismatch(self) -> None:
        from expertforge.checkpoints.restore import RestoreTransaction

        # A bare-bones fake archive: only the compatibility comparison matters.
        class _FakeArchive:
            def __init__(self, desc: CompatibilityDescriptor) -> None:
                self.manifest = type("M", (), {"compatibility": desc})()

        captured_desc = _compat_descriptor()
        expected_desc = captured_desc.model_copy(
            update={"specification_fingerprint": "spec-v1-sha256-" + "1" * 64}
        )
        from tests._checkpoint_fixtures import make_rng_bundle

        txn = RestoreTransaction(
            archive=_FakeArchive(captured_desc),  # type: ignore[arg-type]
            factory=None,  # type: ignore[arg-type]
            rng_bundle_loader=make_rng_bundle,
            rng_consumer=lambda b: None,
            expected_descriptor=expected_desc,
        )
        with pytest.raises(RestoreIncompatibleError):
            txn.prepare()

    def test_incompatible_error_is_typed_restore_error(self) -> None:
        assert issubclass(RestoreIncompatibleError, RestoreError)


# ---------------------------------------------------------------------------
# Item 7 + 14: compatibility comparison of added required fields
# ---------------------------------------------------------------------------


class TestItem7And14CompatibilityFields:
    def _mismatch_components(self, result: CompatibilityResult) -> set[str]:
        return {m.component for m in result.mismatches}

    def test_optimizer_options_mismatch_blocks(self) -> None:
        from expertforge.checkpoints.store import check_compatibility
        from tests._checkpoint_fixtures import make_captured_state

        captured = make_captured_state()
        a = captured.optimizer_descriptor
        # Add an option (lr) to the expected descriptor's group.
        opt_with_option = a.model_copy(
            update={
                "param_groups": (
                    OptimizerParamGroup(
                        group_index=0,
                        param_names=("layer.weight",),
                        options=(("lr", _float_sv(0.1)),),
                    ),
                )
            }
        )
        expected = _compat_descriptor(optimizer_descriptor=opt_with_option)
        actual = _compat_descriptor()
        result = check_compatibility(actual, expected)
        assert result.status == "incompatible"
        assert "optimizer_options" in self._mismatch_components(result)

    def test_optimizer_slot_dtype_mismatch_blocks(self) -> None:
        from expertforge.checkpoints.store import check_compatibility
        from tests._checkpoint_fixtures import make_captured_state

        captured = make_captured_state()
        slot_float = captured.optimizer_descriptor.state_slots[0]
        slot_int = slot_float.model_copy(update={"dtype": "int32"})
        opt_int = captured.optimizer_descriptor.model_copy(update={"state_slots": (slot_int,)})
        expected = _compat_descriptor(optimizer_descriptor=opt_int)
        actual = _compat_descriptor()
        result = check_compatibility(actual, expected)
        assert result.status == "incompatible"
        assert "optimizer_slots" in self._mismatch_components(result)

    def test_scheduler_active_mismatch_blocks(self) -> None:
        from expertforge.checkpoints.store import check_compatibility

        inactive = SchedulerDescriptor(scheduler_type="constant", active=False)
        expected = _compat_descriptor(scheduler_descriptor=inactive)
        actual = _compat_descriptor()
        result = check_compatibility(actual, expected)
        assert result.status == "incompatible"
        assert "scheduler_identity" in self._mismatch_components(result)

    def test_rng_framework_version_mismatch_blocks(self) -> None:
        from expertforge.checkpoints.store import check_compatibility
        from tests._checkpoint_fixtures import make_captured_state

        captured = make_captured_state()
        rng_with_fw = RngDescriptor(
            adapter_set=captured.rng_descriptor.adapter_set,
            framework_versions=(("numpy", "2.0.0"),),
        )
        expected = _compat_descriptor(rng_descriptor=rng_with_fw)
        actual = _compat_descriptor()
        result = check_compatibility(actual, expected)
        assert result.status == "incompatible"
        assert "rng_schema" in self._mismatch_components(result)

    def test_topology_rank_assignment_mismatch_blocks(self) -> None:
        from expertforge.checkpoints.store import check_compatibility
        from tests._checkpoint_fixtures import make_captured_state

        captured = make_captured_state()
        topo = captured.topology_descriptor.model_copy(update={"rank_assignment": (9,)})
        expected = _compat_descriptor(topology_descriptor=topo)
        actual = _compat_descriptor()
        result = check_compatibility(actual, expected)
        assert result.status == "incompatible"
        assert "topology" in self._mismatch_components(result)

    def test_data_identity_length_mismatch_blocks(self) -> None:
        from expertforge.checkpoints.store import check_compatibility
        from tests._checkpoint_fixtures import make_captured_state

        captured = make_captured_state()
        wrong_len = captured.data_descriptor.model_copy(
            update={
                "identity": captured.data_descriptor.identity.model_copy(update={"length": 9999})
            }
        )
        expected = _compat_descriptor(data_descriptor=wrong_len)
        actual = _compat_descriptor()
        result = check_compatibility(actual, expected)
        assert result.status == "incompatible"
        assert "data_identity" in self._mismatch_components(result)


def _float_sv(value: float) -> SafeValue:
    import struct

    from expertforge.checkpoints.models import _SafeFloat

    bit = int.from_bytes(struct.pack(">d", value), "big", signed=False)
    return _SafeFloat(width_bits=64, bit_pattern=bit)


# ---------------------------------------------------------------------------
# Item 4: encoder rejects a disagreeing separately-supplied rng_bundle
# ---------------------------------------------------------------------------


class TestItem4EncoderUsesCapturedRng:
    def test_disagreeing_rng_bundle_rejected(self, tmp_path: Any) -> None:
        from datetime import UTC, datetime

        from expertforge.checkpoints.encoder import EncoderError, build_archive
        from expertforge.config.resolve import resolve_config
        from expertforge.rng.derivation import SeedContext
        from expertforge.rng.manager import RngManager
        from tests._checkpoint_fixtures import (
            CONFIGS,
            make_captured_state,
            make_identity_and_provenance,
        )

        identity, provenance = make_identity_and_provenance(tmp_path)
        captured = make_captured_state()
        envelope = resolve_config(CONFIGS / "smoke.yaml")
        # A DIFFERENT bundle (different seed) than captured.rng_bundle_bytes.
        other = RngManager(root_seed=99, context=SeedContext(component="run"))
        other.initialize()
        other_bundle = other.capture_state()
        with pytest.raises(EncoderError):
            build_archive(
                identity=identity,
                captured=captured,
                configuration_envelope=envelope,
                provenance=provenance,
                rng_bundle=other_bundle,
                parent=None,
                created_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
            )


# ---------------------------------------------------------------------------
# Item 16: package public API surface
# ---------------------------------------------------------------------------


class TestItem16PackagePublicApi:
    def test_all_documented_names_exported(self) -> None:
        required = {
            "CheckpointStore",
            "CheckpointManifest",
            "CheckpointArchive",
            "RestoreTransaction",
            "SafeValue",
            "CapturedTensor",
            "CapturedCheckpointState",
            "StateProvider",
            "StateConsumer",
            "check_compatibility",
            "build_archive",
            "TensorMemberRef",
            "CompatibilityResult",
        }
        exported = set(cp.__all__)
        missing = required - exported
        assert not missing, f"public API missing: {missing}"

    def test_star_import_succeeds(self) -> None:
        # ``from expertforge.checkpoints import *`` must not raise and must bind
        # the documented names (item 16 package-import regression).
        ns: dict[str, Any] = {}
        exec("from expertforge.checkpoints import *", ns)
        assert "CheckpointStore" in ns
        assert "RestoreTransaction" in ns
        assert "CapturedCheckpointState" in ns

    def test_no_stale_empty_all(self) -> None:
        # The previously-empty __all__ is now populated.
        assert len(cp.__all__) > 30
