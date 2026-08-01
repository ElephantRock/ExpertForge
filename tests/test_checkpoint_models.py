"""Regression tests for the checkpoint frozen models (Issue #11)."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import pytest
from pydantic import ValidationError

from expertforge.checkpoints.models import (
    MAX_SAFE_DEPTH,
    AliasGroup,
    CapturedTensor,
    CompatibilityDescriptor,
    CompatibilityMismatch,
    CompatibilityResult,
    CounterSnapshot,
    DataCursor,
    DataDescriptor,
    DataIdentity,
    ModelDescriptor,
    ModelParameterDescriptor,
    OptimizerDescriptor,
    OptimizerParamGroup,
    OptimizerStateSlot,
    ResourceLimitError,
    RngDescriptor,
    SafeStateDecodeError,
    ScalerDescriptor,
    SchedulerDescriptor,
    TopologyDescriptor,
    assert_safe_bounds,
)


def _sv(value: dict[str, Any]) -> Any:
    """Build a SafeValue from a plain dict payload."""
    from expertforge.checkpoints.models import (
        _SafeBool,
        _SafeBytes,
        _SafeFloat,
        _SafeInt,
        _SafeMapping,
        _SafeNull,
        _SafeSequence,
        _SafeString,
    )

    kind = value["kind"]
    if kind == "null":
        return _SafeNull()
    if kind == "bool":
        return _SafeBool(value=value["value"])
    if kind == "int":
        return _SafeInt(
            width_bits=value["width_bits"],
            signed=value["signed"],
            value=value["value"],
        )
    if kind == "float":
        return _SafeFloat(width_bits=value["width_bits"], bit_pattern=value["bit_pattern"])
    if kind == "string":
        return _SafeString(value=value["value"])
    if kind == "bytes":
        raw = value["raw"]
        return _SafeBytes(
            value=base64.b64encode(raw).decode("ascii"),
            byte_length=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    if kind == "sequence":
        return _SafeSequence(value=tuple(_sv(v) for v in value["items"]))
    if kind == "mapping":
        items = sorted(value["items"], key=lambda kv: kv[0])
        return _SafeMapping(value=tuple((k, _sv(v)) for k, v in items))
    raise AssertionError(f"unknown kind {kind!r}")


class TestSafeValue:
    def test_null_round_trip(self) -> None:
        v = _sv({"kind": "null"})
        assert v.kind == "null"

    def test_bool_round_trip(self) -> None:
        v = _sv({"kind": "bool", "value": True})
        assert v.kind == "bool" and v.value is True

    def test_int_signed_range(self) -> None:
        v = _sv({"kind": "int", "width_bits": 8, "signed": True, "value": -1})
        assert v.value == -1
        with pytest.raises(ValidationError):
            _sv({"kind": "int", "width_bits": 8, "signed": True, "value": 128})
        with pytest.raises(ValidationError):
            _sv({"kind": "int", "width_bits": 8, "signed": True, "value": -129})

    def test_int_unsigned_range(self) -> None:
        v = _sv({"kind": "int", "width_bits": 8, "signed": False, "value": 255})
        assert v.value == 255
        with pytest.raises(ValidationError):
            _sv({"kind": "int", "width_bits": 8, "signed": False, "value": 256})
        with pytest.raises(ValidationError):
            _sv({"kind": "int", "width_bits": 8, "signed": False, "value": -1})

    def test_bool_not_accepted_as_int(self) -> None:
        # strict mode rejects bool where int is required
        from expertforge.checkpoints.models import _SafeInt

        with pytest.raises(ValidationError):
            _SafeInt(width_bits=8, signed=True, value=True)

    def test_float_bit_pattern_preserves_nan(self) -> None:
        # float32 NaN bit pattern: 0x7fc00000
        v = _sv({"kind": "float", "width_bits": 32, "bit_pattern": 0x7FC00000})
        assert v.bit_pattern == 0x7FC00000

    def test_float_invalid_width(self) -> None:
        with pytest.raises(ValidationError):
            _sv({"kind": "float", "width_bits": 24, "bit_pattern": 0})

    def test_bytes_canonical_base64(self) -> None:
        raw = b"hello"
        v = _sv({"kind": "bytes", "raw": raw})
        assert v.byte_length == 5
        assert v.sha256 == hashlib.sha256(raw).hexdigest()

    def test_bytes_digest_mismatch_rejected(self) -> None:
        raw = b"hello"
        with pytest.raises(ValidationError):
            from expertforge.checkpoints.models import _SafeBytes

            _SafeBytes(
                value=base64.b64encode(raw).decode("ascii"),
                byte_length=len(raw),
                sha256="0" * 64,
            )

    def test_string_round_trip(self) -> None:
        v = _sv({"kind": "string", "value": "héllo"})
        assert v.value == "héllo"

    def test_sequence_round_trip(self) -> None:
        v = _sv(
            {
                "kind": "sequence",
                "items": [
                    {"kind": "int", "width_bits": 32, "signed": True, "value": 1},
                    {"kind": "bool", "value": False},
                ],
            }
        )
        assert len(v.value) == 2

    def test_mapping_canonical_sorted_keys(self) -> None:
        v = _sv(
            {
                "kind": "mapping",
                "items": [
                    ("b", {"kind": "null"}),
                    ("a", {"kind": "null"}),
                ],
            }
        )
        assert [k for k, _ in v.value] == ["a", "b"]

    def test_mapping_unsorted_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _sv(
                {
                    "kind": "mapping",
                    "items": [
                        ("a", {"kind": "null"}),
                        ("b", {"kind": "null"}),
                        ("a", {"kind": "null"}),
                    ],
                }
            )

    def test_nested_round_trip(self) -> None:
        v = _sv(
            {
                "kind": "mapping",
                "items": [
                    (
                        "a",
                        {
                            "kind": "sequence",
                            "items": [
                                {"kind": "int", "width_bits": 8, "signed": True, "value": 3},
                                {"kind": "string", "value": "x"},
                            ],
                        },
                    ),
                    ("b", {"kind": "null"}),
                ],
            }
        )
        assert_safe_bounds(v)

    def test_depth_limit(self) -> None:
        # Build a chain of nesting deeper than MAX_SAFE_DEPTH.
        inner: dict[str, Any] = {"kind": "null"}
        for _ in range(MAX_SAFE_DEPTH + 2):
            inner = {"kind": "sequence", "items": [inner]}
        v = _sv(inner)
        with pytest.raises(ResourceLimitError):
            assert_safe_bounds(v)


class TestCapturedTensor:
    def test_basic(self) -> None:
        t = CapturedTensor(
            logical_name="layer.weight",
            dtype="float32",
            shape=(2, 3),
            raw_bytes=b"\x00" * 24,
        )
        assert t.member_byte_size() == 24

    def test_byte_size_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CapturedTensor(logical_name="w", dtype="float32", shape=(2, 3), raw_bytes=b"\x00" * 5)

    def test_zero_element_tensor(self) -> None:
        t = CapturedTensor(logical_name="empty", dtype="float32", shape=(0,), raw_bytes=b"")
        assert t.member_byte_size() == 0

    def test_rank_zero_tensor(self) -> None:
        t = CapturedTensor(
            logical_name="scalar", dtype="int32", shape=(), raw_bytes=b"\x00\x00\x00\x00"
        )
        assert t.member_byte_size() == 4

    def test_bool_canonical_bytes(self) -> None:
        CapturedTensor(logical_name="m", dtype="bool", shape=(2,), raw_bytes=b"\x00\x01")
        with pytest.raises(ValidationError):
            CapturedTensor(logical_name="m", dtype="bool", shape=(2,), raw_bytes=b"\x00\x02")

    def test_rank_limit(self) -> None:
        with pytest.raises(ValidationError):
            CapturedTensor(
                logical_name="w", dtype="float32", shape=(1,) * 20, raw_bytes=b"\x00" * 4
            )


def _data_identity() -> DataIdentity:
    return DataIdentity(
        dataset_digest="a" * 64,
        split="train",
        length=1000,
        preprocessing_identity="prep.v1",
        tokenizer_identity="tok.v1",
        packing_policy="no_packing",
        sequence_policy="fixed_128",
        shard_selection="single",
        data_config_digest="b" * 64,
    )


def _data_descriptor() -> DataDescriptor:
    return DataDescriptor(
        identity=_data_identity(),
        sampler_type="sequential",
        sampler_version=1,
        batch_size=8,
        sequence_length=128,
        drop_last=False,
    )


def _model_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        parameters=(ModelParameterDescriptor(name="layer.weight", shape=(2, 3), dtype="float32"),),
        buffers=(),
    )


def _optimizer_descriptor() -> OptimizerDescriptor:
    return OptimizerDescriptor(
        optimizer_type="SGD",
        param_groups=(
            OptimizerParamGroup(
                group_index=0,
                param_names=("layer.weight",),
                options=(),
            ),
        ),
        state_slots=(
            OptimizerStateSlot(
                group_index=0,
                param_name="layer.weight",
                slot_name="momentum",
                shape=(2, 3),
                dtype="float32",
            ),
        ),
    )


def _compat_descriptor(*, scaler: bool = False) -> CompatibilityDescriptor:
    return CompatibilityDescriptor(
        specification_fingerprint="spec-v1-sha256-" + "0" * 64,
        model_descriptor=_model_descriptor(),
        optimizer_descriptor=_optimizer_descriptor(),
        scheduler_descriptor=SchedulerDescriptor(scheduler_type="constant", active=True),
        scaler_descriptor=ScalerDescriptor(scaler_type="grad_scaler") if scaler else None,
        rng_descriptor=RngDescriptor(adapter_set=(), framework_versions=()),
        data_descriptor=_data_descriptor(),
        topology_descriptor=TopologyDescriptor(world_size=1, rank_assignment=(0,)),
    )


class TestCounters:
    def test_valid(self) -> None:
        c = CounterSnapshot(
            global_update=5,
            completed_microsteps=10,
            accumulation_position=0,
            accepted_samples=80,
            accepted_sequences=10,
            processed_tokens=1000,
        )
        assert c.global_update == 5

    def test_accumulation_position_nonzero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CounterSnapshot(
                global_update=5,
                completed_microsteps=10,
                accumulation_position=2,
                accepted_samples=80,
                accepted_sequences=10,
                processed_tokens=1000,
            )


class TestDataCursor:
    def test_sequential(self) -> None:
        c = DataCursor(
            sampler_type="sequential",
            sampler_version=1,
            batch_size=8,
            sequence_length=128,
            drop_last=False,
            epoch=0,
            position=10,
            accepted_samples=80,
            accepted_sequences=10,
        )
        assert c.permutation_seed is None

    def test_shuffled_requires_seed(self) -> None:
        DataCursor(
            sampler_type="shuffled",
            sampler_version=1,
            batch_size=8,
            sequence_length=128,
            drop_last=True,
            epoch=0,
            position=10,
            permutation_seed=42,
            accepted_samples=80,
            accepted_sequences=10,
        )
        with pytest.raises(ValidationError):
            DataCursor(
                sampler_type="shuffled",
                sampler_version=1,
                batch_size=8,
                sequence_length=128,
                drop_last=True,
                epoch=0,
                position=10,
                accepted_samples=80,
                accepted_sequences=10,
            )

    def test_sequential_forbids_seed(self) -> None:
        with pytest.raises(ValidationError):
            DataCursor(
                sampler_type="sequential",
                sampler_version=1,
                batch_size=8,
                sequence_length=128,
                drop_last=False,
                epoch=0,
                position=10,
                permutation_seed=42,
                accepted_samples=80,
                accepted_sequences=10,
            )


class TestModelDescriptor:
    def test_alias_group_round_trip(self) -> None:
        m = ModelDescriptor(
            parameters=(
                ModelParameterDescriptor(name="a", shape=(2,), dtype="float32"),
                ModelParameterDescriptor(name="b", shape=(2,), dtype="float32"),
            ),
            buffers=(),
            alias_groups=(AliasGroup(canonical_member="a", aliased_names=("b",)),),
        )
        assert m.alias_groups[0].canonical_member == "a"

    def test_alias_unknown_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelDescriptor(
                parameters=(ModelParameterDescriptor(name="a", shape=(2,), dtype="float32"),),
                buffers=(),
                alias_groups=(AliasGroup(canonical_member="a", aliased_names=("z",)),),
            )


class TestTopology:
    def test_world_size_1_ok(self) -> None:
        TopologyDescriptor(world_size=1, rank_assignment=(0,))

    def test_world_size_2_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyDescriptor(world_size=2, rank_assignment=(0, 1))


class TestCompatibilityResult:
    def test_exact_no_mismatches(self) -> None:
        r = CompatibilityResult(status="exact", mismatches=())
        assert r.status == "exact"

    def test_exact_with_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompatibilityResult(
                status="exact",
                mismatches=(
                    CompatibilityMismatch(
                        component="optimizer_type",
                        diagnostic_code="optimizer_type_mismatch",
                        severity="blocking",
                        expected="SGD",
                        actual="Adam",
                    ),
                ),
            )

    def test_incompatible_sorted_mismatches(self) -> None:
        r = CompatibilityResult(
            status="incompatible",
            mismatches=(
                CompatibilityMismatch(
                    component="model_shapes",
                    diagnostic_code="parameter_shape_mismatch",
                    severity="blocking",
                    expected="(2,3)",
                    actual="(2,4)",
                    path="layer.weight",
                ),
            ),
        )
        assert r.status == "incompatible"


def test_safe_state_decode_error_importable() -> None:
    assert issubclass(SafeStateDecodeError, Exception)
