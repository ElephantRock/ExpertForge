"""Tests for the specification fingerprint (Issue #6 decision §1, review §2–§4).

A versioned deterministic envelope is canonicalized as compact sorted-key UTF-8
JSON, then SHA-256'd, yielding ``spec-v1-sha256-<64 hex>``. The fingerprint is a
deeply immutable nested Pydantic record (envelope + canonical-config digest +
immutable inputs + public digest) that round-trips through the sidecar.
Immutable inputs are validated: stable lowercase names, ``sha256`` only for v1,
64 lowercase-hex digests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from expertforge.config.resolve import canonical_bytes, resolve_config
from expertforge.identity.fingerprint import (
    FINGERPRINT_VERSION,
    ImmutableInput,
    SpecificationFingerprintRecord,
    specification_fingerprint,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _valid_input(name: str = "dataset", digest: str = "a" * 64) -> ImmutableInput:
    return ImmutableInput(name=name, algorithm="sha256", digest=digest)


# --- envelope construction -------------------------------------------------


class TestFingerprintEnvelope:
    def test_fingerprint_format_is_spec_v1_sha256_hex(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        assert fp.digest_str.startswith("spec-v1-sha256-")
        hexpart = fp.digest_str.removeprefix("spec-v1-sha256-")
        assert len(hexpart) == 64
        int(hexpart, 16)
        assert hexpart == hexpart.lower()

    def test_envelope_carries_schema_version_canonical_digest(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        assert fp.schema_name == "expertforge.specification-fingerprint"
        assert fp.version == FINGERPRINT_VERSION
        assert fp.canonical_config.algorithm == "sha256"
        assert fp.canonical_config.digest

    def test_immutable_inputs_default_empty_tuple(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        assert fp.immutable_inputs == ()


# --- stability & sensitivity ----------------------------------------------


class TestFingerprintStability:
    def test_identical_canonical_bytes_same_fingerprint(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        cb = canonical_bytes(env)
        assert specification_fingerprint(cb).digest_str == specification_fingerprint(cb).digest_str

    def test_repeated_resolution_same_fingerprint(self) -> None:
        fps = {
            specification_fingerprint(
                canonical_bytes(resolve_config(CONFIGS / "smoke.yaml"))
            ).digest_str
            for _ in range(5)
        }
        assert len(fps) == 1

    def test_meaningful_config_change_yields_different_fingerprint(self) -> None:
        base = resolve_config(CONFIGS / "smoke.yaml")
        overridden = resolve_config(CONFIGS / "smoke.yaml", ["training.seed=999"])
        assert (
            specification_fingerprint(canonical_bytes(base)).digest_str
            != specification_fingerprint(canonical_bytes(overridden)).digest_str
        )

    def test_override_to_same_value_same_fingerprint(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        base_fp = specification_fingerprint(canonical_bytes(env)).digest_str
        same = resolve_config(CONFIGS / "smoke.yaml", ["training.seed=7"])
        same_fp = specification_fingerprint(canonical_bytes(same)).digest_str
        assert base_fp == same_fp


# --- immutable inputs ------------------------------------------------------


class TestImmutableInputs:
    def test_immutable_inputs_sorted_by_name(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(
            canonical_bytes(env),
            immutable_inputs=[_valid_input("zzz", "1" * 64), _valid_input("aaa", "2" * 64)],
        )
        names = [ii.name for ii in fp.immutable_inputs]
        assert names == ["aaa", "zzz"]

    def test_duplicate_immutable_input_names_rejected(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        with pytest.raises(ValueError):
            specification_fingerprint(
                canonical_bytes(env),
                immutable_inputs=[
                    _valid_input("dataset", "a" * 64),
                    _valid_input("dataset", "b" * 64),
                ],
            )

    def test_immutable_input_changes_fingerprint(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        without = specification_fingerprint(canonical_bytes(env))
        with_input = specification_fingerprint(
            canonical_bytes(env), immutable_inputs=[_valid_input("dataset", "c" * 64)]
        )
        assert without.digest_str != with_input.digest_str

    def test_immutable_input_is_frozen(self) -> None:
        ii = _valid_input()
        with pytest.raises(ValidationError):
            ii.name = "y"  # type: ignore[misc]


# --- immutable-input validation (review item 3) ---------------------------


class TestImmutableInputValidation:
    @pytest.mark.parametrize(
        "bad_name", ["Dataset", "data/set", "data\\set", ".hidden", "-leading", "", "has space"]
    )
    def test_invalid_name_rejected(self, bad_name: str) -> None:
        with pytest.raises(ValidationError):
            ImmutableInput(name=bad_name, algorithm="sha256", digest="a" * 64)

    def test_unsupported_algorithm_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ImmutableInput(name="dataset", algorithm="md5", digest="a" * 64)

    @pytest.mark.parametrize("bad_digest", ["A" * 64, "g" * 64, "a" * 63, "a" * 65, "", "xyz"])
    def test_invalid_digest_rejected(self, bad_digest: str) -> None:
        with pytest.raises(ValidationError):
            ImmutableInput(name="dataset", algorithm="sha256", digest=bad_digest)

    def test_supported_name_shapes_accepted(self) -> None:
        for ok in ["dataset", "tokenizer.bpe", "source.git", "a1-2.3"]:
            ImmutableInput(name=ok, algorithm="sha256", digest="d" * 64)


# --- deep immutability + round-trip (review item 2) -----------------------


class TestTrailingNewlineRejection:
    """Regression: validators must use .fullmatch() so trailing newlines are
    rejected (Python's `$` matches before a final newline under .match())."""

    def test_fingerprint_id_newline_rejected(self) -> None:
        from expertforge.identity.fingerprint import validate_fingerprint_id

        good = "spec-v1-sha256-" + "a" * 64
        validate_fingerprint_id(good)  # base accepts
        with pytest.raises(ValueError):
            validate_fingerprint_id(good + "\n")

    def test_immutable_input_name_newline_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ImmutableInput(name="dataset\n", algorithm="sha256", digest="a" * 64)

    def test_immutable_input_digest_newline_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ImmutableInput(name="dataset", algorithm="sha256", digest="a" * 64 + "\n")


class TestFingerprintRecordImmutability:
    def test_record_is_frozen(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        with pytest.raises(ValidationError):
            fp.digest_str = "x"  # type: ignore[misc]

    def test_immutable_inputs_is_a_tuple_not_list(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(
            canonical_bytes(env), immutable_inputs=[_valid_input("dataset", "a" * 64)]
        )
        assert isinstance(fp.immutable_inputs, tuple)
        assert not hasattr(fp.immutable_inputs, "append")

    def test_record_round_trips_through_json(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(
            canonical_bytes(env), immutable_inputs=[_valid_input("dataset", "a" * 64)]
        )
        dumped = fp.model_dump_json()
        restored = SpecificationFingerprintRecord.model_validate_json(dumped)
        assert restored == fp
        # The restored record's digest must still verify against its envelope.
        restored.verify_digest()

    def test_unknown_fingerprint_version_rejected_on_construction(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        bad = fp.model_dump()
        bad["version"] = 999
        with pytest.raises(ValidationError):
            SpecificationFingerprintRecord.model_validate(bad)

    def test_digest_mismatch_detected_on_verify(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        bad = fp.model_copy(update={"digest_str": f"spec-v1-sha256-{'0' * 64}"})
        with pytest.raises(ValueError):
            bad.verify_digest()

    def test_wrong_schema_name_rejected_on_construction(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        bad = fp.model_dump(by_alias=True)
        bad["schema"] = "wrong-schema-name"
        with pytest.raises(ValidationError):
            SpecificationFingerprintRecord.model_validate(bad)

    def test_unsorted_immutable_inputs_rejected_on_construction(self) -> None:
        # Build a canonical fp then tamper the order on load.
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(
            canonical_bytes(env),
            immutable_inputs=[_valid_input("aaa", "1" * 64), _valid_input("zzz", "2" * 64)],
        )
        bad = fp.model_dump(by_alias=True)
        # Reverse the order — the digest becomes inconsistent too, but the
        # sortedness check fires first.
        bad["immutable_inputs"] = list(reversed(bad["immutable_inputs"]))
        with pytest.raises(ValidationError):
            SpecificationFingerprintRecord.model_validate(bad)

    def test_duplicate_immutable_inputs_rejected_on_construction(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        bad = fp.model_dump(by_alias=True)
        bad["immutable_inputs"] = [
            {"name": "dup", "algorithm": "sha256", "digest": "a" * 64},
            {"name": "dup", "algorithm": "sha256", "digest": "b" * 64},
        ]
        with pytest.raises(ValidationError):
            SpecificationFingerprintRecord.model_validate(bad)

    def test_inconsistent_digest_rejected_on_construction(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        bad = fp.model_dump(by_alias=True)
        # Tamper the digest_str so it no longer matches the recomputed envelope.
        bad["digest_str"] = f"spec-v1-sha256-{'0' * 64}"
        with pytest.raises(ValidationError):
            SpecificationFingerprintRecord.model_validate(bad)

    def test_serialization_uses_schema_key_not_schema_name(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        d = fp.model_dump(mode="json")
        assert "schema" in d
        assert "schema_name" not in d

    def test_sidecar_parent_path_serializes_schema_key(self) -> None:
        # Regression: the nested serializer must emit `schema` even when the
        # fingerprint is serialized via the parent AttemptIdentityRecord's
        # model_dump_json (which does not call the child's overridden dump).
        from expertforge.identity.record import AttemptIdentityRecord

        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        rec = AttemptIdentityRecord(
            specification_fingerprint=fp,
            run_id="run-20260101t000000z-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbb",
            attempt_id="attempt-20260101t000000z-cccccccccccccccccccc",
            created_at_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        import json as _json

        d = _json.loads(rec.to_deterministic_json())
        fp_obj = d["specification_fingerprint"]
        assert "schema" in fp_obj
        assert fp_obj["schema"] == "expertforge.specification-fingerprint"
        assert "schema_name" not in fp_obj


# --- determinism of envelope canonicalization -----------------------------


class TestEnvelopeDeterminism:
    def test_envelope_canonical_form_is_compact_sorted(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        text = fp.envelope_canonical_bytes().decode("utf-8")
        assert ", " not in text
        assert ": " not in text


# --- recomputation verifier (review item 4) -------------------------------


class TestRecomputationVerifier:
    def test_verify_accepts_matching_config_and_inputs(self) -> None:
        from expertforge.identity.fingerprint import verify_fingerprint

        env = resolve_config(CONFIGS / "smoke.yaml")
        cb = canonical_bytes(env)
        fp = specification_fingerprint(cb, immutable_inputs=[_valid_input("dataset", "a" * 64)])
        # Recomputing from the same canonical bytes + same inputs verifies.
        verify_fingerprint(
            fp, cb, [ImmutableInput(name="dataset", algorithm="sha256", digest="a" * 64)]
        )

    def test_verify_detects_config_mismatch(self) -> None:
        from expertforge.identity.fingerprint import FingerprintMismatch, verify_fingerprint

        env = resolve_config(CONFIGS / "smoke.yaml")
        cb = canonical_bytes(env)
        fp = specification_fingerprint(cb)
        # Different canonical bytes (different effective config) -> mismatch.
        other = canonical_bytes(resolve_config(CONFIGS / "smoke.yaml", ["training.seed=999"]))
        with pytest.raises(FingerprintMismatch):
            verify_fingerprint(fp, other, [])

    def test_verify_detects_immutable_input_mismatch(self) -> None:
        from expertforge.identity.fingerprint import FingerprintMismatch, verify_fingerprint

        env = resolve_config(CONFIGS / "smoke.yaml")
        cb = canonical_bytes(env)
        fp = specification_fingerprint(cb, immutable_inputs=[_valid_input("dataset", "a" * 64)])
        # Same config, different immutable input -> mismatch.
        with pytest.raises(FingerprintMismatch):
            verify_fingerprint(fp, cb, [_valid_input("dataset", "b" * 64)])

    def test_verify_detects_missing_immutable_input(self) -> None:
        from expertforge.identity.fingerprint import FingerprintMismatch, verify_fingerprint

        env = resolve_config(CONFIGS / "smoke.yaml")
        cb = canonical_bytes(env)
        fp = specification_fingerprint(cb, immutable_inputs=[_valid_input("dataset", "a" * 64)])
        # Declared input omitted at verify time -> mismatch.
        with pytest.raises(FingerprintMismatch):
            verify_fingerprint(fp, cb, [])

    def test_verify_detects_additional_immutable_input(self) -> None:
        from expertforge.identity.fingerprint import FingerprintMismatch, verify_fingerprint

        env = resolve_config(CONFIGS / "smoke.yaml")
        cb = canonical_bytes(env)
        fp = specification_fingerprint(cb)
        # No declared inputs, but verifier supplies one -> mismatch.
        with pytest.raises(FingerprintMismatch):
            verify_fingerprint(fp, cb, [_valid_input("dataset", "a" * 64)])

    def test_verify_detects_version_mismatch(self) -> None:
        from expertforge.identity.fingerprint import FingerprintMismatch, verify_fingerprint

        env = resolve_config(CONFIGS / "smoke.yaml")
        cb = canonical_bytes(env)
        fp = specification_fingerprint(cb)
        # Tamper with the stored envelope version.
        bad = fp.model_copy(update={"version": 999})
        with pytest.raises(FingerprintMismatch):
            verify_fingerprint(bad, cb, [])
