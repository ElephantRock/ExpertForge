"""Tests for the specification fingerprint (Issue #6 decision §1).

A versioned deterministic envelope is canonicalized as compact sorted-key UTF-8
JSON, then SHA-256'd, yielding ``spec-v1-sha256-<64 hex>``. Immutable inputs are
frozen, uniquely named, sorted by name, limited to stable names/algorithms/
digests. v1 uses an empty immutable-input tuple.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from expertforge.config.resolve import canonical_bytes, resolve_config
from expertforge.identity.fingerprint import (
    FINGERPRINT_VERSION,
    ImmutableInput,
    specification_fingerprint,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# --- envelope construction -------------------------------------------------


class TestFingerprintEnvelope:
    def test_fingerprint_format_is_spec_v1_sha256_hex(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        assert fp.digest_str.startswith("spec-v1-sha256-")
        # 64 lowercase hex after the prefix.
        hexpart = fp.digest_str.removeprefix("spec-v1-sha256-")
        assert len(hexpart) == 64
        int(hexpart, 16)  # valid hex
        assert hexpart == hexpart.lower()

    def test_envelope_carries_schema_version_canonical_digest(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        assert fp.envelope["schema"] == "expertforge.specification-fingerprint"
        assert fp.envelope["version"] == FINGERPRINT_VERSION
        assert fp.envelope["canonical_config"]["algorithm"] == "sha256"
        assert fp.envelope["canonical_config"]["digest"] == fp.canonical_config_digest

    def test_immutable_inputs_default_empty(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        assert fp.envelope["immutable_inputs"] == []


# --- stability & sensitivity ----------------------------------------------


class TestFingerprintStability:
    def test_identical_canonical_bytes_same_fingerprint(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        cb = canonical_bytes(env)
        a = specification_fingerprint(cb)
        b = specification_fingerprint(cb)
        assert a.digest_str == b.digest_str

    def test_repeated_resolution_same_fingerprint(self) -> None:
        # Resolving the same fixture repeatedly must yield the same fingerprint.
        fps = {
            specification_fingerprint(
                canonical_bytes(resolve_config(CONFIGS / "smoke.yaml"))
            ).digest_str
            for _ in range(5)
        }
        assert len(fps) == 1

    def test_meaningful_config_change_yields_different_fingerprint(self, tmp_path: Path) -> None:
        base = resolve_config(CONFIGS / "smoke.yaml")
        # A different effective config (override changes training.seed) -> different
        # canonical bytes -> different fingerprint.
        overridden = resolve_config(CONFIGS / "smoke.yaml", ["training.seed=999"])
        a = specification_fingerprint(canonical_bytes(base))
        b = specification_fingerprint(canonical_bytes(overridden))
        assert a.digest_str != b.digest_str

    def test_override_to_same_value_same_fingerprint(self, tmp_path: Path) -> None:
        # Overriding to the value the config already has -> same effective config.
        env = resolve_config(CONFIGS / "smoke.yaml")
        base_fp = specification_fingerprint(canonical_bytes(env))
        same = resolve_config(CONFIGS / "smoke.yaml", ["training.seed=7"])
        same_fp = specification_fingerprint(canonical_bytes(same))
        assert base_fp.digest_str == same_fp.digest_str


# --- immutable inputs ------------------------------------------------------


class TestImmutableInputs:
    def test_immutable_inputs_sorted_by_name(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        inputs = [
            ImmutableInput(name="dataset", algorithm="sha256", digest="b" * 64),
            ImmutableInput(name="tokenizer", algorithm="sha256", digest="a" * 64),
        ]
        fp = specification_fingerprint(canonical_bytes(env), immutable_inputs=inputs)
        # Sorted by name: dataset, tokenizer (already sorted), but enforce the
        # envelope serialization is name-sorted regardless of input order.
        names = [ii["name"] for ii in fp.envelope["immutable_inputs"]]
        assert names == sorted(names)

    def test_immutable_inputs_unsorted_input_still_sorted_in_envelope(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        inputs = [
            ImmutableInput(name="zzz", algorithm="sha256", digest="1" * 64),
            ImmutableInput(name="aaa", algorithm="sha256", digest="2" * 64),
        ]
        fp = specification_fingerprint(canonical_bytes(env), immutable_inputs=inputs)
        names = [ii["name"] for ii in fp.envelope["immutable_inputs"]]
        assert names == ["aaa", "zzz"]

    def test_duplicate_immutable_input_names_rejected(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        inputs = [
            ImmutableInput(name="dataset", algorithm="sha256", digest="a" * 64),
            ImmutableInput(name="dataset", algorithm="sha256", digest="b" * 64),
        ]
        with pytest.raises(ValueError):
            specification_fingerprint(canonical_bytes(env), immutable_inputs=inputs)

    def test_immutable_input_changes_fingerprint(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        without = specification_fingerprint(canonical_bytes(env))
        with_input = specification_fingerprint(
            canonical_bytes(env),
            immutable_inputs=[ImmutableInput(name="dataset", algorithm="sha256", digest="c" * 64)],
        )
        assert without.digest_str != with_input.digest_str

    def test_immutable_input_is_frozen(self) -> None:
        ii = ImmutableInput(name="x", algorithm="sha256", digest="d" * 64)
        with pytest.raises(ValidationError):
            ii.name = "y"  # type: ignore[misc]


# --- determinism of envelope canonicalization -----------------------------


class TestEnvelopeDeterminism:
    def test_envelope_canonical_form_is_compact_sorted(self) -> None:
        env = resolve_config(CONFIGS / "smoke.yaml")
        fp = specification_fingerprint(canonical_bytes(env))
        b = fp.envelope_canonical_bytes
        text = b.decode("utf-8")
        # Compact: no insignificant whitespace.
        assert ", " not in text
        assert ": " not in text
