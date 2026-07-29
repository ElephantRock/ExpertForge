"""Tests for dotted-path --set overrides (Issue #5 decision §4).

Semantics:
- split at the first '=';
- require an existing leaf path (reject unknown + non-leaf paths);
- reject duplicate overrides (not last-write-wins);
- parse valid JSON values as JSON, otherwise retain a literal string;
- record both the raw token and the normalized typed value.
"""

from __future__ import annotations

import pytest

from expertforge.config.overrides import (
    OverrideError,
    parse_overrides,
)


class TestParseOverridesBasic:
    def test_single_leaf_override_parsed(self) -> None:
        recs = parse_overrides(["training.seed=7"])
        assert len(recs) == 1
        assert recs[0].path == "training.seed"
        assert recs[0].raw_token == "training.seed=7"

    def test_json_int_parsed_as_int(self) -> None:
        # The decision says: parse valid JSON values as JSON. A bare integer
        # token is valid JSON (a number), so the normalized value is int 7.
        recs = parse_overrides(["training.seed=7"])
        assert recs[0].value == 7
        assert isinstance(recs[0].value, int)

    def test_json_float_parsed_as_float(self) -> None:
        recs = parse_overrides(["training.lr=0.0003"])
        assert recs[0].value == 0.0003
        assert isinstance(recs[0].value, float)

    def test_json_bool_parsed_as_bool(self) -> None:
        recs = parse_overrides(['logging.level="DEBUG"'])
        assert recs[0].value == "DEBUG"

    def test_literal_string_when_not_json(self) -> None:
        recs = parse_overrides(["run.name=smoke-a"])
        assert recs[0].value == "smoke-a"
        assert isinstance(recs[0].value, str)

    def test_value_with_equals_sign(self) -> None:
        # Split at FIRST '=' only.
        recs = parse_overrides(["run.name=a=b"])
        assert recs[0].path == "run.name"
        assert recs[0].value == "a=b"


class TestParseOverridesErrors:
    def test_missing_equals_rejected(self) -> None:
        with pytest.raises(OverrideError):
            parse_overrides(["training.seed"])

    def test_empty_path_rejected(self) -> None:
        with pytest.raises(OverrideError):
            parse_overrides(["=7"])

    def test_duplicate_path_rejected(self) -> None:
        with pytest.raises(OverrideError):
            parse_overrides(["training.seed=7", "training.seed=8"])


class TestOverrideRecord:
    def test_record_carries_raw_and_typed(self) -> None:
        rec = parse_overrides(["training.seed=42"])[0]
        assert rec.raw_token == "training.seed=42"
        assert rec.path == "training.seed"
        assert rec.typed_repr == "42"  # normalized typed value serialized

    def test_json_null_parsed_as_none(self) -> None:
        rec = parse_overrides(["training.seq_len=null"])[0]
        assert rec.value is None


class TestEmptyStringValueAccepted:
    """Regression: an empty override value is not valid JSON, so per the
    JSON-or-literal rule it must become the literal empty string. Only an
    empty PATH remains rejected."""

    def test_empty_value_becomes_empty_string(self) -> None:
        rec = parse_overrides(["run.description="])[0]
        assert rec.path == "run.description"
        assert rec.value == ""
        assert rec.value is not None  # not absent / not rejected

    def test_empty_path_still_rejected(self) -> None:
        with pytest.raises(OverrideError):
            parse_overrides(["=7"])


class TestNonFiniteValuesRejected:
    """Regression: JSON parses NaN/Infinity, and Pydantic may accept +inf for a
    float field. Such values must be rejected at parse time, before they can
    reach the configuration and fail canonicalization."""

    def test_infinity_token_rejected(self) -> None:
        with pytest.raises(OverrideError):
            parse_overrides(["training.lr=Infinity"])

    def test_nan_token_rejected(self) -> None:
        with pytest.raises(OverrideError):
            parse_overrides(["training.lr=NaN"])

    def test_negative_infinity_rejected(self) -> None:
        with pytest.raises(OverrideError):
            parse_overrides(["training.lr=-Infinity"])
