"""Tests for configuration resolution: envelope, apply path-validation,
and behavioral canonical bytes (Issue #5 decision §4/§canonical).

Two logically separate representations:

- Resolution envelope: source path, content hash, raw + normalized overrides,
  fully resolved configuration.
- Behavioral canonical bytes: effective config only, deterministic UTF-8 JSON,
  sorted keys, compact separators, no non-finite numbers.

Two different source files or override sequences that produce the same effective
configuration must produce identical canonical bytes.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from expertforge.config.resolve import (
    ConfigResolutionError,
    ResolutionEnvelope,
    apply_overrides_to_mapping,
    canonical_bytes,
    resolve_config,
)

# --- fixtures --------------------------------------------------------------


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


MINIMAL_YAML = """\
run:
  name: smoke-a
model:
  dim: 64
  n_layers: 2
  n_heads: 2
  ffn_dim: 128
training:
  seed: 7
  tokens: 1024
  batch_size: 4
  lr: 0.0001
"""


# --- resolution envelope ---------------------------------------------------


class TestResolutionEnvelope:
    def test_resolve_returns_envelope(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "c.yaml", MINIMAL_YAML)
        env = resolve_config(src)
        assert isinstance(env, ResolutionEnvelope)
        assert env.source_path == src.resolve()
        assert env.content_hash  # sha256 hex
        assert env.overrides == ()
        assert env.config.run.name == "smoke-a"

    def test_content_hash_is_sha256_hex_64(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "c.yaml", MINIMAL_YAML)
        env = resolve_config(src)
        assert len(env.content_hash) == 64
        int(env.content_hash, 16)  # valid hex

    def test_identical_content_same_hash(self, tmp_path: Path) -> None:
        a = _write(tmp_path, "a.yaml", MINIMAL_YAML)
        b = _write(tmp_path, "b.yaml", MINIMAL_YAML)
        assert resolve_config(a).content_hash == resolve_config(b).content_hash

    def test_envelope_is_immutable(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "c.yaml", MINIMAL_YAML)
        env = resolve_config(src)
        # Frozen pydantic models raise ValidationError on attribute assignment.
        with pytest.raises(ValidationError):
            env.config.run.name = "x"  # type: ignore[misc]


# --- apply overrides: path validation --------------------------------------


class TestApplyOverridesPathValidation:
    def test_known_leaf_applies(self) -> None:
        mapping = {"run": {"name": "a"}, "training": {"seed": 1}}
        out = apply_overrides_to_mapping(mapping, ["training.seed=2"])
        assert out["training"]["seed"] == 2

    def test_unknown_path_rejected(self) -> None:
        mapping = {"run": {"name": "a"}}
        with pytest.raises(ConfigResolutionError):
            apply_overrides_to_mapping(mapping, ["run.nope=1"])

    def test_non_leaf_path_rejected(self) -> None:
        # 'run' is a section (non-leaf); cannot overwrite a whole section.
        mapping = {"run": {"name": "a"}}
        with pytest.raises(ConfigResolutionError):
            apply_overrides_to_mapping(mapping, ["run=5"])

    def test_unknown_section_rejected(self) -> None:
        mapping = {"run": {"name": "a"}}
        with pytest.raises(ConfigResolutionError):
            apply_overrides_to_mapping(mapping, ["bogus.x=1"])

    def test_override_applied_before_validation(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "c.yaml", MINIMAL_YAML)
        env = resolve_config(src, ["training.seed=99"])
        assert env.config.training.seed == 99
        assert env.overrides[0].raw_token == "training.seed=99"


# --- behavioral canonical bytes -------------------------------------------


class TestCanonicalBytes:
    def test_deterministic_across_dict_order(self, tmp_path: Path) -> None:
        # Same effective config from two differently-ordered YAML files.
        ordered = """\
run:
  name: x
model:
  dim: 64
  n_layers: 2
  n_heads: 2
  ffn_dim: 128
training:
  seed: 1
  tokens: 1024
  batch_size: 1
  lr: 0.001
"""
        reordered = """\
training:
  lr: 0.001
  batch_size: 1
  tokens: 1024
  seed: 1
model:
  ffn_dim: 128
  n_heads: 2
  n_layers: 2
  dim: 64
run:
  name: x
"""
        a = _write(tmp_path, "a.yaml", ordered)
        b = _write(tmp_path, "b.yaml", reordered)
        assert canonical_bytes(resolve_config(a)) == canonical_bytes(resolve_config(b))

    def test_override_to_same_value_same_bytes(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "c.yaml", MINIMAL_YAML)
        # seed is already 7; overriding to 7 must yield identical canonical bytes.
        base = resolve_config(src)
        overridden = resolve_config(src, ["training.seed=7"])
        assert canonical_bytes(base) == canonical_bytes(overridden)

    def test_bytes_are_compact_sorted_utf8(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "c.yaml", MINIMAL_YAML)
        b = canonical_bytes(resolve_config(src))
        text = b.decode("utf-8")
        # No insignificant whitespace (compact separators).
        assert ", " not in text
        assert ": " not in text
        # Keys are sorted at every level (model < run alphabetically at root).
        i_run = text.find('"run"')
        i_model = text.find('"model"')
        assert i_model < i_run

    def test_repeated_resolution_identical(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "c.yaml", MINIMAL_YAML)
        first = canonical_bytes(resolve_config(src))
        second = canonical_bytes(resolve_config(src))
        assert first == second


# --- validation surfaces actionable errors --------------------------------


class TestActionableValidationErrors:
    def test_invalid_config_reports_useful_error(self, tmp_path: Path) -> None:
        bad = MINIMAL_YAML.replace("dim: 64", "dim: notanumber")
        src = _write(tmp_path, "bad.yaml", bad)
        with pytest.raises(ConfigResolutionError) as exc:
            resolve_config(src)
        # The error must mention the offending field.
        assert "dim" in str(exc.value)


class TestOverrideOnDefaultedField:
    """Regression: an override must reach a field whose SECTION is omitted from
    YAML but supplied by schema defaults. The override surface is defined by the
    schema, not by which defaults the author wrote into YAML."""

    OMITTED_SECTION_YAML = """\
run:
  name: smoke-a
model:
  dim: 64
  n_layers: 2
  n_heads: 2
  ffn_dim: 128
training:
  seed: 7
  tokens: 1024
  batch_size: 4
  lr: 0.0001
"""

    def test_override_into_omitted_section(self, tmp_path: Path) -> None:
        # hardware section is entirely omitted; override must still apply.
        src = _write(tmp_path, "c.yaml", self.OMITTED_SECTION_YAML)
        env = resolve_config(src, ["hardware.device=cpu"])
        assert env.config.hardware.device == "cpu"

    def test_override_into_omitted_nested_section(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "c.yaml", self.OMITTED_SECTION_YAML)
        env = resolve_config(src, ["logging.level=DEBUG", "logging.log_interval_steps=5"])
        assert env.config.logging.level == "DEBUG"
        assert env.config.logging.log_interval_steps == 5

    def test_unknown_section_still_rejected(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "c.yaml", self.OMITTED_SECTION_YAML)
        with pytest.raises(ConfigResolutionError):
            resolve_config(src, ["bogus.field=1"])


class TestRawByteSourceHash:
    """Regression: the source hash must cover the original file BYTES, so that
    LF-vs-CRLF variants of the same logical content hash distinctly. read_text()
    normalized CRLF before hashing in the prior implementation."""

    def test_lf_and_cRLF_variants_have_different_source_hashes(self, tmp_path: Path) -> None:
        # Write both variants as explicit bytes so the test does not depend on
        # the platform's text-mode newline translation.
        lf_path = tmp_path / "lf.yaml"
        lf_path.write_bytes(MINIMAL_YAML.encode("utf-8"))
        crlf_path = tmp_path / "crlf.yaml"
        crlf_path.write_bytes(MINIMAL_YAML.replace("\n", "\r\n").encode("utf-8"))
        env_lf = resolve_config(lf_path)
        env_crlf = resolve_config(crlf_path)
        assert env_lf.content_hash != env_crlf.content_hash

    def test_lf_and_crlf_produce_identical_canonical_bytes(self, tmp_path: Path) -> None:
        # Same logical config -> same behavioral canonical bytes even though the
        # raw source bytes differ (provenance vs behavior split).
        lf_path = tmp_path / "lf.yaml"
        lf_path.write_bytes(MINIMAL_YAML.encode("utf-8"))
        crlf_path = tmp_path / "crlf.yaml"
        crlf_path.write_bytes(MINIMAL_YAML.replace("\n", "\r\n").encode("utf-8"))
        assert canonical_bytes(resolve_config(lf_path)) == canonical_bytes(
            resolve_config(crlf_path)
        )


class TestEnvelopeDeepImmutability:
    """Regression: a frozen dataclass holding a mutable list is not deeply
    immutable. The envelope's overrides must not be appendable/clearable."""

    def test_overrides_attribute_is_not_a_mutable_list(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "c.yaml", MINIMAL_YAML)
        env = resolve_config(src, ["training.seed=99"])
        # Tuples raise on append/clear; lists do not.
        assert not hasattr(env.overrides, "append"), "overrides must be a tuple, not a list"

    def test_overrides_record_itself_is_immutable(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "c.yaml", MINIMAL_YAML)
        env = resolve_config(src, ["training.seed=99"])
        rec = env.overrides[0]
        # Frozen dataclass attribute assignment raises FrozenInstanceError.
        with pytest.raises(FrozenInstanceError):
            rec.path = "other"  # type: ignore[misc]


class TestCanonicalBytesSerializationBoundary:
    """Regression (PR #16 re-review): canonical_bytes must convert any
    serialization failure (e.g. a Unicode surrogate that slips past validation,
    or any value UTF-8/JSON cannot encode) into a ConfigResolutionError, never a
    raw UnicodeEncodeError/PydanticSerializationError traceback.

    Pydantic strict string validation currently rejects YAML ``\\uD800`` escapes
    at the model layer, so a surrogate cannot reach canonical_bytes through the
    normal path today. This test exercises the boundary directly to guarantee it
    holds regardless of future field/config changes."""

    def test_canonical_bytes_wraps_encoding_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _write(tmp_path, "c.yaml", MINIMAL_YAML)
        env = resolve_config(src)
        # Force the config to dump a value UTF-8 cannot encode (an unpaired
        # Unicode surrogate). This simulates a non-serializable value reaching
        # canonicalization if a future field accepted one.
        monkeypatch.setattr(
            type(env.config),
            "model_dump",
            lambda self, **kw: {"run": {"name": "\ud800"}},  # noqa: ARG005
            raising=False,
        )
        with pytest.raises(ConfigResolutionError):
            canonical_bytes(env)
