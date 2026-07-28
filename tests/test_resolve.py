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
        assert env.overrides == []
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
