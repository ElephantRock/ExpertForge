"""Tests for the restricted YAML loader (Issue #5 decision §1).

YAML is the authoring format only. The loader must enforce a strict subset:
reject duplicate keys, custom tags, anchors/aliases, merge keys, multiple
documents; require a single mapping at the root; use SafeLoader semantics.
"""

from __future__ import annotations

import pytest

from expertforge.config.yaml_loader import (
    RestrictedYAMLError,
    load_restricted_yaml,
)


class TestValidYAML:
    def test_simple_mapping_loads(self) -> None:
        result = load_restricted_yaml("a: 1\nb: hello\n")
        assert result == {"a": 1, "b": "hello"}

    def test_nested_mapping_loads(self) -> None:
        result = load_restricted_yaml("outer:\n  inner: 5\n")
        assert result == {"outer": {"inner": 5}}

    def test_empty_string_rejected_not_silently_none(self) -> None:
        # An empty document has no root mapping; must fail, not return None.
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("")


class TestRootMustBeMapping:
    def test_root_sequence_rejected(self) -> None:
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("- a\n- b\n")

    def test_root_scalar_rejected(self) -> None:
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("just a scalar\n")

    def test_root_null_rejected(self) -> None:
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("null\n")


class TestDuplicateKeysRejected:
    def test_duplicate_top_level_keys_rejected(self) -> None:
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("a: 1\na: 2\n")

    def test_duplicate_nested_keys_rejected(self) -> None:
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("outer:\n  x: 1\n  x: 2\n")


class TestCustomTagsRejected:
    def test_explicit_python_tag_rejected(self) -> None:
        # !!python/object would be a code-execution vector; must be refused.
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("x: !!python/object/apply:os.system ['echo hi']\n")

    def test_arbitrary_local_tag_rejected(self) -> None:
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("x: !custom value\n")


class TestAnchorsAliasesMergeRejected:
    def test_anchor_rejected(self) -> None:
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("a: &anchor 1\nb: *anchor\n")

    def test_alias_rejected(self) -> None:
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("base: &b\n  k: 1\nderived: *b\n")

    def test_merge_key_rejected(self) -> None:
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("base: &b\n  k: 1\nchild:\n  <<: *b\n  j: 2\n")


class TestMultipleDocumentsRejected:
    def test_two_documents_rejected(self) -> None:
        with pytest.raises(RestrictedYAMLError):
            load_restricted_yaml("a: 1\n---\nb: 2\n")
