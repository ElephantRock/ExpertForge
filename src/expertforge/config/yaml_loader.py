"""Restricted YAML loader for ExpertForge configuration files (Issue #5 §1).

YAML is the authoring format only — not the canonical serialized representation.
This loader enforces a strict, predictable subset:

- SafeLoader construction semantics (no arbitrary Python objects);
- a single document, whose root is a mapping;
- no duplicate keys (at any depth);
- no custom tags (only the standard safe scalar/collection resolvers);
- no anchors, aliases, or ``<<`` merge keys.

It rejects includes, environment interpolation, and executable expressions by
construction (they are simply not part of the accepted grammar).
"""

from __future__ import annotations

from typing import Any

import yaml

__all__ = ["RestrictedYAMLError", "load_restricted_yaml"]


class RestrictedYAMLError(Exception):
    """Raised when a YAML document violates the accepted restricted subset."""


# Tags permitted by the safe resolver set. Anything outside this set is rejected
# as a custom/unsupported tag.
_ALLOWED_TAGS = {
    yaml.resolver.Resolver.DEFAULT_SCALAR_TAG,
    yaml.resolver.Resolver.DEFAULT_SEQUENCE_TAG,
    yaml.resolver.Resolver.DEFAULT_MAPPING_TAG,
    "tag:yaml.org,2002:str",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:null",
    "tag:yaml.org,2002:timestamp",
}


class _RestrictedLoader(yaml.SafeLoader):  # noqa: S101 - internal helper class
    """SafeLoader subclass enforcing ExpertForge's restricted YAML subset."""

    # --- reject anchors / aliases ------------------------------------------
    def compose_node(self, parent: Any, index: Any) -> yaml.Node:
        # Aliases ('*name') surface as AliasEvent and are handled by the base
        # composer by returning the previously-stored node. Block them first.
        if self.check_event(yaml.AliasEvent):  # type: ignore[no-untyped-call]
            raise RestrictedYAMLError(
                "YAML aliases (references via '*') are not permitted in ExpertForge configuration."
            )
        node = super().compose_node(parent, index)
        # The base composer returns None only on end-of-stream; we are mid-tree,
        # so a node is expected. Guard for type safety.
        if node is None:  # pragma: no cover - defensive, not reachable mid-tree
            raise RestrictedYAMLError("Unexpected end of YAML node stream.")
        # Anchors ('&name') are recorded on the produced node.
        if getattr(node, "anchor", None) is not None:
            raise RestrictedYAMLError(
                "YAML anchors (definitions via '&') are not permitted in ExpertForge configuration."
            )
        return node

    # --- reject custom tags ------------------------------------------------
    def construct_object(self, node: yaml.Node, deep: bool = False) -> Any:
        if node.tag not in _ALLOWED_TAGS:
            raise RestrictedYAMLError(
                f"Custom or unsupported YAML tag {node.tag!r} is not permitted "
                "in ExpertForge configuration."
            )
        return super().construct_object(node, deep=deep)

    # --- reject duplicate keys + merge keys --------------------------------
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _value_node in node.value:
            # Reject the merge key '<<' outright.
            if isinstance(key_node, yaml.ScalarNode) and key_node.value == "<<":
                raise RestrictedYAMLError(
                    "YAML merge keys ('<<') are not permitted in ExpertForge configuration."
                )
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise RestrictedYAMLError(
                    f"Duplicate YAML key {key!r} is not permitted in ExpertForge configuration."
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_restricted_yaml(text: str) -> dict[str, Any]:
    """Parse ``text`` as restricted YAML, returning the root mapping.

    Raises :class:`RestrictedYAMLError` for any disallowed construct or when the
    document is empty, multi-document, or does not have a mapping at its root.
    """
    # Multi-document rejection: require exactly one document.
    documents = list(yaml.load_all(text, Loader=_RestrictedLoader))
    if len(documents) == 0:
        raise RestrictedYAMLError("Empty YAML document; a mapping root is required.")
    if len(documents) > 1:
        raise RestrictedYAMLError(
            "Multiple YAML documents (use of '---') are not permitted in ExpertForge configuration."
        )
    result = documents[0]
    if not isinstance(result, dict):
        raise RestrictedYAMLError(
            f"ExpertForge configuration root must be a YAML mapping; got {type(result).__name__!r}."
        )
    return result
