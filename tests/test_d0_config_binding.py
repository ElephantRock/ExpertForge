from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from expertforge.config.resolve import canonical_bytes, resolve_config
from expertforge.identity.fingerprint import (
    ImmutableInput,
    SpecificationFingerprintRecord,
    specification_fingerprint,
)
from scripts.validate_d0_config_binding import (
    CONFIG_PATHS,
    CONTRACT_PATH,
    FINGERPRINT_PATHS,
    ROOT,
    validate_all,
    validate_profile,
)
from scripts.validate_d0_source_manifests import ContractValidationError, load_json_object

LEGACY_SMOKE_CANONICAL_SHA256 = "f6cf719aab809aaaf0d59b79cfba15bda7138c9138089bc7bbd4495cca087217"
EXPECTED_FINGERPRINTS = {
    "qualification": (
        "spec-v1-sha256-02093a3c05c8b5079e2c4fbb8bafaada6365c0776a3a94ca3a8143873167f6d3"
    ),
    "canonical": (
        "spec-v1-sha256-2faaa0f0beed2d1c674c7d29eb969944f9e208825a527e9f7a0024ba538815ea"
    ),
}


def _inputs(profile: str) -> list[ImmutableInput]:
    config = resolve_config(CONFIG_PATHS[profile]).config
    assert config.d0 is not None
    return [
        ImmutableInput(
            name="dataset.manifest",
            algorithm="sha256",
            digest=config.d0.sources.dataset_manifest_sha256,
        ),
        ImmutableInput(
            name="tokenizer.manifest",
            algorithm="sha256",
            digest=config.d0.sources.tokenizer_manifest_sha256,
        ),
    ]


def test_both_d0_profiles_validate() -> None:
    report = validate_all()

    assert set(report) == {"qualification", "canonical"}
    assert (
        report["qualification"]["specification_fingerprint"]
        == EXPECTED_FINGERPRINTS["qualification"]
    )
    assert report["canonical"]["specification_fingerprint"] == EXPECTED_FINGERPRINTS["canonical"]


def test_committed_fingerprint_records_are_self_consistent() -> None:
    for profile, path in FINGERPRINT_PATHS.items():
        record = SpecificationFingerprintRecord.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        record.verify_digest()
        assert record.digest_str == EXPECTED_FINGERPRINTS[profile]


def test_legacy_smoke_canonical_bytes_are_unchanged() -> None:
    digest = hashlib.sha256(
        canonical_bytes(resolve_config(ROOT / "configs/smoke.yaml"))
    ).hexdigest()
    assert digest == LEGACY_SMOKE_CANONICAL_SHA256


def test_yaml_key_order_does_not_change_fingerprint(tmp_path: Path) -> None:
    source = CONFIG_PATHS["qualification"]
    mapping = yaml.safe_load(source.read_text(encoding="utf-8"))
    reordered = dict(reversed(list(mapping.items())))
    rewritten = tmp_path / "qualification-reordered.yaml"
    rewritten.write_text(yaml.safe_dump(reordered, sort_keys=False), encoding="utf-8")

    original = specification_fingerprint(
        canonical_bytes(resolve_config(source)),
        _inputs("qualification"),
    )
    candidate = specification_fingerprint(
        canonical_bytes(resolve_config(rewritten)),
        _inputs("qualification"),
    )
    assert candidate.digest_str == original.digest_str


def test_meaningful_config_change_changes_fingerprint(tmp_path: Path) -> None:
    source = CONFIG_PATHS["qualification"]
    mapping = yaml.safe_load(source.read_text(encoding="utf-8"))
    mapping["training"]["seed"] = 2026080201
    mapping["d0"]["seeds"]["master_seed"] = 2026080201
    changed = tmp_path / "qualification-seed-changed.yaml"
    changed.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")

    original = specification_fingerprint(
        canonical_bytes(resolve_config(source)),
        _inputs("qualification"),
    )
    candidate = specification_fingerprint(
        canonical_bytes(resolve_config(changed)),
        _inputs("qualification"),
    )
    assert candidate.digest_str != original.digest_str


def test_manifest_digest_change_changes_fingerprint() -> None:
    envelope = resolve_config(CONFIG_PATHS["qualification"])
    original = specification_fingerprint(canonical_bytes(envelope), _inputs("qualification"))
    changed_inputs = [
        ImmutableInput(name="dataset.manifest", algorithm="sha256", digest="0" * 64),
        _inputs("qualification")[1],
    ]
    candidate = specification_fingerprint(canonical_bytes(envelope), changed_inputs)
    assert candidate.digest_str != original.digest_str


def test_contract_binding_rejects_parameter_drift(tmp_path: Path) -> None:
    source = CONFIG_PATHS["qualification"]
    mapping = yaml.safe_load(source.read_text(encoding="utf-8"))
    mapping["d0"]["model"]["trainable_parameters"] += 1
    changed = tmp_path / "qualification-parameter-drift.yaml"
    changed.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")

    with pytest.raises(ContractValidationError, match="disagrees with contract"):
        validate_profile(
            load_json_object(CONTRACT_PATH),
            "qualification",
            config_path=changed,
            fingerprint_path=FINGERPRINT_PATHS["qualification"],
        )


def test_fingerprint_files_are_deterministic_json() -> None:
    for path in FINGERPRINT_PATHS.values():
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert path.read_text(encoding="utf-8") == json.dumps(parsed, indent=2) + "\n"
