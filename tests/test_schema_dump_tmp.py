from __future__ import annotations

import base64
import json
import zlib

from expertforge.experiments import experiment_manifest_json_schema


def test_dump_schema_for_commit() -> None:
    payload = json.dumps(
        experiment_manifest_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    encoded = base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")
    raise AssertionError(f"SCHEMA_ZLIB_B64={encoded}")
