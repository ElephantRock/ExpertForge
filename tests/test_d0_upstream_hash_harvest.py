from __future__ import annotations

import hashlib
import json
import urllib.request


def test_harvest_pinned_gpt_neox_tokenizer_hashes() -> None:
    revision = "364ae95407723fadd1d47b023c1efb92a4d891c3"
    files = (
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    inventory: list[dict[str, object]] = []
    for path in files:
        url = (
            "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/"
            f"{revision}/{path}?download=true"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "ExpertForge-D0-contract/1"})
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(request, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        inventory.append({"path": path, "sha256": digest.hexdigest(), "size_bytes": size})

    raise AssertionError("D0_TOKENIZER_INVENTORY=" + json.dumps(inventory, sort_keys=True))
