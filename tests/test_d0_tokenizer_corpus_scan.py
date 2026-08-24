from __future__ import annotations

from dataclasses import dataclass

from scripts.run_d0_tokenizer_corpus_scan import (
    MAX_REACHABLE_ID,
    ScanAccumulator,
    TokenizerHandle,
    _scan_one_document,
)


@dataclass(frozen=True)
class _Encoding:
    ids: list[int]


class _Tokenizer:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def encode(self, _text: str, *, add_special_tokens: bool) -> _Encoding:
        assert add_special_tokens is False
        return _Encoding(ids=list(self._ids))


def _handle(ids: list[int]) -> TokenizerHandle:
    return TokenizerHandle(
        manifest_id="test-tokenizer",
        manifest_sha256="0" * 64,
        tokenizer_json_sha256="1" * 64,
        impl=_Tokenizer(ids),
    )


def test_scan_one_document_counts_retained_document_and_tokens() -> None:
    acc = ScanAccumulator()

    _scan_one_document(
        _handle([7, 11, 13]),
        document_id="doc-retained-count",
        normalized_text="retained document",
        acc=acc,
    )

    assert acc.documents_tokenized == 1
    assert acc.document_terminators == 1
    assert acc.token_count == 4  # three encoded ids plus the document terminator
    assert acc.train_documents + acc.validation_documents == 1
    assert acc.train_tokens + acc.validation_tokens == 4


def test_scan_one_document_records_out_of_range_ids_without_losing_count() -> None:
    acc = ScanAccumulator()
    first_oor = MAX_REACHABLE_ID + 1
    second_oor = MAX_REACHABLE_ID + 2

    _scan_one_document(
        _handle([3, first_oor, second_oor, first_oor]),
        document_id="doc-out-of-range",
        normalized_text="out of range",
        acc=acc,
    )

    assert acc.documents_tokenized == 1
    assert acc.documents_with_out_of_range == 1
    assert acc.out_of_range_occurrences == 3
    assert acc.per_oor_id_count == {first_oor: 2, second_oor: 1}
    assert acc.max_emitted_id == second_oor
