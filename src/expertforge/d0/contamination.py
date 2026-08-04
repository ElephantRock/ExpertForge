"""Deterministic three-tier D0 contamination matching."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from expertforge.d0.errors import ContaminationError
from expertforge.d0.normalization import normalize_text

CheckName = Literal[
    "exact_normalized_prompt_substring",
    "exact_normalized_probe_substring",
    "any_exact_contiguous_64_codepoint_prompt_window",
]

_FULL_CHECK: CheckName = "exact_normalized_prompt_substring"
_PROBE_CHECK: CheckName = "exact_normalized_probe_substring"
_WINDOW_CHECK: CheckName = "any_exact_contiguous_64_codepoint_prompt_window"
_CHECKS: tuple[CheckName, ...] = (_FULL_CHECK, _PROBE_CHECK, _WINDOW_CHECK)
_CHECK_PRECEDENCE: dict[CheckName, int] = {check: index for index, check in enumerate(_CHECKS)}


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContaminationError(f"{field_name} must be an object")
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContaminationError(f"{field_name} must be an array")
    return value


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContaminationError(f"{field_name} must be a non-empty string")
    return value


def _exact_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContaminationError(f"{field_name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class PromptDefinition:
    """One normalized prompt and its dedicated contamination probe."""

    prompt_id: str
    text: str
    probe: str


@dataclass(frozen=True, slots=True)
class HitRecord:
    """One precedence-selected contamination hit for a prompt/document pair."""

    prompt_id: str
    check: CheckName
    document_id: str
    source_file_path: str
    physical_row_index: int
    match_start_codepoint: int
    match_end_codepoint: int

    def as_dict(self) -> dict[str, object]:
        """Return the frozen machine-record field set."""

        return {
            "prompt_id": self.prompt_id,
            "check": self.check,
            "document_id": self.document_id,
            "source_file_path": self.source_file_path,
            "physical_row_index": self.physical_row_index,
            "match_start_codepoint": self.match_start_codepoint,
            "match_end_codepoint": self.match_end_codepoint,
        }


@dataclass(frozen=True, slots=True)
class _PatternTarget:
    prompt_id: str
    check: CheckName
    prompt_window_start: int


@dataclass(slots=True)
class _AutomatonNode:
    transitions: dict[str, int] = field(default_factory=dict)
    failure: int = 0
    outputs: list[str] = field(default_factory=list)


class ContaminationMatcher:
    """Aho–Corasick matcher preserving frozen per-prompt precedence semantics."""

    def __init__(self, prompts: Sequence[PromptDefinition], *, window_codepoints: int) -> None:
        if not prompts:
            raise ContaminationError("at least one prompt is required")
        if window_codepoints <= 0:
            raise ContaminationError("window_codepoints must be positive")
        self._prompts = tuple(prompts)
        self._window_codepoints = window_codepoints
        self._prompt_by_id = {prompt.prompt_id: prompt for prompt in self._prompts}
        if len(self._prompt_by_id) != len(self._prompts):
            raise ContaminationError("prompt identifiers must be unique")
        self._patterns = self._build_patterns()
        self._nodes = self._build_automaton()

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> ContaminationMatcher:
        """Construct a matcher from the ratified prompt manifest."""

        policy = _mapping(manifest.get("contamination_policy"), "contamination_policy")
        checks = tuple(_sequence(policy.get("checks_in_precedence_order"), "checks"))
        if checks != _CHECKS:
            raise ContaminationError("contamination check precedence changed")
        window_codepoints = _exact_int(
            policy.get("partial_window_codepoints"),
            "partial_window_codepoints",
            minimum=1,
        )
        prompts: list[PromptDefinition] = []
        for index, raw_prompt in enumerate(_sequence(manifest.get("prompts"), "prompts")):
            prompt = _mapping(raw_prompt, f"prompts[{index}]")
            prompt_id = _string(prompt.get("id"), f"prompts[{index}].id")
            text = normalize_text(_string(prompt.get("text"), f"prompts[{index}].text"))
            probe = normalize_text(
                _string(
                    prompt.get("contamination_probe_text"),
                    f"prompts[{index}].contamination_probe_text",
                )
            )
            if probe not in text:
                raise ContaminationError(f"{prompt_id} probe is not an exact prompt substring")
            if len(text) < window_codepoints:
                raise ContaminationError(f"{prompt_id} is shorter than the contamination window")
            prompts.append(PromptDefinition(prompt_id=prompt_id, text=text, probe=probe))
        return cls(prompts, window_codepoints=window_codepoints)

    def _add_pattern(
        self,
        patterns: dict[str, list[_PatternTarget]],
        pattern: str,
        target: _PatternTarget,
    ) -> None:
        targets = patterns.setdefault(pattern, [])
        if target not in targets:
            targets.append(target)

    def _build_patterns(self) -> dict[str, tuple[_PatternTarget, ...]]:
        patterns: dict[str, list[_PatternTarget]] = {}
        for prompt in self._prompts:
            self._add_pattern(
                patterns,
                prompt.text,
                _PatternTarget(prompt.prompt_id, _FULL_CHECK, 0),
            )
            self._add_pattern(
                patterns,
                prompt.probe,
                _PatternTarget(prompt.prompt_id, _PROBE_CHECK, 0),
            )
            last_start = len(prompt.text) - self._window_codepoints
            for prompt_start in range(last_start + 1):
                window = prompt.text[prompt_start : prompt_start + self._window_codepoints]
                self._add_pattern(
                    patterns,
                    window,
                    _PatternTarget(prompt.prompt_id, _WINDOW_CHECK, prompt_start),
                )
        return {
            pattern: tuple(
                sorted(
                    targets,
                    key=lambda target: (
                        target.prompt_id,
                        _CHECK_PRECEDENCE[target.check],
                        target.prompt_window_start,
                    ),
                )
            )
            for pattern, targets in patterns.items()
        }

    def _build_automaton(self) -> list[_AutomatonNode]:
        nodes = [_AutomatonNode()]
        for pattern in sorted(self._patterns):
            state = 0
            for character in pattern:
                next_state = nodes[state].transitions.get(character)
                if next_state is None:
                    next_state = len(nodes)
                    nodes[state].transitions[character] = next_state
                    nodes.append(_AutomatonNode())
                state = next_state
            nodes[state].outputs.append(pattern)

        queue: deque[int] = deque()
        for _, state in sorted(nodes[0].transitions.items()):
            nodes[state].failure = 0
            queue.append(state)
        while queue:
            state = queue.popleft()
            for character, next_state in sorted(nodes[state].transitions.items()):
                queue.append(next_state)
                fallback = nodes[state].failure
                while fallback and character not in nodes[fallback].transitions:
                    fallback = nodes[fallback].failure
                nodes[next_state].failure = nodes[fallback].transitions.get(character, 0)
                inherited = nodes[nodes[next_state].failure].outputs
                for pattern in inherited:
                    if pattern not in nodes[next_state].outputs:
                        nodes[next_state].outputs.append(pattern)
                nodes[next_state].outputs.sort()
        return nodes

    def scan_document(
        self,
        *,
        document_id: str,
        source_file_path: str,
        physical_row_index: int,
        text: str,
    ) -> tuple[HitRecord, ...]:
        """Return at most one precedence-selected hit per prompt."""

        if not document_id:
            raise ContaminationError("document_id must not be blank")
        if not source_file_path:
            raise ContaminationError("source_file_path must not be blank")
        _exact_int(physical_row_index, "physical_row_index")
        document = normalize_text(text)

        candidates: dict[tuple[str, CheckName], tuple[int, int]] = {}
        state = 0
        for end_index, character in enumerate(document):
            while state and character not in self._nodes[state].transitions:
                state = self._nodes[state].failure
            state = self._nodes[state].transitions.get(character, 0)
            for pattern in self._nodes[state].outputs:
                match_start = end_index - len(pattern) + 1
                for target in self._patterns[pattern]:
                    key = (target.prompt_id, target.check)
                    candidate = (match_start, target.prompt_window_start)
                    previous = candidates.get(key)
                    if previous is None or candidate < previous:
                        candidates[key] = candidate

        hits: list[HitRecord] = []
        for prompt in self._prompts:
            selected_check: CheckName | None = None
            selected: tuple[int, int] | None = None
            for check in _CHECKS:
                prompt_candidate = candidates.get((prompt.prompt_id, check))
                if prompt_candidate is not None:
                    selected_check = check
                    selected = prompt_candidate
                    break
            if selected_check is None or selected is None:
                continue
            match_start = selected[0]
            if selected_check == _FULL_CHECK:
                width = len(prompt.text)
            elif selected_check == _PROBE_CHECK:
                width = len(prompt.probe)
            else:
                width = self._window_codepoints
            hits.append(
                HitRecord(
                    prompt_id=prompt.prompt_id,
                    check=selected_check,
                    document_id=document_id,
                    source_file_path=source_file_path,
                    physical_row_index=physical_row_index,
                    match_start_codepoint=match_start,
                    match_end_codepoint=match_start + width,
                )
            )
        return tuple(hits)


def sort_hit_records(hits: Sequence[HitRecord]) -> tuple[HitRecord, ...]:
    """Sort hit records by the frozen report order with deterministic tie-breaks."""

    return tuple(
        sorted(
            hits,
            key=lambda hit: (
                hit.prompt_id,
                _CHECK_PRECEDENCE[hit.check],
                hit.document_id,
                hit.match_start_codepoint,
                hit.source_file_path,
                hit.physical_row_index,
                hit.match_end_codepoint,
            ),
        )
    )
