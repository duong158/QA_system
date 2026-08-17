from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AnswerCandidate:
    """One grounded answer proposal retained until the final QA decision.

    Reader margin is intentionally a signal, not a deletion switch. Hard gates
    are reserved for invalid spans, unsupported evidence, and impossible
    answer-type/relation matches.
    """

    text: str
    method: str
    passage_id: str
    start_char: int
    end_char: int
    reader_score: float
    score_margin: float | None
    answer_type_score: float = 0.0
    relation_type: str | None = None
    relation_score: float = 0.0
    evidence_score: float = 0.0
    fallback_penalty: float = 1.0
    ranking_score: float = 0.0
    valid_span: bool = False
    passes_reader_threshold: bool = False
    passes_type_gate: bool = False
    passes_relation_gate: bool = False
    passes_evidence_gate: bool = False
    passes_final_gate: bool = False
    rejection_reason: str | None = None
    answer_type_reason: str | None = None
    fallback_method: str | None = None
    display_text: str | None = None
    evidence_sentence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["AnswerCandidate"]
