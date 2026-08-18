from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from reader.candidate_validation import GateResult


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
    reader_rank: int | None = None
    raw_span_score: float | None = None
    raw_text: str | None = None
    raw_start_char: int | None = None
    raw_end_char: int | None = None
    refinement_method: str = "UNCHANGED"
    refinement_changed: bool = False
    question_relation: str = "FACTOID"
    semantic_relation: str = "GENERAL"
    question_predicate: str | None = None
    question_modifier: str | None = None
    expected_answer_type: str | None = None
    semantic_status: str = "UNKNOWN"
    semantic_policy: str = "GENERAL"
    gate_results: dict[str, GateResult] = field(default_factory=dict)
    relation_validation_reason: str | None = None
    subject_match_reason: str | None = None
    completeness_score: float = 1.0
    completeness_before: float = 1.0
    completeness_after: float = 1.0
    relation_complete: bool = True
    completeness_reasons: tuple[str, ...] = ()
    answer_type_score: float = 0.0
    relation_type: str | None = None
    relation_score: float = 0.0
    relation_method: str | None = None
    question_subject: str | None = None
    question_target: str | None = None
    cause_pattern_score: float = 0.0
    subject_match_score: float = 0.0
    target_relation_score: float = 0.0
    relation_rejection_reason: str | None = None
    evidence_score: float = 0.0
    boundary_score: float = 1.0
    boundary_reasons: tuple[str, ...] = ()
    fallback_penalty: float = 1.0
    ranking_score: float = 0.0
    valid_span: bool = False
    passes_reader_threshold: bool = False
    passes_type_gate: bool = False
    passes_relation_gate: bool = False
    passes_evidence_gate: bool = False
    passes_completeness_gate: bool = True
    passes_final_gate: bool = False
    rejection_reason: str | None = None
    answer_type_reason: str | None = None
    fallback_method: str | None = None
    display_text: str | None = None
    evidence_sentence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["AnswerCandidate"]
