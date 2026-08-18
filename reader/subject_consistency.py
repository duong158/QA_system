from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from reader.cause_relations import fold_text
from reader.question_semantics import QuestionSemantics
from reader.semantic_policy import SEMANTIC_POLICIES


class SemanticStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SubjectConsistency:
    status: str
    score: float
    reason: str
    matched_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_COREFERENCE = re.compile(
    r"\b(?:ong|ba|nguoi nay|nhan vat nay|nha van nay|triet gia nay|vi nay)\b"
)
_SUBJECT_STOPWORDS = {
    "cac", "cua", "mot", "nhung", "tai", "the", "trong", "va", "vao", "ve",
}


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[\w]+", fold_text(text), flags=re.UNICODE)


def _sentence_window(context: str, evidence_sentence: str) -> tuple[str, str]:
    if not context or not evidence_sentence:
        return "", ""
    start = context.find(evidence_sentence)
    if start < 0:
        return "", ""
    prefix = context[:start].rstrip()
    previous_start = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?")) + 1
    previous = prefix[previous_start:].strip()
    suffix = context[start + len(evidence_sentence):].lstrip()
    boundaries = [index for mark in ".!?" if (index := suffix.find(mark)) >= 0]
    next_sentence = suffix[: min(boundaries) + 1].strip() if boundaries else suffix.strip()
    return previous, next_sentence


def score_subject_consistency(
    semantics: QuestionSemantics,
    evidence_sentence: str,
    candidate: str = "",
    context: str = "",
) -> SubjectConsistency:
    subject = str(semantics.subject or "").strip()
    if not subject:
        return SubjectConsistency(SemanticStatus.UNKNOWN.value, 0.0, "QUESTION_SUBJECT_NOT_PARSED")

    subject_folded = " ".join(_normalized_words(subject))
    evidence_folded = " ".join(_normalized_words(evidence_sentence))
    if subject_folded and re.search(rf"\b{re.escape(subject_folded)}\b", evidence_folded):
        return SubjectConsistency(
            SemanticStatus.VALID.value,
            1.0,
            "DIRECT_SUBJECT_MATCH",
            subject,
        )

    subject_tokens = [
        token for token in _normalized_words(subject) if token not in _SUBJECT_STOPWORDS
    ]
    evidence_tokens = set(_normalized_words(evidence_sentence))
    overlap = len(set(subject_tokens) & evidence_tokens) / max(1, len(set(subject_tokens)))
    if overlap >= SEMANTIC_POLICIES.validator_threshold("subject", "token_overlap") and len(subject_tokens) >= 2:
        return SubjectConsistency(
            SemanticStatus.VALID.value,
            0.78,
            "SUBJECT_TOKEN_MATCH",
            " ".join(sorted(set(subject_tokens) & evidence_tokens)),
        )

    previous, next_sentence = _sentence_window(context, evidence_sentence)
    if _COREFERENCE.search(evidence_folded):
        previous_folded = " ".join(_normalized_words(previous))
        if subject_folded and re.search(rf"\b{re.escape(subject_folded)}\b", previous_folded):
            return SubjectConsistency(
                SemanticStatus.VALID.value,
                0.80,
                "PREVIOUS_SENTENCE_COREFERENCE",
                subject,
            )

    # Adjacent context can confirm the entity, but without an explicit
    # pronoun it remains unknown rather than a positive match.
    adjacent = " ".join(_normalized_words(f"{previous} {next_sentence}"))
    if subject_folded and re.search(rf"\b{re.escape(subject_folded)}\b", adjacent):
        return SubjectConsistency(
            SemanticStatus.UNKNOWN.value,
            0.45,
            "ADJACENT_SUBJECT_WITHOUT_COREFERENCE",
            subject,
        )

    return SubjectConsistency(
        SemanticStatus.INVALID.value,
        0.0,
        "SUBJECT_NOT_IN_LOCAL_EVIDENCE",
    )


__all__ = ["SemanticStatus", "SubjectConsistency", "score_subject_consistency"]
