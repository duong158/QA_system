from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from reader.cause_relations import fold_text


_DANGLING_CONNECTORS = (
    "trong khi",
    "hoac",
    "nhung",
    "rang",
    "hay",
    "va",
    "ma",
    "vi",
    "do",
    "boi",
    "nen",
    "neu",
    "khi",
    "con",
)
_DANGLING_RE = re.compile(
    r"(?:^|[\s,;:])(?P<connector>" + "|".join(
        re.escape(item) for item in _DANGLING_CONNECTORS
    ) + r")\s*$"
)
_RIGHT_STOP = re.compile(
    r"(?P<stop>[.;!?]|\s+va\s+vi\s+the\b|\s+vi\s+the\b|\s+do\s+do\b|"
    r"\s+nen\b|\s+khien\b|\s+dan\s+den\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompletenessAssessment:
    complete: bool
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClauseRefinement:
    start: int
    end: int
    method: str
    before: CompletenessAssessment
    after: CompletenessAssessment


def assess_answer_completeness(answer: str) -> CompletenessAssessment:
    text = str(answer or "").strip()
    if not text:
        return CompletenessAssessment(False, 0.0, ("EMPTY_ANSWER",))
    folded = fold_text(text).strip()
    reasons: list[str] = []
    score = 1.0
    if _DANGLING_RE.search(folded):
        score = 0.10
        reasons.append("DANGLING_CONNECTOR")
    if text.count("(") != text.count(")"):
        score = min(score, 0.15)
        reasons.append("UNBALANCED_DELIMITER")
    return CompletenessAssessment(
        complete=score >= 0.50,
        score=score,
        reasons=tuple(reasons) or ("COMPLETE",),
    )


def refine_dangling_clause(
    context: str,
    start: int,
    end: int,
    max_extra_chars: int = 120,
) -> ClauseRefinement:
    before = assess_answer_completeness(context[start:end] if 0 <= start < end <= len(context) else "")
    if before.complete or not (0 <= start < end <= len(context)):
        return ClauseRefinement(start, end, "UNCHANGED", before, before)

    folded = fold_text(context[start:end]).strip()
    connector = _DANGLING_RE.search(folded)
    if connector:
        tail = context[end : min(len(context), end + max_extra_chars)]
        leading = len(tail) - len(tail.lstrip())
        searchable = tail[leading:]
        stop = _RIGHT_STOP.search(fold_text(searchable))
        addition_end = stop.start() if stop else len(searchable)
        addition = searchable[:addition_end].rstrip(" ,;:")
        addition_words = re.findall(r"[\w]+", fold_text(addition), flags=re.UNICODE)
        if addition_words:
            expanded_end = end + leading + len(searchable[:addition_end].rstrip(" ,;:"))
            after = assess_answer_completeness(context[start:expanded_end])
            if after.complete:
                return ClauseRefinement(
                    start,
                    expanded_end,
                    "dangling_connector_expand_right",
                    before,
                    after,
                )

        # Expansion was unsafe or unavailable: remove the connector together
        # with a preceding comma, producing a clean prior clause.
        connector_start = connector.start("connector")
        trimmed_end = start + connector_start
        while trimmed_end > start and context[trimmed_end - 1] in " ,;:":
            trimmed_end -= 1
        after = assess_answer_completeness(context[start:trimmed_end])
        if trimmed_end > start and after.complete:
            return ClauseRefinement(
                start,
                trimmed_end,
                "dangling_connector_trim_clause",
                before,
                after,
            )

    return ClauseRefinement(start, end, "DANGLING_CONNECTOR_UNRESOLVED", before, before)


__all__ = [
    "ClauseRefinement",
    "CompletenessAssessment",
    "assess_answer_completeness",
    "refine_dangling_clause",
]
