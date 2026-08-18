from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SpanCandidate:
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    score: float
    start_score: float | None = None
    end_score: float | None = None


def select_span_candidates(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    offsets: Sequence[Sequence[int] | None],
    sequence_ids: Sequence[int | None] | None = None,
    top_n_start: int = 20,
    top_n_end: int = 20,
    max_answer_length: int = 40,
    limit: int = 20,
) -> list[SpanCandidate]:
    """Rank valid joint start/end spans restricted to context tokens."""

    start_indexes = np.argsort(np.asarray(start_logits))[::-1][:top_n_start]
    end_indexes = np.argsort(np.asarray(end_logits))[::-1][:top_n_end]
    candidates: dict[tuple[int, int], SpanCandidate] = {}
    for raw_start_index in start_indexes:
        start_index = int(raw_start_index)
        start_offset = offsets[start_index]
        if start_offset is None or (sequence_ids is not None and sequence_ids[start_index] != 1):
            continue
        for raw_end_index in end_indexes:
            end_index = int(raw_end_index)
            end_offset = offsets[end_index]
            if end_offset is None or (sequence_ids is not None and sequence_ids[end_index] != 1):
                continue
            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                continue
            start_char = int(start_offset[0])
            end_char = int(end_offset[1])
            if end_char <= start_char:
                continue
            score = float(start_logits[start_index] + end_logits[end_index])
            key = (start_char, end_char)
            candidate = SpanCandidate(
                start_index,
                end_index,
                start_char,
                end_char,
                score,
                float(start_logits[start_index]),
                float(end_logits[end_index]),
            )
            if key not in candidates or score > candidates[key].score:
                candidates[key] = candidate
    return sorted(candidates.values(), key=lambda candidate: candidate.score, reverse=True)[:limit]


def select_best_span(*args, **kwargs) -> SpanCandidate | None:
    candidates = select_span_candidates(*args, **kwargs, limit=1)
    return candidates[0] if candidates else None


def score_margin_to_confidence(score_margin: float, temperature: float = 10.0) -> float:
    """Map an uncalibrated logit margin to a bounded ranking signal.

    This value is explicitly not a calibrated probability. The no-answer
    decision is made against the raw score margin, never this display score.
    """

    scaled = score_margin / max(0.01, temperature)
    if scaled >= 0:
        return 1.0 / (1.0 + math.exp(-scaled))
    value = math.exp(scaled)
    return value / (1.0 + value)


def should_return_answer(score_margin: float, threshold: float) -> bool:
    """A larger best-span-minus-null margin is stronger evidence for an answer."""

    return bool(math.isfinite(score_margin) and score_margin >= threshold)


def has_clean_word_boundaries(context: str, start: int, end: int) -> bool:
    if start < 0 or end <= start or end > len(context):
        return False
    starts_inside_word = start > 0 and context[start - 1].isalnum() and context[start].isalnum()
    ends_inside_word = end < len(context) and context[end - 1].isalnum() and context[end].isalnum()
    return not starts_inside_word and not ends_inside_word
