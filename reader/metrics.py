from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any, Iterable, Mapping


def normalize_answer(text: str) -> str:
    """Normalize Vietnamese extractive answers consistently across all evaluators."""

    text = unicodedata.normalize("NFC", str(text or "")).casefold().replace("_", " ")
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split())


def exact_match(gold: str, prediction: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(prediction))


def f1_score(gold: str, prediction: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    prediction_tokens = normalize_answer(prediction).split()
    if not gold_tokens and not prediction_tokens:
        return 1.0
    if not gold_tokens or not prediction_tokens:
        return 0.0
    overlap = Counter(gold_tokens) & Counter(prediction_tokens)
    matches = sum(overlap.values())
    if not matches:
        return 0.0
    precision = matches / len(prediction_tokens)
    recall = matches / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# Backwards-compatible names used by existing scripts/tests.
compute_exact = exact_match
compute_f1 = f1_score


def _value(record: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def evaluate_predictions(records: Iterable[Mapping[str, Any] | Any]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction set")

    overall_em: list[int] = []
    overall_f1: list[float] = []
    answerable_em: list[int] = []
    answerable_f1: list[float] = []
    unanswerable_correct: list[int] = []
    predicted_empty = 0
    answerable_predicted_empty = 0

    for row in rows:
        gold = str(_value(row, "gold_answer", _value(row, "gold", "")) or "")
        prediction = str(_value(row, "predicted_answer", _value(row, "prediction", "")) or "")
        is_answerable = bool(_value(row, "is_answerable", bool(gold)))
        em = exact_match(gold, prediction)
        f1 = f1_score(gold, prediction)
        overall_em.append(em)
        overall_f1.append(f1)
        if not prediction.strip():
            predicted_empty += 1
        if is_answerable:
            answerable_em.append(em)
            answerable_f1.append(f1)
            if not prediction.strip():
                answerable_predicted_empty += 1
        else:
            unanswerable_correct.append(int(not prediction.strip()))

    total = len(rows)
    answerable_count = len(answerable_em)
    unanswerable_count = len(unanswerable_correct)
    percent = lambda values: 100.0 * sum(values) / len(values) if values else 0.0
    return {
        "overall": {
            "count": total,
            "em": percent(overall_em),
            "f1": percent(overall_f1),
        },
        "answerable": {
            "count": answerable_count,
            "em": percent(answerable_em),
            "f1": percent(answerable_f1),
            "predicted_empty": answerable_predicted_empty,
            "predicted_empty_rate": (
                100.0 * answerable_predicted_empty / answerable_count if answerable_count else 0.0
            ),
        },
        "unanswerable": {
            "count": unanswerable_count,
            "accuracy": percent(unanswerable_correct),
            "em": percent(unanswerable_correct),
        },
        "predicted_no_answer": {
            "count": predicted_empty,
            "rate": 100.0 * predicted_empty / total,
        },
    }
