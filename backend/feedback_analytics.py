from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


NO_ANSWER_COMPLAINTS = {"NO_ANSWER_BUT_SHOULD_HAVE", "ANSWERED_BUT_SHOULD_NOT"}


def blind_spot_score(failure_rate: float, sample_count: int) -> float:
    return round(max(0.0, failure_rate) * math.log1p(max(0, sample_count)), 6)


def _weight(record: Mapping[str, Any]) -> int:
    try:
        return max(1, int(record.get("duplicate_count", 1)))
    except (TypeError, ValueError):
        return 1


def _bucket(records: Iterable[Mapping[str, Any]], field: str, empty_label: str) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "incorrect": 0, "no_answer": 0}
    )
    for record in records:
        key = str(record.get(field) or empty_label)
        count = _weight(record)
        bucket = buckets[key]
        bucket["total"] += count
        kind = str(record.get("feedback_type") or "")
        if kind == "CORRECT":
            bucket["correct"] += count
        elif kind in NO_ANSWER_COMPLAINTS:
            bucket["no_answer"] += count
        else:
            bucket["incorrect"] += count

    rows: list[dict[str, Any]] = []
    for key, counts in buckets.items():
        failures = counts["incorrect"] + counts["no_answer"]
        rate = failures / counts["total"] if counts["total"] else 0.0
        rows.append(
            {
                field: key,
                **counts,
                "failure_rate": round(rate, 6),
                "blind_spot_score": blind_spot_score(rate, counts["total"]),
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["blind_spot_score"], row["total"], str(row[field])),
        reverse=True,
    )


def build_feedback_analytics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    total = sum(_weight(row) for row in rows)
    correct = sum(_weight(row) for row in rows if row.get("feedback_type") == "CORRECT")
    no_answer = sum(
        _weight(row) for row in rows if row.get("feedback_type") in NO_ANSWER_COMPLAINTS
    )
    incorrect = total - correct
    real = sum(_weight(row) for row in rows if not row.get("synthetic"))
    synthetic = total - real
    pending = sum(1 for row in rows if row.get("status") == "PENDING")

    reason_counts: Counter[str] = Counter()
    gap_counts: Counter[str] = Counter()
    trend: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "incorrect": 0, "no_answer": 0}
    )
    heatmap: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "failures": 0}
    )
    for row in rows:
        count = _weight(row)
        reason = str(row.get("rejection_reason") or "").strip()
        if reason:
            reason_counts[reason] += count
        gap = str(row.get("gap_type") or "").strip()
        if gap:
            gap_counts[gap] += count

        day = str(row.get("timestamp") or "unknown")[:10]
        trend_row = trend[day]
        trend_row["total"] += count
        kind = str(row.get("feedback_type") or "")
        if kind == "CORRECT":
            trend_row["correct"] += count
        elif kind in NO_ANSWER_COMPLAINTS:
            trend_row["no_answer"] += count
            trend_row["incorrect"] += count
        else:
            trend_row["incorrect"] += count

        question_type = str(row.get("question_type") or "UNKNOWN")
        relation = str(row.get("semantic_relation") or "UNKNOWN")
        cell = heatmap[(question_type, relation)]
        cell["total"] += count
        if kind != "CORRECT":
            cell["failures"] += count

    heatmap_cells = []
    for (question_type, relation), counts in heatmap.items():
        rate = counts["failures"] / counts["total"] if counts["total"] else 0.0
        heatmap_cells.append(
            {
                "question_type": question_type,
                "relation": relation,
                **counts,
                "failure_rate": round(rate, 6),
                "blind_spot_score": blind_spot_score(rate, counts["total"]),
            }
        )

    return {
        "summary": {
            "total_feedback": total,
            "unique_records": len(rows),
            "correct": correct,
            "incorrect": incorrect,
            "correct_rate": round(correct / total, 6) if total else 0.0,
            "incorrect_rate": round(incorrect / total, 6) if total else 0.0,
            "no_answer_complaints": no_answer,
            "no_answer_complaint_rate": round(no_answer / total, 6) if total else 0.0,
            "pending_review": pending,
            "real_feedback": real,
            "synthetic_feedback": synthetic,
        },
        "relations": _bucket(rows, "semantic_relation", "UNKNOWN"),
        "question_types": _bucket(rows, "question_type", "UNKNOWN"),
        "gap_types": [
            {"gap_type": key, "count": value}
            for key, value in gap_counts.most_common()
        ],
        "top_rejection_reasons": [
            {"reason": key, "count": value}
            for key, value in reason_counts.most_common(12)
        ],
        "trend": [
            {"date": day, **counts}
            for day, counts in sorted(trend.items())
        ],
        "heatmap": {
            "dimensions": ["question_type", "semantic_relation"],
            "question_types": sorted({item["question_type"] for item in heatmap_cells}),
            "relations": sorted({item["relation"] for item in heatmap_cells}),
            "cells": sorted(
                heatmap_cells,
                key=lambda item: (item["question_type"], item["relation"]),
            ),
        },
        "methodology": {
            "failure_definition": "all feedback except CORRECT",
            "blind_spot_score": "failure_rate * ln(1 + sample_count)",
            "synthetic_is_separated": True,
        },
    }


__all__ = ["blind_spot_score", "build_feedback_analytics"]
