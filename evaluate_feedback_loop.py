from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from backend.feedback import FeedbackStore
from backend.feedback_analytics import build_feedback_analytics


ROOT = Path(__file__).resolve().parent


def build_report(store: FeedbackStore) -> dict:
    records = store.list_feedback(limit=1_000)
    analytics = build_feedback_analytics(records)
    statuses = Counter(record.get("status") or "UNKNOWN" for record in records)
    positive = sum(
        max(1, int(record.get("duplicate_count", 1)))
        for record in records
        if record.get("feedback_type") == "CORRECT"
    )
    total = analytics["summary"]["total_feedback"]
    return {
        "total_feedback": total,
        "approved": statuses["APPROVED"],
        "rejected": statuses["REJECTED"],
        "reviewed": statuses["REVIEWED"],
        "pending": statuses["PENDING"],
        "positive": positive,
        "negative": total - positive,
        "real_feedback": analytics["summary"]["real_feedback"],
        "synthetic_feedback": analytics["summary"]["synthetic_feedback"],
        "gap_distribution": analytics["gap_types"],
        "failure_by_relation": analytics["relations"],
        "failure_by_question_type": analytics["question_types"],
        "top_rejection_reasons": analytics["top_rejection_reasons"],
        "heatmap": analytics["heatmap"],
        "methodology": analytics["methodology"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the human-in-the-loop feedback store.")
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "data" / "feedback" / "feedback.db",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "results" / "knowledge_blind_spot_report.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=ROOT / "results" / "knowledge_blind_spot_report.csv",
    )
    args = parser.parse_args()

    report = build_report(FeedbackStore(args.db))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with args.csv_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dimension",
                "value",
                "total",
                "correct",
                "incorrect",
                "no_answer",
                "failure_rate",
                "blind_spot_score",
            ),
        )
        writer.writeheader()
        for dimension, key, rows in (
            ("semantic_relation", "semantic_relation", report["failure_by_relation"]),
            ("question_type", "question_type", report["failure_by_question_type"]),
        ):
            for row in rows:
                writer.writerow(
                    {
                        "dimension": dimension,
                        "value": row[key],
                        "total": row["total"],
                        "correct": row["correct"],
                        "incorrect": row["incorrect"],
                        "no_answer": row["no_answer"],
                        "failure_rate": row["failure_rate"],
                        "blind_spot_score": row["blind_spot_score"],
                    }
                )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
