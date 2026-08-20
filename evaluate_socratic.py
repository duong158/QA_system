from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from backend.socratic import question_similarity
from backend.viqa_api import _lookup_socratic_passage, socratic_followups


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "tests" / "data" / "socratic_diagnostic_v1.json"


def _canonical_relation(value: Any) -> str:
    relation = str(value or "GENERAL").upper()
    return {"ENTITY": "IDENTITY", "DEFINITION": "IDENTITY"}.get(relation, relation)


def evaluate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    grades: Counter[str] = Counter()
    latencies: list[int] = []

    for case in payload.get("cases", []):
        selected_id = case.get("selected_passage_id")
        result = socratic_followups(
            {
                "question": case.get("question"),
                "answer": case.get("answer"),
                "subject": case.get("subject"),
                "relation": case.get("relation"),
                "selected_passage_id": selected_id,
                "retrieved_passage_ids": [selected_id] if selected_id else [],
                "visited_relations": [case.get("relation")],
                "asked_questions": [case.get("question")],
                "limit": 3,
            }
        )
        latencies.append(int(result.get("processing_time_ms", 0)))
        followups = []
        for followup in result.get("followups", []):
            source_exists = _lookup_socratic_passage(followup.get("source_passage_id")) is not None
            redundant = (
                _canonical_relation(followup.get("relation")) == _canonical_relation(case.get("relation"))
                or question_similarity(followup.get("question", ""), case.get("question", "")) >= 0.72
            )
            off_topic = str(followup.get("subject") or "").casefold() != str(case.get("subject") or "").casefold()
            if not source_exists:
                grade = "UNANSWERABLE"
            elif redundant:
                grade = "REDUNDANT"
            elif off_topic:
                grade = "OFF_TOPIC"
            else:
                grade = "USEFUL"
            grades[grade] += 1
            followups.append({**followup, "diagnostic_grade": grade})
        rows.append(
            {
                "id": case.get("id"),
                "question": case.get("question"),
                "current_relation": case.get("relation"),
                "followups": followups,
                "latency_ms": result.get("processing_time_ms", 0),
            }
        )

    total_followups = sum(grades.values())
    denominator = max(1, total_followups)
    return {
        "diagnostic": payload.get("name", path.stem),
        "case_count": len(rows),
        "followup_count": total_followups,
        "cases_with_followups": sum(bool(row["followups"]) for row in rows),
        "average_followups_per_case": round(total_followups / max(1, len(rows)), 3),
        "average_followup_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
        "grades": dict(grades),
        "useful_rate": round(grades["USEFUL"] / denominator, 4),
        "answerable_rate": round((total_followups - grades["UNANSWERABLE"]) / denominator, 4),
        "duplicate_rate": round(grades["REDUNDANT"] / denominator, 4),
        "off_topic_rate": round(grades["OFF_TOPIC"] / denominator, 4),
        "grading_note": "Automated diagnostic proxy; usefulness still requires human review before benchmark claims.",
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grounded Socratic follow-up generation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
