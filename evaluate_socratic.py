from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from backend.socratic import normalize_question, question_similarity
from backend.viqa_api import INDEX, _lookup_socratic_passage, socratic_followups


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "tests" / "data" / "socratic_diagnostic_v1.json"
DEFAULT_BASELINE = ROOT / "tests" / "data" / "socratic_baseline_v1.json"


def _canonical_relation(value: Any) -> str:
    relation = str(value or "GENERAL").upper()
    return {
        "ENTITY": "IDENTITY",
        "DEFINITION": "IDENTITY",
        "GENERIC_LOCATION": "LOCATION",
    }.get(relation, relation)


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _retrieved_ids(question: str, selected_id: str | None, include_retrieved: bool) -> list[str]:
    identifiers = [selected_id] if selected_id else []
    if include_retrieved:
        identifiers.extend(
            hit.passage.metadata.passage_id
            for hit in INDEX.retrieve(question, "bm25", 10)
        )
    return list(dict.fromkeys(identifier for identifier in identifiers if identifier))


def _accepted_trace(debug: dict[str, Any], question: str) -> dict[str, Any]:
    normalized = normalize_question(question)
    return next(
        (
            candidate
            for candidate in debug.get("candidates", [])
            if candidate.get("accepted")
            and normalize_question(candidate.get("question", "")) == normalized
        ),
        {},
    )


def evaluate(
    path: Path,
    *,
    include_retrieved: bool = False,
    baseline_path: Path | None = DEFAULT_BASELINE,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    human_review: list[dict[str, Any]] = []
    grades: Counter[str] = Counter()
    rejection_distribution: Counter[str] = Counter()
    relation_cases: Counter[str] = Counter()
    relation_cases_with_followups: Counter[str] = Counter()
    overall_latencies: list[float] = []
    tier_one_latencies: list[float] = []
    probe_latencies: list[float] = []
    generated_total = 0
    opportunities_total = 0
    accepted_total = 0

    for case in payload.get("cases", []):
        selected_id = case.get("selected_passage_id")
        current_relation = _canonical_relation(case.get("relation"))
        relation_cases[current_relation] += 1
        result = socratic_followups(
            {
                "question": case.get("question"),
                "answer": case.get("answer"),
                "subject": case.get("subject"),
                "relation": case.get("relation"),
                "target": case.get("target"),
                "predicate": case.get("predicate"),
                "selected_passage_id": selected_id,
                "retrieved_passage_ids": _retrieved_ids(
                    str(case.get("question") or ""),
                    selected_id,
                    include_retrieved,
                ),
                "visited_relations": [case.get("relation")],
                "asked_questions": [case.get("question")],
                "limit": 3,
                "debug": True,
            }
        )
        debug = result.get("debug", {})
        generation = debug.get("candidate_generation", {})
        opportunities = debug.get("semantic_opportunities", {})
        latency = debug.get("latency", {})
        generated_total += int(generation.get("generated", 0))
        opportunities_total += int(opportunities.get("detected", 0))
        rejection_distribution.update(debug.get("rejection_distribution", {}))
        overall_latencies.append(float(result.get("processing_time_ms", 0)))
        tier_one_latencies.append(float(latency.get("tier_1_ms", 0.0)))
        probe_latencies.append(float(latency.get("bm25_probe_ms", 0.0)))

        followups = []
        for followup in result.get("followups", []):
            accepted_total += 1
            source = _lookup_socratic_passage(followup.get("source_passage_id"))
            followup_relation = _canonical_relation(followup.get("relation"))
            semantic_duplicate = followup_relation == current_relation
            near_exact_duplicate = question_similarity(
                followup.get("question", ""),
                case.get("question", ""),
            ) >= 0.93
            off_topic = normalize_question(followup.get("subject") or "") != normalize_question(
                case.get("subject") or ""
            )
            if source is None:
                grade = "UNANSWERABLE"
            elif semantic_duplicate or near_exact_duplicate:
                grade = "REDUNDANT"
            elif off_topic:
                grade = "OFF_TOPIC"
            else:
                grade = "USEFUL"
            grades[grade] += 1
            trace = _accepted_trace(debug, followup.get("question", ""))
            followups.append({**followup, "diagnostic_grade": grade})
            human_review.append(
                {
                    "original_question": case.get("question"),
                    "original_answer": case.get("answer"),
                    "followup": followup.get("question"),
                    "relation": followup.get("relation"),
                    "source_passage_id": followup.get("source_passage_id"),
                    "source_passage": (source or {}).get("text"),
                    "evidence_sentence": trace.get("evidence_sentence"),
                    "why_accepted": trace.get("why_accepted"),
                    "answerability_score": followup.get("answerability_score"),
                    "topic_relevance_score": trace.get("topic_relevance_score"),
                    "automated_grade": grade,
                }
            )
        if followups:
            relation_cases_with_followups[current_relation] += 1
        rows.append(
            {
                "id": case.get("id"),
                "question": case.get("question"),
                "current_relation": case.get("relation"),
                "followups": followups,
                "status": debug.get("status"),
                "candidate_generation": generation,
                "rejections": debug.get("rejection_distribution", {}),
                "latency_ms": result.get("processing_time_ms", 0),
                "tier_1_latency_ms": latency.get("tier_1_ms", 0.0),
                "bm25_probe_latency_ms": latency.get("bm25_probe_ms", 0.0),
            }
        )

    total_cases = len(rows)
    cases_with_followups = sum(bool(row["followups"]) for row in rows)
    total_followups = sum(grades.values())
    denominator = max(1, total_followups)
    coverage_by_relation = {
        relation: {
            "total_cases": relation_cases[relation],
            "cases_with_followup": relation_cases_with_followups[relation],
            "coverage_rate": round(
                relation_cases_with_followups[relation] / max(1, relation_cases[relation]),
                4,
            ),
        }
        for relation in sorted(relation_cases)
    }
    report: dict[str, Any] = {
        "diagnostic": payload.get("name", path.stem),
        "context_mode": "selected_plus_top_retrieved" if include_retrieved else "selected_only",
        "total_cases": total_cases,
        "cases_with_followup": cases_with_followups,
        "cases_without_followup": total_cases - cases_with_followups,
        "coverage_rate": round(cases_with_followups / max(1, total_cases), 4),
        "total_semantic_opportunities": opportunities_total,
        "total_candidates_generated": generated_total,
        "total_candidates_accepted": accepted_total,
        "avg_followups_per_answer": round(total_followups / max(1, total_cases), 3),
        "grades": dict(grades),
        "useful_proxy": round(grades["USEFUL"] / denominator, 4),
        "answerability_proxy": round((total_followups - grades["UNANSWERABLE"]) / denominator, 4),
        "duplicate_rate": round(grades["REDUNDANT"] / denominator, 4),
        "off_topic_rate": round(grades["OFF_TOPIC"] / denominator, 4),
        "coverage_by_current_relation": coverage_by_relation,
        "rejection_distribution": dict(rejection_distribution.most_common()),
        "latency_ms": {
            "tier_1_average": round(mean(tier_one_latencies), 3) if tier_one_latencies else 0.0,
            "bm25_probe_average": round(mean(probe_latencies), 3) if probe_latencies else 0.0,
            "overall_average": round(mean(overall_latencies), 3) if overall_latencies else 0.0,
            "p50": round(_percentile(overall_latencies, 0.50), 3),
            "p95": round(_percentile(overall_latencies, 0.95), 3),
        },
        "grading_note": (
            "Automated diagnostic proxy; human_review_samples require manual judgment and are not "
            "claims of 100% usefulness."
        ),
        "human_review_samples": human_review[:50],
        "cases": rows,
    }
    if baseline_path and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        report["before_after"] = {
            "before": baseline,
            "after": {
                "coverage_rate": report["coverage_rate"],
                "cases_with_followup": cases_with_followups,
                "cases_without_followup": total_cases - cases_with_followups,
                "avg_followups_per_answer": report["avg_followups_per_answer"],
                "answerability_proxy": report["answerability_proxy"],
                "useful_proxy": report["useful_proxy"],
                "duplicate_rate": report["duplicate_rate"],
                "off_topic_rate": report["off_topic_rate"],
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grounded Socratic follow-up generation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--include-retrieved", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--human-review-output",
        type=Path,
        help="Optional standalone JSON export of up to 50 review samples.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = evaluate(
        args.input,
        include_retrieved=args.include_retrieved,
        baseline_path=args.baseline,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.human_review_output:
        args.human_review_output.parent.mkdir(parents=True, exist_ok=True)
        args.human_review_output.write_text(
            json.dumps(report["human_review_samples"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if not args.quiet:
        print(rendered)


if __name__ == "__main__":
    main()
