from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evaluate_reranking import WEIGHT_CONFIGS, _evaluate_configuration
from reader.answer_refinement import QuestionRelation, detect_question_relation
from reader.cause_relations import assess_cause_candidate, extract_cause_question
from reader.question_type import detect_question_type


ROOT = Path(__file__).resolve().parent


def is_cause_question(question: str) -> bool:
    question_type = detect_question_type(question)[0]
    return detect_question_relation(question, question_type) is QuestionRelation.CAUSE


def apply_cause_semantics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated = copy.deepcopy(rows)
    for row in updated:
        question = str(row.get("question") or "")
        frame = extract_cause_question(question)
        if frame is None:
            continue
        for candidate in row.get("candidates", []):
            answer = str(candidate.get("text") or "")
            passage = str(candidate.get("passage_text") or "")
            cause = assess_cause_candidate(question, passage, answer)
            explicitly_causal = bool(
                cause.cause_pattern_score > 0.0
                or candidate.get("relation_type") == "CAUSE"
                or candidate.get("fallback_method") == "cause_clause_pattern"
            )
            if not explicitly_causal:
                continue
            candidate.update(
                {
                    "question_relation": "CAUSE",
                    "question_subject": frame.subject,
                    "question_target": frame.target,
                    "relation_type": "CAUSE",
                    "relation_method": cause.relation_method,
                    "relation_score": cause.relation_score,
                    "cause_pattern_score": cause.cause_pattern_score,
                    "subject_match_score": cause.subject_match_score,
                    "target_relation_score": cause.target_relation_score,
                    "relation_rejection_reason": cause.rejection_reason,
                    "passes_relation_gate": cause.relation_evidence,
                }
            )
    return updated


def evaluate_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _evaluate_configuration(
        rows,
        WEIGHT_CONFIGS["B_R40_Reader40_Type20"],
        0.60,
        phrase_fallback_penalty=1.0,
    )


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    ranking = metrics.get("ranking", {})
    return {
        "overall_f1": metrics["overall"]["f1"],
        "answerable_f1": metrics["answerable"]["f1"],
        "unanswerable_accuracy": metrics["unanswerable"]["accuracy"],
        "false_positives": ranking.get("false_positive_answer_emitted", 0),
        "false_negatives": ranking.get("answerable_empty_prediction", 0),
        "answerable_count": metrics["answerable"]["count"],
        "unanswerable_count": metrics["unanswerable"]["count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a fixed candidate pool through CAUSE semantics")
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "results" / "reranking_candidate_cache_refinement100_v4.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "cause_semantic_replay100.json",
    )
    args = parser.parse_args()

    payload = json.loads(args.cache.read_text(encoding="utf-8"))
    rows = list(payload.get("rows") or [])
    after_rows = apply_cause_semantics(rows)
    cause_rows_before = [row for row in rows if is_cause_question(str(row.get("question") or ""))]
    cause_rows_after = [row for row in after_rows if is_cause_question(str(row.get("question") or ""))]

    before_metrics, before_predictions = evaluate_rows(rows)
    after_metrics, after_predictions = evaluate_rows(after_rows)
    cause_before_metrics, cause_before_predictions = evaluate_rows(cause_rows_before)
    cause_after_metrics, cause_after_predictions = evaluate_rows(cause_rows_after)
    report = {
        "methodology": (
            "Fixed-candidate semantic replay: retriever, Reader checkpoint, tokenizer, spans, "
            "weights, threshold 0.60, and phrase penalty 1.0 are unchanged. Only CAUSE "
            "subject/target relation fields and gates are recomputed."
        ),
        "cache": str(args.cache),
        "rows": len(rows),
        "cause_rows": len(cause_rows_before),
        "config": WEIGHT_CONFIGS["B_R40_Reader40_Type20"],
        "threshold": 0.60,
        "before": compact_metrics(before_metrics),
        "after": compact_metrics(after_metrics),
        "cause_before": compact_metrics(cause_before_metrics),
        "cause_after": compact_metrics(cause_after_metrics),
        "cause_predictions_before": cause_before_predictions,
        "cause_predictions_after": cause_after_predictions,
        "predictions_changed": sum(
            int(before.get("predicted_answer") != after.get("predicted_answer"))
            for before, after in zip(before_predictions, after_predictions)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "cause_rows", "before", "after", "cause_before", "cause_after", "predictions_changed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
