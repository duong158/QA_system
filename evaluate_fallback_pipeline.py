from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_reranking import load_stratified_subset
from reader.metrics import evaluate_predictions, exact_match, normalize_answer
from reader.question_type import detect_question_type

import polars as pl


ROOT = Path(__file__).resolve().parent


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]


def selected_candidate(response: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (
            candidate
            for candidate in response.get("passages", [])
            if candidate.get("selection_status") == "SELECTED"
        ),
        None,
    )


def load_question_type_subset(
    question_type: str,
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = pl.read_parquet(ROOT / "data" / "processed" / "viquad_val_clean.parquet").to_dicts()
    rows = [
        row
        for row in rows
        if detect_question_type(str(row["question"]))[0].value == question_type
    ]
    random.Random(seed).shuffle(rows)
    buckets: dict[bool, list[dict[str, Any]]] = {True: [], False: []}
    for row in rows:
        raw_start = row.get("answer_start", -1)
        answer_start = int(raw_start if raw_start is not None else -1)
        answerable = answer_start >= 0 and bool(row.get("answer_text"))
        buckets[answerable].append(row)

    selected: list[dict[str, Any]] = []
    while len(selected) < min(size, len(rows)) and (buckets[True] or buckets[False]):
        for answerable in (True, False):
            if buckets[answerable] and len(selected) < size:
                selected.append(buckets[answerable].pop())
    return selected


def classify_fallback_errors(
    response: dict[str, Any],
    *,
    gold: str,
    answerable: bool,
    gold_passage_available: bool,
) -> list[str]:
    if not answerable or response.get("has_answer") or not gold_passage_available:
        return []

    gold_normalized = normalize_answer(gold)
    supporting = [
        candidate
        for candidate in response.get("passages", [])
        if gold_normalized and gold_normalized in normalize_answer(candidate.get("text", ""))
    ]
    errors: list[str] = []
    fallback_candidates = [
        candidate
        for candidate in supporting
        if candidate.get("reader_method") == "sentence_fallback"
    ]
    if fallback_candidates and not any(
        candidate.get("fallback_method") not in {None, "whole_sentence"}
        for candidate in fallback_candidates
    ):
        errors.append("FALLBACK_PHRASE_NOT_EXTRACTED")
    if any(
        candidate.get("rejection_reason") == "INSUFFICIENT_FALLBACK_EVIDENCE"
        for candidate in fallback_candidates
    ):
        errors.append("FALLBACK_TYPE_GATE_FALSE_NEGATIVE")
    if any(
        candidate.get("fallback_method") in {None, "whole_sentence"}
        for candidate in fallback_candidates
    ):
        errors.append("FALLBACK_WHOLE_SENTENCE_TOO_BROAD")
    if any(
        gold_normalized
        and gold_normalized in normalize_answer(candidate.get("fallback_answer", ""))
        and exact_match(gold, candidate.get("neural_reader_answer", "")) == 0
        for candidate in supporting
    ):
        errors.append("NEURAL_SPAN_WRONG_FALLBACK_CORRECT")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark production fallback behavior on a deterministic validation subset"
    )
    parser.add_argument("--subset-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--question-type",
        choices=("TIME", "PERSON", "LOCATION", "NUMBER", "DEFINITION", "ENTITY", "GENERAL"),
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "fallback_pipeline_subset.json",
    )
    args = parser.parse_args()

    from backend.viqa_api import ask_question

    records = (
        load_question_type_subset(args.question_type, args.subset_size, args.seed)
        if args.question_type
        else load_stratified_subset(args.subset_size, args.seed)
    )
    predictions: list[dict[str, Any]] = []
    ranking = Counter()
    fallback = Counter(
        {
            "fallback_candidates_total": 0,
            "fallback_phrase_success": 0,
            "fallback_whole_sentence": 0,
            "fallback_rejected_type": 0,
            "fallback_rejected_evidence": 0,
            "fallback_rejected_ranking": 0,
            "fallback_answered_correctly": 0,
            "fallback_answered_incorrectly": 0,
        }
    )
    error_counts = Counter(
        {
            "FALLBACK_PHRASE_NOT_EXTRACTED": 0,
            "FALLBACK_TYPE_GATE_FALSE_NEGATIVE": 0,
            "NEURAL_SPAN_WRONG_FALLBACK_CORRECT": 0,
            "FALLBACK_WHOLE_SENTENCE_TOO_BROAD": 0,
            "LOCATION_PROPER_NOUN_FALSE_POSITIVE": 0,
            "LOCATION_RELATION_MISMATCH": 0,
            "LOCATION_WHOLE_SENTENCE_TOO_BROAD": 0,
            "LOCATION_PHRASE_NOT_EXTRACTED": 0,
            "LOCATION_FALSE_NEGATIVE": 0,
        }
    )
    location = Counter(
        {
            "count": 0,
            "correct": 0,
            "wrong_location": 0,
            "no_answer": 0,
            "false_positive": 0,
            "false_negative": 0,
            "whole_sentence_accepted": 0,
            "phrase_accepted": 0,
        }
    )
    error_rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    for index, record in enumerate(records, start=1):
        question = str(record["question"])
        raw_answer_start = record.get("answer_start", -1)
        answer_start = int(raw_answer_start if raw_answer_start is not None else -1)
        gold = str(record.get("answer_text") or "")
        answerable = answer_start >= 0 and bool(gold)
        if not answerable:
            gold = ""

        started = time.perf_counter()
        response = ask_question(
            {"question": question, "retriever": "bm25", "reader": "phobert", "top_k": args.top_k}
        )
        latencies.append((time.perf_counter() - started) * 1000)
        prediction = str(response.get("answer") or "")
        predictions.append(
            {
                "id": str(record.get("id", index)),
                "gold_answer": gold,
                "predicted_answer": prediction,
                "is_answerable": answerable,
            }
        )

        candidates = list(response.get("passages") or [])
        gold_normalized = normalize_answer(gold)
        available = bool(gold_normalized) and any(
            gold_normalized in normalize_answer(candidate.get("text", ""))
            for candidate in candidates
        )
        selected = selected_candidate(response)
        selected_has_gold = bool(
            selected
            and gold_normalized
            and gold_normalized in normalize_answer(selected.get("text", ""))
        )
        ranking["gold_answer_passage_available_top10"] += int(answerable and available)
        ranking["gold_passage_selected"] += int(answerable and selected_has_gold)
        ranking["wrong_passage_selected"] += int(
            answerable and selected is not None and not selected_has_gold
        )
        ranking["false_positive_answer_emitted"] += int(not answerable and bool(prediction))
        ranking["false_negative_no_answer_emitted"] += int(answerable and not prediction)
        ranking["no_answer_correctly_emitted"] += int(not answerable and not prediction)

        if response.get("question_type") == "LOCATION":
            location["count"] += 1
            location["correct"] += exact_match(gold, prediction)
            location["wrong_location"] += int(bool(prediction) and not exact_match(gold, prediction))
            location["no_answer"] += int(not prediction)
            location["false_positive"] += int(not answerable and bool(prediction))
            location["false_negative"] += int(answerable and not prediction)
            location["whole_sentence_accepted"] += int(
                bool(selected)
                and selected.get("reader_method") == "sentence_fallback"
                and selected.get("fallback_method") in {None, "whole_sentence"}
            )
            location["phrase_accepted"] += int(
                bool(selected)
                and selected.get("reader_method") == "sentence_fallback"
                and selected.get("fallback_method") not in {None, "whole_sentence"}
            )

            location_errors: list[str] = []
            if selected and selected.get("answer_type_reason") == "NAMED_ENTITY_LOCATION_CANDIDATE":
                if not exact_match(gold, prediction):
                    location_errors.append("LOCATION_PROPER_NOUN_FALSE_POSITIVE")
            if selected and not selected.get("relation_evidence", False):
                location_errors.append("LOCATION_RELATION_MISMATCH")
            if selected and selected.get("fallback_method") in {None, "whole_sentence"}:
                location_errors.append("LOCATION_WHOLE_SENTENCE_TOO_BROAD")
            if not selected and any(
                candidate.get("fallback_method") in {None, "whole_sentence"}
                for candidate in candidates
                if candidate.get("reader_method") == "sentence_fallback"
            ):
                location_errors.append("LOCATION_PHRASE_NOT_EXTRACTED")
            if answerable and not prediction:
                location_errors.append("LOCATION_FALSE_NEGATIVE")
            for error in location_errors:
                error_counts[error] += 1
            if location_errors:
                error_rows.append(
                    {
                        "id": str(record.get("id", index)),
                        "question": question,
                        "gold_answer": gold,
                        "predicted_answer": prediction,
                        "errors": location_errors,
                        "rejection_reason": response.get("rejection_reason"),
                    }
                )

        fallback_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("reader_method") == "sentence_fallback"
        ]
        fallback["fallback_candidates_total"] += len(fallback_candidates)
        for candidate in fallback_candidates:
            method = candidate.get("fallback_method") or "whole_sentence"
            if method == "whole_sentence":
                fallback["fallback_whole_sentence"] += 1
            else:
                fallback["fallback_phrase_success"] += 1
            reason = candidate.get("rejection_reason")
            if reason in {"ANSWER_TYPE_MISMATCH", "INSUFFICIENT_FALLBACK_EVIDENCE"}:
                fallback["fallback_rejected_type"] += 1
            elif reason == "EVIDENCE_UNSUPPORTED":
                fallback["fallback_rejected_evidence"] += 1
            elif reason == "LOW_RANKING_SCORE":
                fallback["fallback_rejected_ranking"] += 1

        if selected and selected.get("reader_method") == "sentence_fallback":
            if exact_match(gold, prediction):
                fallback["fallback_answered_correctly"] += 1
            else:
                fallback["fallback_answered_incorrectly"] += 1

        errors = classify_fallback_errors(
            response,
            gold=gold,
            answerable=answerable,
            gold_passage_available=available,
        )
        for error in errors:
            error_counts[error] += 1
        if errors:
            error_rows.append(
                {
                    "id": str(record.get("id", index)),
                    "question": question,
                    "gold_answer": gold,
                    "predicted_answer": prediction,
                    "errors": errors,
                    "rejection_reason": response.get("rejection_reason"),
                }
            )

        if index % 10 == 0 or index == len(records):
            print(f"Evaluated {index}/{len(records)}", flush=True)

    metrics = evaluate_predictions(predictions)
    total_fallback = fallback["fallback_candidates_total"]
    fallback_metrics = dict(fallback)
    fallback_metrics["fallback_phrase_rate"] = (
        100.0 * fallback["fallback_phrase_success"] / total_fallback if total_fallback else 0.0
    )
    fallback_metrics["fallback_whole_sentence_rate"] = (
        100.0 * fallback["fallback_whole_sentence"] / total_fallback if total_fallback else 0.0
    )

    payload = {
        "dataset": (
            f"UIT-ViQuAD2.0 validation diagnostic {args.question_type} subset"
            if args.question_type
            else "UIT-ViQuAD2.0 validation stratified subset"
        ),
        "subset_size": len(records),
        "seed": args.seed,
        "top_k": args.top_k,
        "question_type_filter": args.question_type,
        "metrics": metrics,
        "ranking": dict(ranking),
        "location": dict(location),
        "fallback": fallback_metrics,
        "error_counts": dict(error_counts),
        "error_analysis": error_rows,
        "latency_ms": {
            "average": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
