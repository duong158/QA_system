from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from reader.metrics import evaluate_predictions, exact_match, f1_score, normalize_answer
from reader.question_type import detect_question_type


ROOT = Path(__file__).resolve().parent
WEIGHT_CONFIGS = {
    "A_R50_Reader30_Type20": {"retriever": 0.50, "reader": 0.30, "answer_type": 0.20, "relation": 0.0},
    "B_R40_Reader40_Type20": {"retriever": 0.40, "reader": 0.40, "answer_type": 0.20, "relation": 0.0},
    "C_R40_Reader30_Type20_Relation10": {"retriever": 0.40, "reader": 0.30, "answer_type": 0.20, "relation": 0.10},
    "D_R35_Reader35_Type15_Relation15": {"retriever": 0.35, "reader": 0.35, "answer_type": 0.15, "relation": 0.15},
}
DEFAULT_FINAL_THRESHOLDS = [round(index / 40, 3) for index in range(12, 37)]
PHRASE_FALLBACK_PENALTIES = (0.6, 0.8, 0.9, 1.0)
CACHE_SCHEMA_VERSION = 4


def load_stratified_subset(size: int, seed: int) -> list[dict[str, Any]]:
    rows = pl.read_parquet(ROOT / "data" / "processed" / "viquad_val_clean.parquet").to_dicts()
    random.Random(seed).shuffle(rows)
    buckets: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_answer_start = row.get("answer_start", -1)
        answer_start = int(raw_answer_start if raw_answer_start is not None else -1)
        answerable = answer_start >= 0 and bool(row.get("answer_text"))
        buckets[(detect_question_type(str(row["question"])).value, answerable)].append(row)

    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < min(size, len(rows)) and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < size:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def candidate_score(
    candidate: dict[str, Any],
    config: dict[str, float],
    phrase_fallback_penalty: float | None = None,
) -> float:
    fallback_penalty = float(candidate.get("fallback_penalty", 1.0))
    if candidate.get("method") == "phrase_fallback" and phrase_fallback_penalty is not None:
        fallback_penalty = phrase_fallback_penalty
    boundary_factor = 0.5 + 0.5 * float(candidate.get("boundary_score", 1.0))
    completeness_factor = 0.5 + 0.5 * float(candidate.get("completeness_score", 1.0))
    reader_signal = (
        float(candidate.get("reader_score") or 0.0)
        * fallback_penalty
        * boundary_factor
        * completeness_factor
    )
    return (
        config["retriever"] * float(candidate.get("retrieval_score") or 0.0)
        + config["reader"] * reader_signal
        + config["answer_type"] * float(candidate.get("answer_type_score") or 0.0)
        + config["relation"] * float(candidate.get("relation_score") or 0.0)
    )


def candidate_passes_hard_gates(candidate: dict[str, Any]) -> bool:
    """Hard-reject only structural, evidence, type, and relation failures."""

    return bool(
        candidate.get("text")
        and candidate.get("valid_span")
        and candidate.get("passes_evidence_gate")
        and candidate.get("passes_type_gate")
        and candidate.get("passes_relation_gate")
        and candidate.get("passes_completeness_gate", True)
    )


def has_strong_cause_evidence(candidate: dict[str, Any]) -> bool:
    """Mirror the production CAUSE override without lowering the global gate."""

    return bool(
        candidate.get("relation_type") == "CAUSE"
        and float(candidate.get("cause_pattern_score") or 0.0) >= 0.85
        and float(candidate.get("subject_match_score") or 0.0) >= 0.75
        and float(candidate.get("target_relation_score") or 0.0) >= 0.55
        and float(candidate.get("relation_score") or 0.0) >= 0.80
    )


def select_candidate(
    candidates: list[dict[str, Any]],
    config: dict[str, float],
    final_threshold: float,
    phrase_fallback_penalty: float | None = None,
) -> tuple[dict[str, Any] | None, float]:
    scored = sorted(
        (
            (candidate_score(candidate, config, phrase_fallback_penalty), candidate)
            for candidate in candidates
        ),
        key=lambda item: (item[0], float(item[1].get("evidence_score") or 0.0)),
        reverse=True,
    )
    for ranking_score, candidate in scored:
        if candidate_passes_hard_gates(candidate) and (
            ranking_score >= final_threshold or has_strong_cause_evidence(candidate)
        ):
            return candidate, ranking_score
    return None, (scored[0][0] if scored else 0.0)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]


def answer_length_histogram(values: list[int]) -> dict[str, int]:
    buckets: Counter[str] = Counter()
    for value in values:
        if value == 0:
            label = "0"
        elif value <= 4:
            label = "1-4"
        elif value <= 8:
            label = "5-8"
        elif value <= 16:
            label = "9-16"
        elif value <= 32:
            label = "17-32"
        else:
            label = "33+"
        buckets[label] += 1
    return dict(buckets)


def _evaluate_configuration(
    rows: list[dict[str, Any]],
    config: dict[str, float],
    threshold: float,
    phrase_fallback_penalty: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    raw_predictions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    refinement_outcomes: Counter[str] = Counter()
    length_buckets: Counter[str] = Counter()
    gold_lengths: list[int] = []
    raw_lengths: list[int] = []
    refined_lengths: list[int] = []
    for row in rows:
        selected, ranking_score = select_candidate(
            row["candidates"], config, threshold, phrase_fallback_penalty
        )
        prediction = str((selected or {}).get("display_text") or (selected or {}).get("text") or "")
        raw_prediction = str((selected or {}).get("raw_text") or prediction)
        raw_f1 = f1_score(row["gold_answer"], raw_prediction)
        refined_f1 = f1_score(row["gold_answer"], prediction)
        refinement_method = str((selected or {}).get("refinement_method") or "UNCHANGED")
        changed = bool((selected or {}).get("refinement_changed", False))
        if refined_f1 > raw_f1 + 1e-12:
            outcome = "IMPROVED"
        elif refined_f1 + 1e-12 < raw_f1:
            outcome = "WORSENED"
        elif changed:
            outcome = "BOUNDARY_FIXED"
        else:
            outcome = "UNCHANGED"
        refinement_outcomes[outcome] += 1

        gold_length = len(normalize_answer(row["gold_answer"]).split())
        raw_length = len(normalize_answer(raw_prediction).split())
        refined_length = len(normalize_answer(prediction).split())
        gold_lengths.append(gold_length)
        raw_lengths.append(raw_length)
        refined_lengths.append(refined_length)
        if row["is_answerable"] and gold_length:
            ratio = refined_length / gold_length
            if ratio < 0.5:
                length_bucket = "SEVERE_UNDER_SPAN"
            elif ratio < 0.8:
                length_bucket = "UNDER_SPAN"
            elif ratio <= 1.25:
                length_bucket = "ROUGHLY_CORRECT"
            elif ratio <= 2.0:
                length_bucket = "OVER_SPAN"
            else:
                length_bucket = "SEVERE_OVER_SPAN"
            length_buckets[length_bucket] += 1
        predictions.append(
            {
                "id": row["id"],
                "gold_answer": row["gold_answer"],
                "predicted_answer": prediction,
                "raw_predicted_answer": raw_prediction,
                "is_answerable": row["is_answerable"],
                "refinement_method": refinement_method,
                "refinement_outcome": outcome,
                "raw_f1": raw_f1,
                "refined_f1": refined_f1,
            }
        )
        raw_predictions.append(
            {
                "id": row["id"],
                "gold_answer": row["gold_answer"],
                "predicted_answer": raw_prediction,
                "is_answerable": row["is_answerable"],
            }
        )
        counts["gold_answer_passage_available_top10"] += int(
            row["is_answerable"] and row["gold_available"]
        )
        selected_has_gold = bool(
            selected
            and row["gold_normalized"]
            and row["gold_normalized"] in normalize_answer(selected.get("passage_text", ""))
        )
        counts["gold_passage_selected"] += int(row["is_answerable"] and selected_has_gold)
        counts["wrong_passage_selected"] += int(
            row["is_answerable"] and selected is not None and not selected_has_gold
        )
        counts["no_answer_correctly_emitted"] += int(not row["is_answerable"] and not prediction)
        counts["false_positive_answer_emitted"] += int(not row["is_answerable"] and bool(prediction))
        counts["high_score_wrong_answer"] += int(
            bool(prediction)
            and exact_match(row["gold_answer"], prediction) == 0
            and ranking_score >= 0.8
        )

    metrics = evaluate_predictions(predictions)
    raw_metrics = evaluate_predictions(raw_predictions)
    answerable_f1 = float(metrics["answerable"]["f1"])
    unanswerable_accuracy = float(metrics["unanswerable"]["accuracy"])
    metrics["reader_priority_score"] = 0.7 * answerable_f1 + 0.3 * unanswerable_accuracy
    metrics["ranking"] = dict(counts)
    metrics["ranking"]["high_score_wrong_answer_rate"] = (
        100.0 * counts["high_score_wrong_answer"] / len(rows)
    )
    metrics["refinement"] = {
        "raw": raw_metrics,
        "refined": {
            "overall": metrics["overall"],
            "answerable": metrics["answerable"],
            "unanswerable": metrics["unanswerable"],
        },
        "answerable_f1_delta": (
            float(metrics["answerable"]["f1"])
            - float(raw_metrics["answerable"]["f1"])
        ),
        "outcomes": dict(sorted(refinement_outcomes.items())),
        "length_ratio_buckets": dict(sorted(length_buckets.items())),
        "mean_token_length": {
            "gold": statistics.fmean(gold_lengths) if gold_lengths else 0.0,
            "raw_prediction": statistics.fmean(raw_lengths) if raw_lengths else 0.0,
            "refined_prediction": statistics.fmean(refined_lengths) if refined_lengths else 0.0,
        },
        "token_length_histogram": {
            "gold": answer_length_histogram(gold_lengths),
            "raw_prediction": answer_length_histogram(raw_lengths),
            "refined_prediction": answer_length_histogram(refined_lengths),
        },
    }
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark candidate-pool reranking weights and final no-answer gate"
    )
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_FINAL_THRESHOLDS)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "reranking_validation_subset.json")
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "results" / "reranking_candidate_cache.json",
        help="Reuse only when subset/seed/top-k metadata match",
    )
    args = parser.parse_args()

    records = load_stratified_subset(args.subset_size, args.seed)
    cache_rows: list[dict[str, Any]] | None = None
    if args.cache.is_file():
        cached = json.loads(args.cache.read_text(encoding="utf-8"))
        if (
            cached.get("schema_version") == CACHE_SCHEMA_VERSION
            and cached.get("subset_size") == len(records)
            and cached.get("seed") == args.seed
            and cached.get("top_k") == args.top_k
        ):
            cache_rows = list(cached.get("rows") or [])

    latencies: list[float] = []
    question_types: Counter[str] = Counter()
    if cache_rows is None:
        from backend.viqa_api import ask_question

        cache_rows = []
        for index, record in enumerate(records, start=1):
            question = str(record["question"])
            gold = str(record.get("answer_text") or "")
            answer_start = int(record.get("answer_start") if record.get("answer_start") is not None else -1)
            answerable = answer_start >= 0 and bool(gold)
            if not answerable:
                gold = ""
            question_types[detect_question_type(question).value] += 1

            started = time.perf_counter()
            response = ask_question(
                {"question": question, "retriever": "bm25", "reader": "phobert", "top_k": args.top_k}
            )
            latencies.append((time.perf_counter() - started) * 1000)
            candidates: list[dict[str, Any]] = []
            gold_normalized = normalize_answer(gold)
            gold_available = False
            for passage in response.get("passages", []):
                passage_text = str(passage.get("text") or "")
                gold_available = gold_available or bool(
                    gold_normalized and gold_normalized in normalize_answer(passage_text)
                )
                for candidate in passage.get("candidates", []):
                    copied = dict(candidate)
                    copied["passage_text"] = passage_text
                    candidates.append(copied)
            cache_rows.append(
                {
                    "id": str(record.get("id", index)),
                    "question": question,
                    "gold_answer": gold,
                    "gold_normalized": gold_normalized,
                    "is_answerable": answerable,
                    "gold_available": gold_available,
                    "candidates": candidates,
                }
            )
            if index % 10 == 0 or index == len(records):
                print(f"Generated candidates {index}/{len(records)}", flush=True)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(
            json.dumps(
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "candidate_pipeline": "multi_span_answer_refinement_v1",
                    "subset_size": len(records),
                    "seed": args.seed,
                    "top_k": args.top_k,
                    "rows": cache_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        for row in cache_rows:
            question_types[detect_question_type(row["question"]).value] += 1

    config_results: dict[str, Any] = {}
    for name, config in WEIGHT_CONFIGS.items():
        sweep = []
        for phrase_penalty in PHRASE_FALLBACK_PENALTIES:
            for threshold in sorted(set(args.thresholds)):
                metrics, predictions = _evaluate_configuration(
                    cache_rows,
                    config,
                    threshold,
                    phrase_fallback_penalty=phrase_penalty,
                )
                sweep.append(
                    {
                        "threshold": threshold,
                        "phrase_fallback_penalty": phrase_penalty,
                        "metrics": metrics,
                        "predictions": predictions,
                    }
                )
        unconstrained_best = max(
            sweep,
            key=lambda item: (
                item["metrics"]["reader_priority_score"],
                item["metrics"]["overall"]["f1"],
                item["metrics"]["unanswerable"]["accuracy"],
            ),
        )
        best_answerable_f1 = max(
            item["metrics"]["answerable"]["f1"] for item in sweep
        )
        # Prevent a superficially strong objective from selecting a degenerate
        # gate that answers almost nothing. A production candidate may trade at
        # most one absolute Answerable-F1 point from this configuration's best.
        safe_sweep = [
            item
            for item in sweep
            if item["metrics"]["answerable"]["f1"] >= best_answerable_f1 - 1.0
        ]
        best = max(
            safe_sweep,
            key=lambda item: (
                item["metrics"]["reader_priority_score"],
                item["metrics"]["overall"]["f1"],
                item["metrics"]["unanswerable"]["accuracy"],
            ),
        )
        config_results[name] = {
            "weights": config,
            "best_final_threshold": best["threshold"],
            "best_phrase_fallback_penalty": best["phrase_fallback_penalty"],
            "metrics": best["metrics"],
            "selection_constraint": {
                "max_answerable_f1_drop_points": 1.0,
                "best_answerable_f1": best_answerable_f1,
            },
            "unconstrained_best": {
                "final_threshold": unconstrained_best["threshold"],
                "phrase_fallback_penalty": unconstrained_best["phrase_fallback_penalty"],
                "metrics": unconstrained_best["metrics"],
            },
            "threshold_sweep": [
                {
                    "threshold": item["threshold"],
                    "phrase_fallback_penalty": item["phrase_fallback_penalty"],
                    "overall_f1": item["metrics"]["overall"]["f1"],
                    "answerable_f1": item["metrics"]["answerable"]["f1"],
                    "unanswerable_accuracy": item["metrics"]["unanswerable"]["accuracy"],
                    "answerable_empty_rate": item["metrics"]["answerable"]["predicted_empty_rate"],
                    "reader_priority_score": item["metrics"]["reader_priority_score"],
                }
                for item in sweep
            ],
            "predictions": best["predictions"],
        }

    winner_name, winner = max(
        config_results.items(),
        key=lambda item: (
            item[1]["metrics"]["reader_priority_score"],
            item[1]["metrics"]["overall"]["f1"],
            item[1]["weights"]["relation"],
        ),
    )
    payload = {
        "dataset": "UIT-ViQuAD2.0 validation stratified subset",
        "subset_size": len(records),
        "seed": args.seed,
        "top_k": args.top_k,
        "question_type_counts": dict(sorted(question_types.items())),
        "latency_ms": {
            "average": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50) if latencies else None,
            "p95": percentile(latencies, 0.95) if latencies else None,
        },
        "methodology": (
            "All configurations rescore the same checkpoint-specific candidate pool. "
            "Only invalid offsets, unsupported evidence, and impossible type/relation candidates "
            "are hard rejected; the final threshold is tuned after global ranking. Production "
            "selection constrains Answerable F1 to remain within one absolute point of the "
            "configuration's best, while the unconstrained optimum is reported separately."
        ),
        "winner": {
            "name": winner_name,
            "weights": winner["weights"],
            "best_final_threshold": winner["best_final_threshold"],
            "phrase_fallback_penalty": winner["best_phrase_fallback_penalty"],
            "reader_priority_score": winner["metrics"]["reader_priority_score"],
        },
        "configs": config_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["winner"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
