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

from reader.metrics import evaluate_predictions, exact_match, normalize_answer
from reader.question_type import detect_question_type


ROOT = Path(__file__).resolve().parent
WEIGHT_CONFIGS = {
    "A_retriever70_reader30": {
        "retriever": 0.70,
        "reader": 0.30,
        "answer_type": 0.0,
        "type_gate": False,
        "fallback_penalty": 1.0,
    },
    "B_retriever60_reader40": {
        "retriever": 0.60,
        "reader": 0.40,
        "answer_type": 0.0,
        "type_gate": False,
        "fallback_penalty": 1.0,
    },
    "C_retriever50_reader50": {
        "retriever": 0.50,
        "reader": 0.50,
        "answer_type": 0.0,
        "type_gate": False,
        "fallback_penalty": 1.0,
    },
    "D_retriever50_reader30_type20": {
        "retriever": 0.50,
        "reader": 0.30,
        "answer_type": 0.20,
        "type_gate": True,
        "fallback_penalty": 0.60,
    },
    "E_retriever40_reader40_type20": {
        "retriever": 0.40,
        "reader": 0.40,
        "answer_type": 0.20,
        "type_gate": True,
        "fallback_penalty": 0.60,
    },
}


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


def candidate_score(candidate: dict[str, Any], config: dict[str, Any]) -> float:
    reader_score = float(candidate.get("reader_score") or 0.0)
    if candidate.get("reader_method") == "sentence_fallback":
        reader_score *= float(config["fallback_penalty"])
    return (
        float(config["retriever"]) * float(candidate.get("retrieval_score_normalized") or 0.0)
        + float(config["reader"]) * reader_score
        + float(config["answer_type"]) * float(candidate.get("answer_type_score") or 0.0)
    )


def candidate_passes(candidate: dict[str, Any], ranking_score: float, config: dict[str, Any]) -> bool:
    from backend.viqa_api import (
        MIN_ANSWER_TYPE_SCORE,
        MIN_FALLBACK_ANSWER_TYPE_SCORE,
        MIN_RANKING_SCORE,
        MIN_READER_SCORE,
    )

    if not candidate.get("reader_answer"):
        return False
    if float(candidate.get("reader_score") or 0.0) < MIN_READER_SCORE:
        return False
    if not config["type_gate"]:
        return True
    if not candidate.get("evidence_supported"):
        return False
    minimum_type = (
        MIN_FALLBACK_ANSWER_TYPE_SCORE
        if candidate.get("reader_method") == "sentence_fallback"
        else MIN_ANSWER_TYPE_SCORE
    )
    return (
        float(candidate.get("answer_type_score") or 0.0) >= minimum_type
        and ranking_score >= MIN_RANKING_SCORE
    )


def select_candidate(candidates: list[dict[str, Any]], config: dict[str, Any]):
    scored = sorted(
        ((candidate_score(candidate, config), candidate) for candidate in candidates),
        key=lambda item: item[0],
        reverse=True,
    )
    for ranking_score, candidate in scored:
        if candidate_passes(candidate, ranking_score, config):
            return candidate, ranking_score
    return None, (scored[0][0] if scored else 0.0)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark QA reranking weights on a validation subset")
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "reranking_validation_subset.json",
    )
    args = parser.parse_args()

    from backend.viqa_api import ask_question

    records = load_stratified_subset(args.subset_size, args.seed)
    predictions: dict[str, list[dict[str, Any]]] = {name: [] for name in WEIGHT_CONFIGS}
    ranking_counts = {
        name: Counter(
            {
                "gold_answer_passage_available_top10": 0,
                "gold_passage_selected": 0,
                "wrong_passage_selected": 0,
                "no_answer_correctly_emitted": 0,
                "false_positive_answer_emitted": 0,
                "high_score_wrong_answer": 0,
            }
        )
        for name in WEIGHT_CONFIGS
    }
    question_types: Counter[str] = Counter()
    latencies: list[float] = []

    for index, record in enumerate(records, start=1):
        question = str(record["question"])
        gold = str(record.get("answer_text") or "")
        raw_answer_start = record.get("answer_start", -1)
        answer_start = int(raw_answer_start if raw_answer_start is not None else -1)
        answerable = answer_start >= 0 and bool(gold)
        if not answerable:
            gold = ""
        question_types[detect_question_type(question).value] += 1

        started = time.perf_counter()
        response = ask_question(
            {"question": question, "retriever": "bm25", "reader": "phobert", "top_k": args.top_k}
        )
        latencies.append((time.perf_counter() - started) * 1000)
        candidates = list(response.get("passages") or [])
        gold_normalized = normalize_answer(gold)
        available = bool(gold_normalized) and any(
            gold_normalized in normalize_answer(candidate.get("text", "")) for candidate in candidates
        )

        for name, config in WEIGHT_CONFIGS.items():
            selected, ranking_score = select_candidate(candidates, config)
            prediction = str(selected.get("reader_answer") or "") if selected else ""
            row = {
                "id": str(record.get("id", index)),
                "gold_answer": gold,
                "predicted_answer": prediction,
                "is_answerable": answerable,
            }
            predictions[name].append(row)
            counts = ranking_counts[name]
            counts["gold_answer_passage_available_top10"] += int(answerable and available)
            selected_has_gold = bool(
                selected
                and gold_normalized
                and gold_normalized in normalize_answer(selected.get("text", ""))
            )
            counts["gold_passage_selected"] += int(answerable and selected_has_gold)
            counts["wrong_passage_selected"] += int(answerable and selected is not None and not selected_has_gold)
            counts["no_answer_correctly_emitted"] += int(not answerable and not prediction)
            counts["false_positive_answer_emitted"] += int(not answerable and bool(prediction))
            counts["high_score_wrong_answer"] += int(
                bool(prediction) and exact_match(gold, prediction) == 0 and ranking_score >= 0.8
            )

        if index % 10 == 0 or index == len(records):
            print(f"Evaluated {index}/{len(records)}", flush=True)

    result_configs = {}
    for name, config in WEIGHT_CONFIGS.items():
        metrics = evaluate_predictions(predictions[name])
        metrics["ranking"] = dict(ranking_counts[name])
        metrics["ranking"]["high_score_wrong_answer_rate"] = (
            100.0 * ranking_counts[name]["high_score_wrong_answer"] / len(records)
        )
        result_configs[name] = {"weights": config, "metrics": metrics}

    payload = {
        "dataset": "UIT-ViQuAD2.0 validation stratified subset",
        "subset_size": len(records),
        "seed": args.seed,
        "top_k": args.top_k,
        "question_type_counts": dict(sorted(question_types.items())),
        "latency_ms": {
            "average": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
        "methodology": (
            "Reader inference is run once per question. A-E are post-hoc rescored on the same returned "
            "Top-10 candidates. A-C reproduce legacy score/gate semantics; D-E add answer-type gates "
            "and a sentence-fallback penalty. This subset benchmark is not the full validation benchmark."
        ),
        "configs": result_configs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
