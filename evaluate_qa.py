from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from backend.config import load_pipeline_config
from reader.evaluate import build_error_rows
from reader.metrics import evaluate_predictions, normalize_answer
from reader.question_type import detect_question_type


ROOT = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_validation() -> list[dict[str, Any]]:
    path = ROOT / "data" / "processed" / "viquad_val_clean.parquet"
    try:
        import polars as pl

        return pl.read_parquet(path).to_dicts()
    except ImportError:  # pragma: no cover
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def _write_error_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gold(record: dict[str, Any]) -> tuple[str, bool]:
    answer = str(record.get("answer") or record.get("answer_text") or "")
    answer_start = record.get("answer_start")
    answerable = bool(answer) if answer_start is None else int(answer_start) >= 0 and bool(answer)
    return (answer if answerable else ""), answerable


def main() -> None:
    config = load_pipeline_config()
    parser = argparse.ArgumentParser(description="Post-processed oracle Reader or end-to-end QA evaluation")
    parser.add_argument("data", type=Path, nargs="?", help="JSONL fixture; omit with --validation")
    parser.add_argument("--validation", action="store_true", help="Use all 3,814 labeled validation questions")
    parser.add_argument("--mode", choices=["oracle", "end-to-end"], required=True)
    parser.add_argument("--model", default=str(config.reader_checkpoint))
    parser.add_argument("--retriever", choices=["bm25", "tfidf"], default=config.default_retriever)
    parser.add_argument("--top-k", type=int, default=config.default_top_k)
    parser.add_argument("--subset-size", type=int, default=-1, help="Smoke/subset benchmark only")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.validation:
        records = load_validation()
        if len(records) != 3814:
            raise ValueError(f"Expected 3,814 validation questions, found {len(records)}")
    elif args.data:
        records = load_jsonl(args.data)
    else:
        parser.error("Provide a JSONL data path or --validation")
    if not records:
        raise ValueError("No evaluation records found")
    if args.subset_size > 0:
        records = records[: args.subset_size]
    if not any(_gold(record)[1] for record in records):
        raise ValueError(
            "This dataset has no answer labels. Reader F1/EM may only be computed on labeled validation data."
        )

    if args.mode == "oracle":
        from reader.predict import ReaderPredictor

        predictor = ReaderPredictor(args.model)
    else:
        from backend.viqa_api import ask_question

    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    reader_methods: Counter[str] = Counter()
    retrieval_hits = 0
    retrieval_evaluated = 0
    ranking_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()

    for index, record in enumerate(records, start=1):
        question = str(record["question"])
        context = str(record.get("context") or "")
        gold, answerable = _gold(record)
        question_type = detect_question_type(question)[0].value
        question_type_counts[question_type] += 1
        started = time.perf_counter()
        if args.mode == "oracle":
            output = predictor.predict(question, context)
            prediction = str(output.get("answer") or "")
            method = "neural_reader" if prediction else "no_answer"
            reader_score = float(output.get("confidence", 0.0))
            answer_confidence = None
            ranking_score = None
            answer_type_score = None
            best_span_score = output.get("best_span_score")
            no_answer_score = output.get("no_answer_score")
            score_margin = output.get("score_margin")
            retrieval_hit = None
        else:
            response = ask_question(
                {
                    "question": question,
                    "retriever": args.retriever,
                    "reader": "phobert",
                    "top_k": args.top_k,
                }
            )
            prediction = str(response.get("answer") or "")
            selected_id = response.get("selected_passage_id")
            selected = next(
                (item for item in response.get("passages", []) if item.get("passage_id") == selected_id),
                None,
            )
            method = str(selected.get("reader_method")) if selected else "no_answer"
            reader_score = float(selected.get("reader_score", 0.0)) if selected else 0.0
            answer_confidence = response.get("answer_confidence")
            ranking_score = selected.get("ranking_score") if selected else response.get("scores", {}).get("ranking")
            answer_type_score = selected.get("answer_type_score") if selected else response.get("scores", {}).get("answer_type")
            best_span_score = selected.get("reader_score_raw") if selected else None
            no_answer_score = (
                (selected.get("reader_null_score") - selected.get("reader_score_raw"))
                if selected
                and selected.get("reader_null_score") is not None
                and selected.get("reader_score_raw") is not None
                else None
            )
            score_margin = selected.get("reader_score_margin") if selected else None
            retrieval_hit = None
            if answerable and context:
                retrieval_evaluated += 1
                normalized_gold = normalize_answer(gold)
                retrieval_hit = any(
                    bool(normalized_gold)
                    and normalized_gold in normalize_answer(item.get("text", ""))
                    for item in response.get("passages", [])
                )
                retrieval_hits += int(retrieval_hit)
                ranking_counts["gold_answer_passage_available_top10"] += int(retrieval_hit)
                selected_has_gold = bool(
                    selected
                    and normalized_gold
                    and normalized_gold in normalize_answer(selected.get("text", ""))
                )
                ranking_counts["gold_passage_selected"] += int(selected_has_gold)
                ranking_counts["wrong_passage_selected"] += int(selected is not None and not selected_has_gold)
            if not answerable:
                ranking_counts["no_answer_correctly_emitted"] += int(not prediction)
                ranking_counts["false_positive_answer_emitted"] += int(bool(prediction))
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)
        reader_methods[method] += 1
        predictions.append(
            {
                "id": str(record.get("id", index)),
                "question": question,
                "gold_answer": gold,
                "predicted_answer": prediction,
                "is_answerable": answerable,
                "reader_score": reader_score,
                "answer_confidence": answer_confidence,
                "ranking_score": ranking_score,
                "answer_type_score": answer_type_score,
                "question_type": question_type,
                "best_span_score": best_span_score,
                "no_answer_score": no_answer_score,
                "score_margin": score_margin,
                "reader_method": method,
                "retrieval_hit": retrieval_hit,
                "latency_ms": elapsed,
            }
        )
        if index % 100 == 0 or index == len(records):
            print(f"Evaluated {index}/{len(records)}", flush=True)

    metrics = evaluate_predictions(predictions)
    metrics["latency_ms"] = {
        "average": statistics.fmean(latencies),
        "p50": _percentile(latencies, 0.50),
        "p95": _percentile(latencies, 0.95),
    }
    metrics["reader_methods"] = {
        method: {"count": count, "rate": 100.0 * count / len(predictions)}
        for method, count in sorted(reader_methods.items())
    }
    if args.mode == "end-to-end":
        metrics["retrieval_hit_rate"] = (
            100.0 * retrieval_hits / retrieval_evaluated if retrieval_evaluated else 0.0
        )
        metrics["retrieval_hit_evaluated"] = retrieval_evaluated
        high_score_wrong = sum(
            bool(row["predicted_answer"])
            and row["ranking_score"] is not None
            and float(row["ranking_score"]) >= 0.8
            and normalize_answer(row["predicted_answer"]) != normalize_answer(row["gold_answer"])
            for row in predictions
        )
        ranking_counts["high_score_wrong_answer"] = high_score_wrong
        metrics["ranking"] = dict(ranking_counts)
        metrics["ranking"]["high_score_wrong_answer_rate"] = 100.0 * high_score_wrong / len(predictions)
        metrics["question_type_counts"] = dict(sorted(question_type_counts.items()))

    payload = {
        "dataset": "validation" if args.validation else str(args.data),
        "mode": args.mode,
        "checkpoint": args.model,
        "retriever": args.retriever if args.mode == "end-to-end" else None,
        "top_k": args.top_k if args.mode == "end-to-end" else None,
        "metrics": metrics,
        "predictions": predictions,
    }
    output = args.output
    if output is None and args.validation:
        output = ROOT / "results" / (
            "end_to_end_eval_results.json" if args.mode == "end-to-end" else "oracle_reader_eval_results.json"
        )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        error_rows = build_error_rows(predictions)
        for error_row, prediction in zip(error_rows, predictions):
            error_row["reader_method"] = prediction["reader_method"]
            error_row["retrieval_hit"] = prediction["retrieval_hit"]
            error_row["latency_ms"] = prediction["latency_ms"]
        _write_error_csv(output.with_name(f"{output.stem}_errors.csv"), error_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
