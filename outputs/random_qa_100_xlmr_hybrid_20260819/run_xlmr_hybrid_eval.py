from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_SAMPLE = ROOT / "outputs" / "random_qa_100_20260819" / "validation_sample_100_seed_20260819.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "random_qa_100_xlmr_hybrid_20260819"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def gold_answer(record: dict[str, Any]) -> tuple[str, bool]:
    answer = str(record.get("answer") or record.get("answer_text") or "")
    answer_start = record.get("answer_start")
    answerable = bool(answer) if answer_start is None else int(answer_start) >= 0 and bool(answer)
    return (answer if answerable else ""), answerable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    from functools import lru_cache

    import reader.question_type as question_type_module
    from reader.metrics import evaluate_predictions, exact_match, f1_score, normalize_answer

    records = load_jsonl(args.sample)[: args.limit]
    if not records:
        raise ValueError("No evaluation records")
    questions = [str(record["question"]).strip() for record in records]
    if len(set(questions)) != len(questions):
        raise ValueError("Sample contains duplicate question text; cached execution requires unique questions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"predictions_{len(records)}.jsonl"
    result_path = args.output_dir / f"combined_results_{len(records)}.json"
    if args.fresh:
        checkpoint_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)

    # Phase 1: preserve the pipeline's exact question-type decisions, then unload
    # the large zero-shot classifier before loading Dense and XLM-R.
    original_detect = question_type_module.detect_question_type
    question_types: dict[str, list[Any]] = {}
    print(f"[phase 1/3] Precomputing question types for {len(questions)} questions", flush=True)
    for index, question in enumerate(questions, start=1):
        question_types[question] = original_detect(question)
        if index % 10 == 0 or index == len(questions):
            print(f"Question types {index}/{len(questions)}", flush=True)

    question_type_module._classifier_pipeline = None
    gc.collect()

    @lru_cache(maxsize=512)
    def cached_detect(question: str):
        normalized = str(question or "").strip()
        if normalized in question_types:
            return question_types[normalized]
        return original_detect(normalized)

    question_type_module.detect_question_type = cached_detect

    # Import after patching so downstream modules bind the cached function.
    import backend.viqa_api as api
    import reader.fallback_policy as fallback_policy
    import reader.predict as reader_predict
    import reader.question_semantics as question_semantics

    api.detect_question_type = cached_detect
    fallback_policy.detect_question_type = cached_detect
    reader_predict.detect_question_type = cached_detect
    question_semantics.detect_question_type = cached_detect

    if api.ReaderManager.MODEL_FOLDERS.get("xlmr") != "xlm-roberta-large-viquad":
        raise RuntimeError("API reader alias 'xlmr' is not mapped to xlm-roberta-large-viquad")
    checkpoint = ROOT / "models" / "reader" / "xlm-roberta-large-viquad"
    if not (checkpoint / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing XLM-R checkpoint: {checkpoint}")

    # Phase 2: run the real hybrid retriever for every question, cache SearchHit
    # objects, and release the Dense model before the 2.2 GB reader is loaded.
    candidate_count = api.PIPELINE_CONFIG.candidate_count(10)
    retrieval_cache: dict[str, list[Any]] = {}
    retrieval_latency_ms: dict[str, float] = {}
    print(
        f"[phase 2/3] Precomputing hybrid retrieval (candidate_count={candidate_count})",
        flush=True,
    )
    for index, question in enumerate(questions, start=1):
        started = time.perf_counter()
        retrieval_cache[question] = api.INDEX.retrieve(question, "hybrid", candidate_count)
        retrieval_latency_ms[question] = (time.perf_counter() - started) * 1000
        if index == 1:
            if not api.DENSE_SCORER.is_available or api.DENSE_SCORER._model is None:
                raise RuntimeError("Hybrid retriever fell back to BM25; Dense model is not active")
            if api.DENSE_SCORER._passage_embeddings is None:
                raise RuntimeError("Hybrid retriever has no Dense passage embeddings")
        if index % 10 == 0 or index == len(questions):
            print(f"Hybrid retrieval {index}/{len(questions)}", flush=True)

    hybrid_verified = bool(
        api.DENSE_SCORER.is_available
        and api.DENSE_SCORER._model is not None
        and api.DENSE_SCORER._passage_embeddings is not None
    )
    api.DENSE_SCORER._model = None
    api.DENSE_SCORER._passage_embeddings = None
    gc.collect()

    original_retrieve = api.INDEX.retrieve

    def cached_retrieve(question: str, method: str, top_k: int):
        if method != "hybrid":
            return original_retrieve(question, method, top_k)
        if question not in retrieval_cache:
            raise KeyError(f"No cached hybrid retrieval for question: {question}")
        return retrieval_cache[question][:top_k]

    api.INDEX.retrieve = cached_retrieve

    # Resume only prediction rows. Semantics and hybrid retrieval are deliberately
    # recomputed on restart so the configuration checks cannot be bypassed.
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[str(item["id"])] = item

    print(
        f"[phase 3/3] XLM-R reader predictions; resumed={len(completed)}/{len(records)}",
        flush=True,
    )
    predictions: list[dict[str, Any]] = []
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint_handle:
        for index, record in enumerate(records, start=1):
            record_id = str(record.get("id", index))
            if record_id in completed:
                predictions.append(completed[record_id])
                continue

            question = str(record["question"]).strip()
            gold, answerable = gold_answer(record)
            response = api.ask_question(
                {
                    "question": question,
                    "retriever": "hybrid",
                    "reader": "xlmr",
                    "top_k": 10,
                }
            )
            if response.get("retriever") != "hybrid" or response.get("reader") != "xlmr":
                raise RuntimeError(
                    f"Unexpected response config: retriever={response.get('retriever')}, "
                    f"reader={response.get('reader')}"
                )

            prediction = str(response.get("answer") or "")
            selected_id = response.get("selected_passage_id")
            selected = next(
                (
                    item
                    for item in response.get("passages", [])
                    if item.get("passage_id") == selected_id
                ),
                None,
            )
            normalized_gold = normalize_answer(gold)
            retrieval_hit = None
            if answerable:
                retrieval_hit = any(
                    normalized_gold
                    and normalized_gold in normalize_answer(item.get("text", ""))
                    for item in response.get("passages", [])
                )

            row = {
                "id": record_id,
                "question": question,
                "gold_answer": gold,
                "predicted_answer": prediction,
                "is_answerable": answerable,
                "exact_match": exact_match(gold, prediction),
                "f1": f1_score(gold, prediction),
                "reader_score": float(selected.get("reader_score", 0.0)) if selected else 0.0,
                "ranking_score": selected.get("ranking_score") if selected else response.get("scores", {}).get("ranking"),
                "answer_type_score": selected.get("answer_type_score") if selected else response.get("scores", {}).get("answer_type"),
                "question_type": (question_types[question][0].value if question_types[question] else "GENERAL"),
                "reader_method": str(selected.get("reader_method")) if selected else "no_answer",
                "retrieval_hit": retrieval_hit,
                "latency_ms": float(retrieval_latency_ms[question]) + float(response.get("processing_time_ms", 0.0)),
                "retriever": "hybrid",
                "reader": "xlmr",
                "checkpoint": str(checkpoint),
            }
            checkpoint_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            checkpoint_handle.flush()
            predictions.append(row)
            if index % 5 == 0 or index == len(records):
                print(f"XLM-R evaluated {index}/{len(records)}", flush=True)

    by_id = {str(item["id"]): item for item in predictions}
    ordered = [by_id[str(record.get("id", index))] for index, record in enumerate(records, start=1)]
    if len(ordered) != len(records) or len(by_id) != len(records):
        raise RuntimeError("Prediction count or IDs do not match the sample")

    metrics = evaluate_predictions(ordered)
    latencies = [float(item["latency_ms"]) for item in ordered]
    metrics["latency_ms"] = {
        "average": statistics.fmean(latencies),
        "p50": statistics.median(latencies),
        "max": max(latencies),
    }
    payload = {
        "dataset": "data/processed/viquad_val_clean.parquet",
        "sample_size": len(records),
        "seed": 20260819,
        "mode": "end-to-end",
        "retriever": "hybrid",
        "dense_model": api.DENSE_MODEL_NAME,
        "reader": "xlmr",
        "checkpoint": str(checkpoint),
        "top_k": 10,
        "hybrid_dense_verified": hybrid_verified,
        "metrics": metrics,
        "predictions": ordered,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(result_path),
                "count": len(ordered),
                "correct": sum(int(item["exact_match"]) for item in ordered),
                "f1": metrics["overall"]["f1"],
                "hybrid_dense_verified": hybrid_verified,
                "reader": "xlmr",
                "checkpoint": str(checkpoint),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
