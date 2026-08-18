from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump all visible QA candidates for one real query")
    parser.add_argument("question")
    parser.add_argument("--retriever", default="bm25", choices=["bm25", "tfidf"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from backend.viqa_api import ask_question

    response = ask_question(
        {
            "question": args.question,
            "retriever": args.retriever,
            "reader": "phobert",
            "top_k": args.top_k,
        }
    )
    fields = (
        "rank",
        "retrieval_rank",
        "passage_id",
        "document_id",
        "title",
        "text",
        "retrieval_score_raw",
        "retrieval_score_normalized",
        "reader_answer",
        "reader_span_answer",
        "reader_score",
        "neural_reader_answer",
        "neural_reader_best_span",
        "neural_reader_start_score",
        "neural_reader_end_score",
        "reader_score_raw",
        "reader_null_score",
        "reader_score_margin",
        "reader_method",
        "fallback_sentence",
        "fallback_answer",
        "fallback_method",
        "fallback_phrase_score",
        "fallback_start",
        "fallback_end",
        "question_type",
        "answer_type_score",
        "answer_type_reason",
        "lexical_evidence",
        "relation_evidence",
        "relation_type",
        "relation_score",
        "phrase_quality",
        "evidence_supported",
        "ranking_score",
        "selection_status",
        "rejection_reason",
        "rejection_detail",
    )
    payload = {
        "question": response["question"],
        "question_type": response["question_type"],
        "answer": response["answer"],
        "has_answer": response["has_answer"],
        "reader_method": response["reader_method"],
        "answer_confidence": response["answer_confidence"],
        "selected_passage_id": response["selected_passage_id"],
        "scores": response["scores"],
        "relation_type": response.get("relation_type"),
        "relation_score": response.get("relation_score"),
        "lexical_evidence": response.get("lexical_evidence"),
        "relation_evidence": response.get("relation_evidence"),
        "rejection_reason": response["rejection_reason"],
        "processing_time_ms": response["processing_time_ms"],
        "reader_candidate": response.get("reader_candidate"),
        "fallback_candidate": response.get("fallback_candidate"),
        "selected_candidate": response.get("selected_candidate"),
        "candidates": [
            {
                **{field: passage.get(field) for field in fields},
                "candidate_pool": passage.get("candidates", []),
            }
            for passage in response["passages"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # ASCII escaping keeps the CLI portable on legacy Windows code pages; the
    # artifact above remains proper UTF-8 Vietnamese JSON.
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
