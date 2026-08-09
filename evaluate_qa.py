from __future__ import annotations

import argparse
import json
from pathlib import Path

from reader.evaluate import compute_exact, compute_f1


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data["data"]
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    raise ValueError("Evaluation data must be .json, .jsonl, or .parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate oracle Reader or end-to-end QA with EM/F1")
    parser.add_argument("data", type=Path)
    parser.add_argument("--mode", choices=["oracle", "end-to-end"], required=True)
    parser.add_argument("--model", default="models/reader/vinai_phobert-base-v2")
    parser.add_argument("--retriever", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    records = load_records(args.data)

    if args.mode == "oracle":
        from reader.predict import ReaderPredictor

        predictor = ReaderPredictor(args.model)
        predict = lambda row: predictor.predict(str(row["question"]), str(row["context"]))["answer"]
    else:
        from backend.viqa_api import ask_question

        predict = lambda row: ask_question({
            "question": str(row["question"]),
            "retriever": args.retriever,
            "reader": "phobert",
            "top_k": args.top_k,
        })["answer"] or ""

    exact, f1 = [], []
    for record in records:
        gold = str(record.get("answer") or record.get("answer_text") or "")
        prediction = str(predict(record) or "")
        exact.append(compute_exact(gold, prediction))
        f1.append(compute_f1(gold, prediction))
    total = max(1, len(records))
    print(json.dumps({"mode": args.mode, "EM": sum(exact) / total, "F1": sum(f1) / total, "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
