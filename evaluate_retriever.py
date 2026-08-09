from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.viqa_api import INDEX, normalize_text


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


def relevant(record: dict, hit) -> bool:
    passage = hit.passage.metadata
    passage_id = str(record.get("passage_id") or "")
    document_id = str(record.get("document_id") or "")
    answer = str(record.get("answer") or record.get("answer_text") or "").strip()
    return bool(
        (passage_id and passage.passage_id == passage_id)
        or (document_id and passage.document_id == document_id)
        or (answer and normalize_text(answer) in normalize_text(passage.text))
    )


def evaluate(records: list[dict], method: str, k_values: list[int]) -> dict[str, float]:
    hits_at_k = {k: 0 for k in k_values}
    reciprocal_ranks: list[float] = []
    max_k = max(k_values)
    for record in records:
        question = str(record.get("question") or "").strip()
        hits = INDEX.retrieve(question, method, max_k)
        rank = next((index for index, hit in enumerate(hits, start=1) if relevant(record, hit)), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in k_values:
            hits_at_k[k] += int(rank is not None and rank <= k)
    total = max(1, len(records))
    return {
        **{f"Recall@{k}": hits_at_k[k] / total for k in k_values},
        "MRR": sum(reciprocal_ranks) / total,
        "count": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate passage retrieval with Recall@k and MRR")
    parser.add_argument("data", type=Path)
    parser.add_argument("--method", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate(load_records(args.data), args.method, sorted(set(args.k)))
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
