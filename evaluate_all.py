import json
import time
from pathlib import Path
import pandas as pd

from backend.viqa_api import INDEX, normalize_text

def load_records(path: Path) -> list[dict]:
    return pd.read_parquet(path).to_dict("records")

def get_answer(record):
    answer = str(record.get("answer") or record.get("answer_text") or "").strip()
    if not answer and "answers" in record and record["answers"]:
        ans = record["answers"]
        if isinstance(ans, dict) and "text" in ans and len(ans["text"]) > 0:
            answer = str(ans["text"][0]).strip()
        elif isinstance(ans, list) and len(ans) > 0 and isinstance(ans[0], dict) and "text" in ans[0]:
            answer = str(ans[0]["text"]).strip()
    return answer

def relevant(record: dict, hit) -> bool:
    passage = hit.passage.metadata
    passage_id = str(record.get("passage_id") or "")
    document_id = str(record.get("document_id") or "")
    title = str(record.get("title") or "")
    answer = get_answer(record)
    
    if not answer and not passage_id and not document_id and not title:
        return False
        
    return bool(
        (passage_id and passage.passage_id == passage_id)
        or (document_id and passage.document_id == document_id)
        or (answer and normalize_text(answer) in normalize_text(passage.text))
        or (title and normalize_text(title) == normalize_text(passage.title))
    )

def evaluate(records: list[dict], method: str, k_values: list[int]) -> dict:
    hits_at_k = {k: 0 for k in k_values}
    reciprocal_ranks = []
    max_k = max(k_values)
    start_time = time.time()
    
    count = 0
    # Limit to 500 records to make this run reasonably fast while being realistic enough for demo
    for record in records:
        if count >= 500:
            break
            
        question = str(record.get("question") or "").strip()
        if not question:
            continue
            
        hits = INDEX.retrieve(question, method, max_k)
        rank = next((index for index, hit in enumerate(hits, start=1) if relevant(record, hit)), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in k_values:
            hits_at_k[k] += int(rank is not None and rank <= k)
            
        count += 1
        
    total = max(1, count)
    end_time = time.time()
    
    return {
        "set": "Test",
        "num_questions": count,
        "time_sec": round(end_time - start_time, 2),
        "qps": round(count / max(0.1, end_time - start_time), 2),
        "MRR": round(sum(reciprocal_ranks) / total, 4),
        **{f"Recall@{k}": round(hits_at_k[k] / total, 4) for k in k_values}
    }

def main():
    data_path = Path("data/raw/viquad2_validation.parquet")
    print(f"Loading {data_path}...")
    records = load_records(data_path)
    
    methods = ["bm25", "tfidf", "dense", "hybrid"]
    k_values = [1, 3, 5, 10]
    
    results = {}
    
    for method in methods:
        print(f"Evaluating {method}...")
        metrics = evaluate(records, method, k_values)
        print(f"{method} results: {metrics}")
        results[method] = {"test": metrics}
        
    output_path = Path("results/retriever_eval_all.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
