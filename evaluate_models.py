import sys
import time
import json
from pathlib import Path

# Add root path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_qa import load_validation, _gold
from reader.predict import ReaderPredictor
from backend.llm_reader import LocalLLMReader
from reader.metrics import normalize_answer

def get_em_f1(pred: str, gold: str) -> tuple[float, float]:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    # Exact match
    em = 1.0 if pred_norm == gold_norm else 0.0
    
    # F1
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    
    if not pred_tokens or not gold_tokens:
        return em, (1.0 if not pred_tokens and not gold_tokens else 0.0)
        
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        f1 = 0.0
    else:
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
    return em, f1

def main():
    print("Loading data...")
    try:
        records = load_validation()
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return
        
    # Filter answerable questions
    records = [r for r in records if _gold(r)[1]]
    records = records[:100] # Take first 100
    print(f"Loaded {len(records)} test samples.")
    
    models_to_evaluate = [
        {"name": "phobert", "type": "reader", "path": "models/reader/vinai_phobert-base-v2"},
        {"name": "xlmr", "type": "reader", "path": "models/reader/xlm-roberta-large-viquad"},
        {"name": "llm", "type": "llm", "path": "LiquidAI/LFM2.5-2.6B"}
    ]
    
    results = {}
    
    for model_info in models_to_evaluate:
        name = model_info["name"]
        print(f"\nEvaluating {name}...")
        
        if model_info["type"] == "reader":
            predictor = ReaderPredictor(model_info["path"])
        else:
            try:
                predictor = LocalLLMReader(model_info["path"])
            except Exception as e:
                print(f"Skipping {name} due to initialization error: {e}")
                continue
            
        em_scores = []
        f1_scores = []
        latencies = []
        
        for i, record in enumerate(records):
            question = str(record["question"])
            context = str(record.get("context") or "")
            gold, _ = _gold(record)
            
            start = time.perf_counter()
            if model_info["type"] == "reader":
                output = predictor.predict(question, context)
                pred = str(output.get("answer") or "")
            else:
                output = predictor.predict_rag(question, [context])
                pred = str(output.get("text") or "")
            
            elapsed = time.perf_counter() - start
            
            em, f1 = get_em_f1(pred, gold)
            em_scores.append(em)
            f1_scores.append(f1)
            latencies.append(elapsed)
            
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{len(records)}...")
            
        avg_em = sum(em_scores) / len(em_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)
        avg_lat = sum(latencies) / len(latencies) * 1000 # ms
        
        results[name] = {
            "Exact Match": round(avg_em * 100, 2),
            "F1 Score": round(avg_f1 * 100, 2),
            "Average Latency (ms)": round(avg_lat, 2)
        }
        
    print("\n\n--- RESULTS ---")
    print(json.dumps(results, indent=2))
    
    Path("results").mkdir(exist_ok=True)
    with open("results/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
