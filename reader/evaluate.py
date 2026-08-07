import os
import sys
import json
import argparse
import string
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collections import Counter
from tqdm import tqdm
from reader.data_utils import load_qa_dataset
from reader.predict import ReaderPredictor

def normalize_answer(s: str) -> str:
    """
    Lowercases, removes punctuation, removes underscores, and normalizes whitespace.
    """
    if not s:
        return ""
    
    exclude = set(string.punctuation) | {'“', '”', '‘', '’', '–', '—', '…'}
    
    # Lowercase & remove underscores (from word segmentation)
    s = s.lower().replace("_", " ")
    
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in exclude)
    
    # Normalize whitespaces
    return " ".join(s.split())

def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    
    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0
        
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def categorize_question(question: str) -> str:
    q_lower = question.lower()
    if "ai " in q_lower or q_lower.startswith("ai") or "là ai" in q_lower:
        return "Ai (Person)"
    elif "ở đâu" in q_lower or "tại đâu" in q_lower or "nơi nào" in q_lower:
        return "Ở đâu (Location)"
    elif any(k in q_lower for k in ["khi nào", "năm nào", "ngày nào", "tháng nào", "bao giờ"]):
        return "Khi nào (Time)"
    elif "bao nhiêu" in q_lower or "mấy" in q_lower or "phần trăm" in q_lower or "%" in q_lower:
        return "Bao nhiêu (Quantity)"
    elif "tại sao" in q_lower or "vì sao" in q_lower or "do đâu" in q_lower:
        return "Tại sao (Reason)"
    elif "như thế nào" in q_lower or "ra sao" in q_lower:
        return "Như thế nào (Manner)"
    elif "gì" in q_lower or "nào" in q_lower:
        return "Cái gì / Nào (What/Which)"
    else:
        return "Khác (Other)"

def evaluate(model_path: str, data_variant: str = "auto", subset_size: int = -1, use_cpu: bool = False, output_file: str = None):
    if data_variant == "auto":
        data_variant = "segmented" if "phobert" in model_path.lower() else "clean"
        
    val_file = f"data/processed/viquad_val_{data_variant}.parquet"
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation file not found: {val_file}")
        
    print(f"Loading evaluation dataset: {val_file}")
    dataset = load_qa_dataset(val_file, data_variant=data_variant, subset_size=subset_size)
    print(f"Loaded {len(dataset)} samples for evaluation.")
    
    # Initialize predictor
    predictor = ReaderPredictor(model_path, use_cpu=use_cpu)
    
    overall_em = []
    overall_f1 = []
    
    has_ans_em = []
    has_ans_f1 = []
    
    no_ans_em = [] # accuracy on unanswerable
    
    # Breakdown by question type
    type_metrics = {}
    
    # Breakdown of answerable vs unanswerable
    # We will record raw predictions for threshold tuning or debugging
    predictions_log = []
    
    print("Running evaluation loop...")
    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        context = item["context"]
        gold_answer = item["answer_text"]
        is_answerable = item["answer_start"] >= 0
        
        # In case of segmented variant, the gold answer is already word-segmented.
        # But we normalized it inside compute functions, so it handles spacing/underscores automatically.
        if gold_answer is None or not is_answerable:
            gold_answer = ""
            
        pred_res = predictor.predict(question, context)
        pred_answer = pred_res["answer"]
        
        em = compute_exact(gold_answer, pred_answer)
        f1 = compute_f1(gold_answer, pred_answer)
        
        overall_em.append(em)
        overall_f1.append(f1)
        
        if is_answerable:
            has_ans_em.append(em)
            has_ans_f1.append(f1)
        else:
            no_ans_em.append(em) # If gold is empty, and prediction is empty -> EM = 1 (correct no-answer)
            
        q_type = categorize_question(question)
        if q_type not in type_metrics:
            type_metrics[q_type] = {"em": [], "f1": [], "count": 0}
        type_metrics[q_type]["em"].append(em)
        type_metrics[q_type]["f1"].append(f1)
        type_metrics[q_type]["count"] += 1
        
        predictions_log.append({
            "id": item["id"],
            "question": question,
            "gold": gold_answer,
            "pred": pred_answer,
            "em": em,
            "f1": f1,
            "is_answerable": is_answerable,
            "confidence": pred_res["confidence"],
            "score": pred_res["score"]
        })
        
    mean_em = sum(overall_em) / len(overall_em) * 100
    mean_f1 = sum(overall_f1) / len(overall_f1) * 100
    
    mean_has_ans_em = sum(has_ans_em) / len(has_ans_em) * 100 if has_ans_em else 0.0
    mean_has_ans_f1 = sum(has_ans_f1) / len(has_ans_f1) * 100 if has_ans_f1 else 0.0
    
    mean_no_ans_em = sum(no_ans_em) / len(no_ans_em) * 100 if no_ans_em else 0.0
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Overall Exact Match (EM): {mean_em:.2f}%")
    print(f"Overall F1 Score:         {mean_f1:.2f}%")
    print(f"Total Samples evaluated:  {len(overall_em)}")
    print(f"----------------------------------------------------")
    print(f"Answerable Questions EM:  {mean_has_ans_em:.2f}% (Count: {len(has_ans_em)})")
    print(f"Answerable Questions F1:  {mean_has_ans_f1:.2f}%")
    print(f"Unanswerable (No Ans) EM: {mean_no_ans_em:.2f}% (Count: {len(no_ans_em)})")
    print(f"====================================================\n")
    
    print("Question Type Breakdown:")
    print(f"{'Question Type':<30} | {'Count':<5} | {'EM (%)':<8} | {'F1 (%)':<8}")
    print("-" * 60)
    for q_type, data in sorted(type_metrics.items()):
        type_em = sum(data["em"]) / len(data["em"]) * 100
        type_f1 = sum(data["f1"]) / len(data["f1"]) * 100
        print(f"{q_type:<30} | {data['count']:<5} | {type_em:<8.2f} | {type_f1:<8.2f}")
    print("=" * 60)
    
    results = {
        "overall": {
            "em": mean_em,
            "f1": mean_f1,
            "count": len(overall_em)
        },
        "answerable": {
            "em": mean_has_ans_em,
            "f1": mean_has_ans_f1,
            "count": len(has_ans_em)
        },
        "unanswerable": {
            "em": mean_no_ans_em,
            "count": len(no_ans_em)
        },
        "breakdown": {
            q_type: {
                "em": sum(data["em"]) / len(data["em"]) * 100,
                "f1": sum(data["f1"]) / len(data["f1"]) * 100,
                "count": data["count"]
            } for q_type, data in type_metrics.items()
        }
    }
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": results,
                "predictions": predictions_log
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved evaluation results to {output_file}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Reader EM/F1 performance")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_variant", type=str, default="auto", choices=["auto", "clean", "segmented"])
    parser.add_argument("--subset_size", type=int, default=-1, help="Subset size to evaluate on")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU use")
    parser.add_argument("--output_file", type=str, default="results/reader_eval_results.json", help="Path to save output results JSON")
    
    args = parser.parse_args()
    evaluate(args.model_path, args.data_variant, args.subset_size, args.use_cpu, args.output_file)

if __name__ == "__main__":
    main()
