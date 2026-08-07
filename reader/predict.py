import os
import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering
from reader.data_utils import get_tokenizer, find_char_span

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

class ReaderPredictor:
    def __init__(self, model_path_or_name: str, use_cpu: bool = False):
        self.device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
        print(f"Loading reader model from {model_path_or_name} to {self.device}...")
        
        self.tokenizer = get_tokenizer(model_path_or_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path_or_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Detect if PhoBERT is used based on name/path
        self.is_phobert = "phobert" in model_path_or_name.lower()
        print(f"Model type: {'PhoBERT (requires word-segmented input)' if self.is_phobert else 'Standard/XLM-R'}")
        
    def predict(self, question: str, context: str, max_seq_len: int = 384, doc_stride: int = 128, max_answer_len: int = 30, no_answer_threshold: float = -8.0):
        # Cap max_seq_len to model limits (PhoBERT max is 256)
        if self.is_phobert:
            max_seq_len = min(max_seq_len, 256)
            doc_stride = min(doc_stride, 32)
            
        # 1. Word segmentation if model is PhoBERT
        raw_context = context
        if self.is_phobert:
            if not HAS_PYVI:
                raise ImportError("pyvi is required to run inference on PhoBERT. Please install it using: pip install pyvi")
            segmented_question = ViTokenizer.tokenize(question)
            segmented_context = ViTokenizer.tokenize(context)
        else:
            segmented_question = question
            segmented_context = context
            
        # 2. Tokenize inputs
        inputs = self.tokenizer(
            segmented_question,
            segmented_context,
            truncation="only_second",
            max_length=max_seq_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 3. Model forward pass
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()
        offset_mapping = inputs["offset_mapping"]
        
        best_score = -float("inf")
        best_span_segmented = ""
        # Track no_answer_score from the SAME chunk as best_score (per-chunk comparison)
        best_chunk_no_answer_score = -float("inf")
        
        # Iterate over all sliding window chunks
        for i in range(len(start_logits)):
            sequence_ids = inputs.sequence_ids(i)
            s_logits = start_logits[i]
            e_logits = end_logits[i]
            offsets = offset_mapping[i]
            
            # Find context start and end indices in the token list
            context_start = 0
            while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
                context_start += 1
                
            context_end = len(sequence_ids) - 1
            while context_end >= 0 and sequence_ids[context_end] != 1:
                context_end -= 1
                
            if context_start > context_end:
                continue
                
            # CLS index represents no answer for this chunk
            chunk_no_answer_score = -float("inf")
            try:
                cls_idx = inputs["input_ids"][i].tolist().index(self.tokenizer.cls_token_id)
                chunk_no_answer_score = s_logits[cls_idx] + e_logits[cls_idx]
            except ValueError:
                pass
                
            # Select top-N candidates
            n_best = 20
            start_indexes = np.argsort(s_logits)[::-1][:n_best]
            end_indexes = np.argsort(e_logits)[::-1][:n_best]
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip indexes that are out of context
                    if start_index < context_start or start_index > context_end:
                        continue
                    if end_index < context_start or end_index > context_end:
                        continue
                    # Skip invalid spans
                    if end_index < start_index or end_index - start_index >= max_answer_len:
                        continue
                        
                    score = s_logits[start_index] + e_logits[end_index]
                    if score > best_score:
                        # Character offsets in the segmented context
                        start_char = int(offsets[start_index][0])
                        end_char = int(offsets[end_index][1])
                        # Slice from segmented context
                        best_span_segmented = segmented_context[start_char:end_char]
                        best_score = score
                        # Track the no_answer_score from THIS chunk (same chunk comparison)
                        best_chunk_no_answer_score = chunk_no_answer_score
                        
        # 4. Deciding if the question is answerable
        # Compare best span score against the no-answer score of the SAME chunk
        if best_score == -float("inf") or best_score < best_chunk_no_answer_score + no_answer_threshold:
            return {
                "answer": "",
                "score": float(max(best_score, best_chunk_no_answer_score)),
                "start": -1,
                "end": -1,
                "confidence": 0.0
            }
            
        # 5. Map segmented span back to original raw context
        raw_start, raw_end = find_char_span(raw_context, best_span_segmented)
        if raw_start == -1 or raw_end == -1:
            raw_answer = best_span_segmented.replace("_", " ").strip()
            raw_start, raw_end = -1, -1
        else:
            raw_answer = raw_context[raw_start:raw_end].strip()
            
        # Calculate normalized confidence score (0-1 range for UI display)
        confidence = float(torch.sigmoid(torch.tensor(best_score + 1.0)).item())
        
        return {
            "answer": raw_answer,
            "score": float(best_score),
            "start": raw_start,
            "end": raw_end,
            "confidence": round(confidence, 4)
        }

def main():
    parser = argparse.ArgumentParser(description="Predict answer span from context")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--question", type=str, required=True, help="Question text")
    parser.add_argument("--context", type=str, required=True, help="Context text")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU use")
    
    args = parser.parse_args()
    
    predictor = ReaderPredictor(args.model_path, use_cpu=args.use_cpu)
    result = predictor.predict(args.question, args.context)
    
    print("\n=== INFERENCE RESULT ===")
    print(f"Question: {args.question}")
    print(f"Predicted Answer: '{result['answer']}'")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Logits Score: {result['score']:.4f}")
    print(f"Original Char Span: [{result['start']}, {result['end']}]")
    print("========================\n")

if __name__ == "__main__":
    main()
