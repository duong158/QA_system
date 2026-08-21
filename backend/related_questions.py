import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
import re

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data" / "raw"

class QuestionBM25Index:
    def __init__(self):
        self.questions = []
        self.bm25 = None
        self._load_and_index()

    def _tokenize(self, text: str) -> list[str]:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def _load_and_index(self):
        files = ["viquad2_train.parquet", "viquad2_validation.parquet", "viquad2_test.parquet"]
        questions_set = set()
        
        for fname in files:
            file_path = DATA_DIR / fname
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    if "question" in df.columns:
                        for q in df["question"].dropna():
                            q_str = str(q).strip()
                            if q_str:
                                questions_set.add(q_str)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        self.questions = list(questions_set)
        
        if not self.questions:
            print("[Warning] No questions found in datasets for BM25 indexing.")
            return

        tokenized_corpus = [self._tokenize(q) for q in self.questions]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[QuestionBM25Index] Indexed {len(self.questions)} unique questions.")

    def get_related_questions(self, query: str, asked_questions: list[str] = None, top_k: int = 3) -> list[dict]:
        if not self.bm25 or not query:
            return []
            
        tokenized_query = self._tokenize(query)
        # Get slightly more candidates so we can filter out exact matches and previously asked
        top_n = self.bm25.get_top_n(tokenized_query, self.questions, n=top_k + 10)
        
        results = []
        q_lower = query.strip().lower()
        asked_lower = {q.strip().lower() for q in (asked_questions or [])}
        
        for q in top_n:
            ql = q.strip().lower()
            if ql != q_lower and ql not in asked_lower:
                results.append({
                    "question": q,
                    "type": "RELATED"
                })
                if len(results) >= top_k:
                    break
                
        return results

_BM25_INDEX = None

def get_bm25_index():
    global _BM25_INDEX
    if _BM25_INDEX is None:
        print("[QuestionBM25Index] Initializing index...")
        _BM25_INDEX = QuestionBM25Index()
    return _BM25_INDEX
