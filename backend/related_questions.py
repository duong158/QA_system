import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
import re
import os
import threading
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data" / "raw"
DENSE_MODEL_NAME = os.getenv("QA_DENSE_MODEL", "keepitreal/vietnamese-sbert")

def rrf_fuse_questions(bm25_hits: list[tuple[int, float]], dense_hits: list[tuple[int, float]], top_k: int, k: int = 60) -> list[int]:
    bm25_ranks = {q_idx: rank for rank, (q_idx, _) in enumerate(bm25_hits, start=1)}
    dense_ranks = {q_idx: rank for rank, (q_idx, _) in enumerate(dense_hits, start=1)}
    
    all_keys = set(bm25_ranks.keys()) | set(dense_ranks.keys())
    missing_rank = max(len(bm25_hits), len(dense_hits)) + 1
    
    scored = []
    for key in all_keys:
        r_bm25 = bm25_ranks.get(key, missing_rank)
        r_dense = dense_ranks.get(key, missing_rank)
        rrf = 1.0 / (k + r_bm25) + 1.0 / (k + r_dense)
        scored.append((key, rrf))
        
    scored.sort(key=lambda x: x[1], reverse=True)
    return [q_idx for q_idx, _ in scored[:top_k]]

class QuestionHybridIndex:
    def __init__(self):
        self.questions = []
        self.bm25 = None
        self._model = None
        self._embeddings = None
        self._lock = threading.Lock()
        self._available = None
        self._load_and_index()

    def _tokenize(self, text: str) -> list[str]:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def inject_dense_model(self, model):
        with self._lock:
            if self._model is None and model is not None:
                self._model = model
                self._available = True
                self._ensure_dense_embeddings()

    def _ensure_dense_embeddings(self):
        safe_name = DENSE_MODEL_NAME.replace("/", "_")
        cache_file = ROOT / "data" / "processed" / f"question_embeddings_{safe_name}.npy"
        if cache_file.exists():
            try:
                cached = np.load(str(cache_file))
                if cached.shape[0] == len(self.questions):
                    self._embeddings = cached
                    print(f"[QuestionHybridIndex] Loaded cached embeddings, shape={self._embeddings.shape}")
                else:
                    self._embeddings = None
            except Exception:
                pass
        
        if self._embeddings is None and self.questions:
            print(f"[QuestionHybridIndex] Encoding {len(self.questions)} questions...")
            self._embeddings = self._model.encode(
                self.questions,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)
            try:
                os.makedirs(cache_file.parent, exist_ok=True)
                np.save(str(cache_file), self._embeddings)
            except Exception:
                pass

    def _ensure_dense_model(self):
        if self._available is False: return False
        if self._model is not None:
            if self._embeddings is None:
                with self._lock:
                    if self._embeddings is None:
                        self._ensure_dense_embeddings()
            return True
        with self._lock:
            if self._model is not None: return True
            if self._available is False: return False
            try:
                print(f"[QuestionHybridIndex] Loading dense model: {DENSE_MODEL_NAME}")
                self._model = SentenceTransformer(DENSE_MODEL_NAME)
                self._available = True
                self._ensure_dense_embeddings()
                return True
            except Exception as e:
                print(f"[QuestionHybridIndex] Dense model failed: {e}")
                self._available = False
                return False

    def _load_and_index(self):
        files = ["viquad2_train.parquet", "viquad2_validation.parquet", "viquad2_test.parquet"]
        questions_set = set()
        
        for fname in files:
            file_path = DATA_DIR / fname
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    if "question" in df.columns:
                        valid_df = df.copy()
                        
                        # 1. Lọc bỏ những câu hỏi đánh dấu is_impossible == True
                        if "is_impossible" in valid_df.columns:
                            valid_df = valid_df[valid_df["is_impossible"].fillna(False) == False]
                            
                        # 2. Lọc bỏ những câu hỏi mà cả answers và plausible_answers đều trống rỗng
                        def _is_empty(val):
                            if val is None: return True
                            if isinstance(val, dict): return len(val.get('text', [])) == 0
                            if hasattr(val, '__len__'): return len(val) == 0
                            try: return pd.isna(val)
                            except: return False
                            
                        if "answers" in valid_df.columns and "plausible_answers" in valid_df.columns:
                            empty_ans = valid_df["answers"].apply(_is_empty)
                            empty_plaus = valid_df["plausible_answers"].apply(_is_empty)
                            valid_df = valid_df[~(empty_ans & empty_plaus)]
                            
                        for q in valid_df["question"].dropna():
                            q_str = str(q).strip()
                            if q_str:
                                questions_set.add(q_str)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        self.questions = list(questions_set)
        
        if not self.questions:
            print("[Warning] No questions found in datasets for Hybrid indexing.")
            return

        tokenized_corpus = [self._tokenize(q) for q in self.questions]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[QuestionHybridIndex] Indexed {len(self.questions)} unique questions.")

    def get_related_questions(self, query: str, asked_questions: list[str] = None, top_k: int = 3) -> list[dict]:
        if not self.bm25 or not query:
            return []
            
        tokenized_query = self._tokenize(query)
        top_n_candidates = top_k + 10
        
        # BM25 scores
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_hits = [(i, score) for i, score in enumerate(bm25_scores) if score > 0]
        bm25_hits.sort(key=lambda x: x[1], reverse=True)
        bm25_top = bm25_hits[:top_n_candidates]
        
        # Dense scores
        dense_top = []
        if self._ensure_dense_model():
            query_vec = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            similarities = (self._embeddings @ query_vec.T).flatten()
            dense_hits = [(i, float(sim)) for i, sim in enumerate(similarities) if sim > 0]
            dense_hits.sort(key=lambda x: x[1], reverse=True)
            dense_top = dense_hits[:top_n_candidates]
            
        if not dense_top:
            fused_indices = [idx for idx, _ in bm25_top]
        else:
            fused_indices = rrf_fuse_questions(bm25_top, dense_top, top_n_candidates)
            
        results = []
        q_lower = query.strip().lower()
        asked_lower = {q.strip().lower() for q in (asked_questions or [])}
        
        for idx in fused_indices:
            q = self.questions[idx]
            ql = q.strip().lower()
            if ql != q_lower and ql not in asked_lower:
                results.append({
                    "question": q,
                    "type": "RELATED"
                })
                if len(results) >= top_k:
                    break
                
        return results

_HYBRID_INDEX = None

def get_hybrid_index():
    global _HYBRID_INDEX
    if _HYBRID_INDEX is None:
        print("[QuestionHybridIndex] Initializing index...")
        _HYBRID_INDEX = QuestionHybridIndex()
    return _HYBRID_INDEX
