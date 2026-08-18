# VIQA Nexus

Vietnamese Intelligent Question Answering frontend built with React, TypeScript, Vite, Tailwind CSS, React Three Fiber, Three.js, Framer Motion, Lucide React, Recharts, Zustand, and Web Speech API.

## Run Locally

```bash
npm install
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`.

## Run With Local DrQA-Style API

Terminal 1:

```bash
npm run api
```

Terminal 2:

```bash
VITE_USE_MOCK_API=false npm run dev
```

On Windows PowerShell:

```powershell
$env:VITE_USE_MOCK_API="false"; npm run dev
```

The local API is implemented in `backend/viqa_api.py` and serves:

- `GET /health`
- `POST /api/ask`
- `POST /api/compare`

It reads `data/processed/docs.db`, chunks each document at sentence boundaries, and retrieves passage-level top-k results with TF-IDF or BM25. Every retrieved passage is sent to the configured extractive QA Reader before score-based reranking.

The real API never fabricates mock answers or scores. It uses the local PhoBERT QA checkpoint when available. A transparent sentence-extraction fallback may propose a candidate, but it receives a ranking penalty and must pass answer-type and evidence gates. Dense/Pyserini and Reader choices without a compatible local model/index return an explicit API error.

The single checked-in production configuration is `config/qa_pipeline.json`.
Environment variables may override it for controlled experiments:

```text
QA_CHUNK_MAX_TOKENS=220
QA_CHUNK_OVERLAP_SENTENCES=2
QA_RETRIEVER_WEIGHT=0.40
QA_READER_WEIGHT=0.40
QA_ANSWER_TYPE_WEIGHT=0.20
QA_TOP_K=10
QA_RETRIEVER_MIN_CANDIDATES=20
QA_READER_SCORE_MARGIN_THRESHOLD=<validation-calibrated margin>
QA_MIN_READER_SCORE=0.30
QA_MIN_ANSWER_TYPE_SCORE=0.50
QA_MIN_FALLBACK_ANSWER_TYPE_SCORE=0.75
QA_MIN_RANKING_SCORE=0.55
QA_FALLBACK_PENALTY=0.60
QA_READER_MAX_LENGTH=256
QA_READER_STRIDE=80
QA_TOP_N_START=20
QA_TOP_N_END=20
QA_MAX_ANSWER_LENGTH=40
QA_DEBUG=false
```

For example, enable technical pipeline logs in PowerShell with:

```powershell
$env:QA_DEBUG="true"; npm run api
```

Retriever and QA evaluation commands:

```bash
python evaluate_retriever.py path/to/eval.jsonl --method bm25 --k 1 3 5 10
python -m reader.evaluate --model_path models/reader/vinai_phobert-base-v2
python evaluate_qa.py --validation --mode oracle
python evaluate_qa.py --validation --mode end-to-end --retriever bm25 --top-k 10
python evaluate_reranking.py --subset-size 100 --top-k 10
```

Production distinguishes four score concepts: normalized retrieval score,
uncalibrated Reader score, candidate ranking score, and answer confidence.
Retrieval and ranking scores are never shown as correctness percentages.
`answer_confidence` remains `null` until a separate correctness estimator is
calibrated on validation data.

Reader EM/F1 and threshold selection use the complete 3,814-question labeled
validation split. The 7,301-question test parquet has no usable gold answers and
is never used for Reader EM/F1. The evaluator reports overall, answerable, and
unanswerable metrics separately and writes threshold/error-analysis artifacts.

Before training, verify all annotated spans:

```bash
python -m reader.validate_spans --splits train validation
```

Train the audited CE baseline (one run per LR/epoch combination):

```bash
python -m reader.train --lr 2e-5 --epochs 2 --max_seq_len 256 --doc_stride 80
python -m reader.train --lr 2e-5 --epochs 3 --max_seq_len 256 --doc_stride 80
python -m reader.train --lr 3e-5 --epochs 2 --max_seq_len 256 --doc_stride 80
python -m reader.train --lr 3e-5 --epochs 3 --max_seq_len 256 --doc_stride 80
```

Each run selects checkpoints with decoded validation `answerable_f1`, uses
standard start/end cross entropy, and calibrates the raw
`best_span_score - CLS/null_score` margin, and writes artifacts under
`results/reader/<run_name>/`. Add `--promote` only after a full validation run
has completed successfully.

Threshold selection prioritizes Reader quality while retaining no-answer
behavior: `0.7 * answerable_f1 + 0.3 * unanswerable_accuracy`.

The optional standalone retrieval package remains available for offline experiments:

- `retrieval.tfidf_retriever.TfidfRetriever`
- `retrieval.bm25_retriever.BM25Retriever`
- `retrieval.dense_retriever.DenseRetriever`
- `retrieval.pyserini_retriever.PyseriniRetriever`

Install sparse retriever dependencies:

```bash
pip install -r requirements-retrieval.txt
```

Build indexes from the processed corpus:

```bash
python retrieval/build_index.py --method sparse
```

Dense and Pyserini are optional because they require heavier dependencies and, for Pyserini, Java. They must have passage-level indexes compatible with the current chunked corpus before being enabled in the real API.

## Environment

Copy `.env.example` to `.env` when you need local overrides.

```text
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCK_API=false
VITE_QA_DEBUG=false
VITE_AVATAR_MODEL_URL=/models/mari.vrm
VITE_AVATAR_MODEL_NAME=Mari 3D VRoid Model
VITE_AVATAR_CREATOR_NAME=wondrous21
VITE_AVATAR_LICENSE=Free to use with credit
```

When `VITE_USE_MOCK_API=true`, the app uses `src/services/mockQaService.ts`.
When `VITE_USE_MOCK_API=false`, the app sends:

```text
POST ${VITE_API_BASE_URL}/api/ask
```

## Features

- Cinematic 3D AI assistant screen with avatar states: idle, listening, retrieving, reading, thinking, speaking, success, no-answer, and error.
- Vietnamese speech recognition (`vi-VN`) and speech synthesis with browser fallbacks.
- Retriever -> Reader -> Answer pipeline animation.
- Mock QA service with simulated latency and top-k passages.
- Local DrQA-style API backed by `data/processed/docs.db` for real retrieval over the processed corpus.
- Answer panel with explicit ranking-signal semantics, source metadata, rejection reasons, and highlighted evidence.
- Retriever comparison mode for TF-IDF, BM25, and Dense Retrieval.
- `/evaluation` route with clearly labeled Mock / Demo metrics; these values are not model benchmarks.
- Settings panel persisted to localStorage.

## Avatar Model

The avatar is loaded from a local VRM file using `GLTFLoader` and `VRMLoaderPlugin`.

The Mari VRM model is not included in this repository. Download it from the original creator and place the file at:

```text
public/models/mari.vrm
```

Do not redistribute or upload the model file to a public repository unless the creator's license explicitly allows it. The currently configured source is the free BOOTH item by `wondrous21`: https://booth.pm/en/items/4507087

## Reader Model

The PhoBERT QA configuration and model-compatible tokenizer artifacts
(`vocab.txt` and `bpe.codes`) are included at:

```text
models/reader/vinai_phobert-base-v2/
```

The trained `model.safetensors` file is about 513 MB and is intentionally excluded from Git. Transfer that checkpoint separately and place it at:

```text
models/reader/vinai_phobert-base-v2/model.safetensors
```

Without the large checkpoint, the API still starts and uses its sentence-extraction fallback. The `/health` response reports which Reader models are available.

Training, validation, and production all use the same path:
raw Vietnamese text → PyVi exactly once → original PhoBERT BPE. The QA offset
adapter maps BPE pieces back to the segmented context without using first-string
search, so repeated answers remain tied to their annotated `answer_start`.

## Build

```bash
npm run build
```
