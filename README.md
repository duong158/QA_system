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

The real API never fabricates mock answers or scores. It uses the local PhoBERT QA checkpoint when available and a transparent sentence-extraction fallback when the neural score is too low or the large checkpoint is not installed. Dense/Pyserini and Reader choices without a compatible local model/index return an explicit API error.

Pipeline settings are environment variables on the API process:

```text
QA_CHUNK_MAX_TOKENS=220
QA_CHUNK_OVERLAP_SENTENCES=2
QA_RETRIEVER_WEIGHT=0.30
QA_READER_WEIGHT=0.70
QA_ANSWER_THRESHOLD=0.30
QA_READER_MAX_LENGTH=384
QA_READER_STRIDE=128
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
python evaluate_qa.py path/to/eval.jsonl --mode oracle
python evaluate_qa.py path/to/eval.jsonl --mode end-to-end --retriever bm25 --top-k 5
```

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
- Answer panel with confidence, source metadata, passage cards, and highlighted evidence.
- Retriever comparison mode for TF-IDF, BM25, and Dense Retrieval.
- `/evaluation` route with mock metrics, charts, reader comparison, and error analysis.
- Settings panel persisted to localStorage.

## Avatar Model

The avatar is loaded from a local VRM file using `GLTFLoader` and `VRMLoaderPlugin`.

The Mari VRM model is not included in this repository. Download it from the original creator and place the file at:

```text
public/models/mari.vrm
```

Do not redistribute or upload the model file to a public repository unless the creator's license explicitly allows it. The currently configured source is the free BOOTH item by `wondrous21`: https://booth.pm/en/items/4507087

## Reader Model

The lightweight PhoBERT QA configuration and tokenizer files are included at:

```text
models/reader/vinai_phobert-base-v2/
```

The trained `model.safetensors` file is about 513 MB and is intentionally excluded from Git. Transfer that checkpoint separately and place it at:

```text
models/reader/vinai_phobert-base-v2/model.safetensors
```

Without the large checkpoint, the API still starts and uses its sentence-extraction fallback. The `/health` response reports which Reader models are available.

## Build

```bash
npm run build
```
