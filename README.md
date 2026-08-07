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

It reads `data/processed/docs.db` and provides lightweight TF-IDF, BM25, Pyserini-BM25 adapter, and dense-lite retrieval for UI testing.
The API adapter prefers the pulled `retrieval/` package:

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

Dense and Pyserini are optional because they require heavier dependencies and, for Pyserini, Java. If their dependency/index is missing, the local API falls back to a lightweight equivalent so the UI remains testable.

## Environment

Copy `.env.example` to `.env` when you need local overrides.

```text
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCK_API=true
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

## Build

```bash
npm run build
```
