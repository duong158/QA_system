export type RetrieverType = 'tfidf' | 'bm25' | 'dense' | 'pyserini';
export type ReaderType = 'mock' | 'phobert' | 'vibert' | 'xlmr';

export interface AskQuestionRequest {
  question: string;
  retriever: RetrieverType;
  reader: ReaderType;
  top_k: number;
}

export interface SourceInfo {
  document_id: string;
  passage_id: string;
  title: string;
  page?: number;
}

export interface PassageResult {
  rank: number;
  document_id: string;
  passage_id: string;
  title: string;
  page?: number;
  text: string;
  retrieval_score: number;
  reader_score?: number;
}

export interface QaResponse {
  question: string;
  answer: string;
  confidence: number;
  processing_time_ms: number;
  retriever: RetrieverType;
  reader: ReaderType;
  source: SourceInfo;
  answer_span?: {
    text: string;
    start: number;
    end: number;
  };
  passages: PassageResult[];
}

export interface RetrieverComparisonRow {
  retriever: RetrieverType;
  label: string;
  correctPassageRank: number;
  recallAt1: boolean;
  recallAt3: boolean;
  responseTimeMs: number;
  topPassagePreview: string;
  retrievalScore: number;
}

export interface EvaluationMetric {
  label: string;
  value: number;
  comparison?: Record<string, number>;
}

export interface EvaluationErrorItem {
  issue: string;
  count: number;
  note: string;
}

export interface ReaderComparisonRow {
  reader: ReaderType;
  exactMatch: number;
  f1: number;
  avgLatencyMs: number;
}