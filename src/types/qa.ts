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
  paragraph_id?: string;
  sentence_start?: number;
  sentence_end?: number;
}

export interface PassageResult {
  rank: number;
  retrieval_rank?: number;
  document_id: string;
  passage_id: string;
  title: string;
  page?: number;
  paragraph_id?: string;
  sentence_start?: number;
  sentence_end?: number;
  text: string;
  retrieval_score: number;
  retrieval_score_raw?: number;
  retrieval_score_normalized?: number;
  reader_answer?: string | null;
  reader_span_answer?: string | null;
  reader_score?: number;
  reader_score_raw?: number | null;
  reader_null_score?: number | null;
  reader_score_margin?: number | null;
  reader_margin_score?: number;
  final_score?: number;
  answer_span?: {
    text: string;
    start: number;
    end: number;
  };
}

export interface QaResponse {
  question: string;
  answer: string | null;
  has_answer?: boolean;
  confidence: number;
  selected_passage_id?: string | null;
  processing_time_ms: number;
  retriever: RetrieverType;
  reader: ReaderType;
  source: SourceInfo | null;
  answer_span?: {
    text: string;
    start: number;
    end: number;
  };
  passages: PassageResult[];
  scoring?: {
    retriever_weight: number;
    reader_weight: number;
    answer_threshold: number;
    retrieval_normalization: string;
    candidate_count?: number;
    rerank?: string;
    final_score_formula?: string;
  };
}

export interface RetrieverComparisonRow {
  retriever: RetrieverType;
  label: string;
  correctPassageRank: number | null;
  recallAt1: boolean | null;
  recallAt3: boolean | null;
  responseTimeMs: number;
  topPassagePreview: string;
  retrievalScore: number;
  evaluationNote?: string;
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
