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
  question_type?: QuestionType;
  reader_method?: string;
  reader_answer?: string | null;
  reader_span_answer?: string | null;
  reader_score?: number;
  neural_reader_answer?: string | null;
  neural_reader_best_span?: string | null;
  neural_reader_has_answer?: boolean;
  neural_reader_score?: number;
  neural_reader_confidence_is_calibrated?: boolean;
  neural_reader_start_score?: number | null;
  neural_reader_end_score?: number | null;
  reader_score_raw?: number | null;
  reader_null_score?: number | null;
  reader_no_answer_score?: number | null;
  reader_score_margin?: number | null;
  reader_decision_threshold?: number;
  fallback_answer?: string | null;
  fallback_sentence?: string | null;
  fallback_method?: string;
  fallback_phrase_score?: number;
  fallback_start?: number;
  fallback_end?: number;
  fallback_score?: number;
  fallback_reason?: string;
  reader_signal?: number;
  answer_type?: QuestionType;
  answer_type_score?: number;
  answer_type_reason?: string;
  lexical_evidence?: boolean;
  relation_evidence?: boolean;
  relation_type?: string | null;
  relation_score?: number;
  phrase_quality?: number;
  evidence_supported?: boolean;
  ranking_score?: number;
  answer_confidence?: number | null;
  selection_status?: 'SELECTED' | 'REJECTED';
  rejection_reason?: string | null;
  rejection_detail?: string | null;
  answer_span?: {
    text: string;
    start: number;
    end: number;
  };
}

export type QuestionType = 'TIME' | 'PERSON' | 'LOCATION' | 'NUMBER' | 'DEFINITION' | 'ENTITY' | 'GENERAL';

export interface QaResponse {
  question: string;
  question_type?: QuestionType;
  answer_type?: QuestionType;
  answer: string | null;
  has_answer?: boolean;
  confidence: number | null;
  answer_confidence?: number | null;
  reader_method?: string;
  fallback_method?: string | null;
  relation_type?: string | null;
  relation_score?: number;
  lexical_evidence?: boolean;
  relation_evidence?: boolean;
  selected_passage_id?: string | null;
  processing_time_ms: number;
  retriever: RetrieverType;
  reader: ReaderType;
  source: SourceInfo | null;
  answer_source?: SourceInfo | null;
  top_retrieved_passage?: PassageResult | null;
  no_answer_reason?: string | null;
  rejection_reason?: string | null;
  rejection_detail?: string | null;
  best_reader_score?: number;
  scores?: {
    retrieval: number | null;
    reader: number | null;
    answer_type: number | null;
    ranking: number | null;
    answer_confidence: number | null;
  };
  answer_span?: {
    text: string;
    start: number;
    end: number;
  };
  passages: PassageResult[];
  scoring?: {
    retriever_weight: number;
    reader_weight: number;
    answer_type_weight: number;
    minimum_reader_score: number;
    minimum_answer_type_score: number;
    minimum_fallback_answer_type_score: number;
    minimum_ranking_score: number;
    fallback_penalty: number;
    reader_score_margin_threshold?: number;
    reader_fallback_threshold?: number;
    sentence_fallback_threshold?: number;
    retrieval_normalization: string;
    candidate_count?: number;
    rerank?: string;
    ranking_score_formula?: string;
    score_semantics?: string;
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
