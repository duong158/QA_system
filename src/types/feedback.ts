import type { PassageResult, QuestionType } from '@/types/qa';

export type FeedbackType =
  | 'CORRECT'
  | 'INCORRECT'
  | 'SPAN_CORRECTION'
  | 'NO_ANSWER_BUT_SHOULD_HAVE'
  | 'ANSWERED_BUT_SHOULD_NOT';

export type FeedbackStatus = 'PENDING' | 'REVIEWED' | 'APPROVED' | 'REJECTED';
export type GapType = 'CORPUS_GAP' | 'RETRIEVAL_GAP' | 'READER_SEMANTIC_GAP' | 'UNKNOWN_GAP';

export interface FeedbackSubmitRequest {
  question: string;
  predicted_answer: string | null;
  feedback_type: FeedbackType;
  question_type?: QuestionType | string | null;
  semantic_relation?: string | null;
  subject?: string | null;
  selected_passage_id?: string | null;
  corrected_passage_id?: string | null;
  corrected_answer?: string | null;
  corrected_start_char?: number | null;
  corrected_end_char?: number | null;
  retrieved_passage_ids: string[];
  rejection_reason?: string | null;
  user_note?: string | null;
  corpus_support_found?: boolean;
  source?: 'DIRECT_QUERY' | 'SOCRATIC_FOLLOWUP';
}

export interface FeedbackRecord extends FeedbackSubmitRequest {
  feedback_id: string;
  timestamp: string;
  status: FeedbackStatus;
  gap_type: GapType | null;
  model_version: string | null;
  corpus_version: string | null;
  semantic_policy_version: string | null;
  synthetic: boolean;
  conflict: boolean;
  duplicate_count: number;
  review_note?: string | null;
  source_passage?: PassageResult | null;
}

export interface FeedbackSubmitResponse {
  feedback: FeedbackRecord & { deduplicated?: boolean };
  message: string;
  runtime_model_updated: false;
}

export interface AnalyticsSummary {
  total_feedback: number;
  unique_records: number;
  correct: number;
  incorrect: number;
  correct_rate: number;
  incorrect_rate: number;
  no_answer_complaints: number;
  no_answer_complaint_rate: number;
  pending_review: number;
  real_feedback: number;
  synthetic_feedback: number;
}

export interface AnalyticsBucket {
  semantic_relation?: string;
  question_type?: string;
  total: number;
  correct: number;
  incorrect: number;
  no_answer: number;
  failure_rate: number;
  blind_spot_score: number;
}

export interface FeedbackAnalytics {
  summary: AnalyticsSummary;
  relations: AnalyticsBucket[];
  question_types: AnalyticsBucket[];
  gap_types: Array<{ gap_type: string; count: number }>;
  top_rejection_reasons: Array<{ reason: string; count: number }>;
  trend: Array<{ date: string; total: number; correct: number; incorrect: number; no_answer: number }>;
  heatmap: {
    dimensions: [string, string];
    question_types: string[];
    relations: string[];
    cells: Array<{
      question_type: string;
      relation: string;
      total: number;
      failures: number;
      failure_rate: number;
      blind_spot_score: number;
    }>;
  };
  methodology: {
    failure_definition: string;
    blind_spot_score: string;
    synthetic_is_separated: boolean;
  };
}

export interface DocumentSubmission {
  submission_id: string;
  title: string;
  content: string;
  timestamp: string;
  status: 'PENDING_REVIEW' | 'APPROVED' | 'REJECTED';
  review_note: string | null;
  source_type: 'PLAIN_TEXT' | 'TXT';
  synthetic: boolean;
}
