import type {
  DocumentSubmission,
  FeedbackAnalytics,
  FeedbackRecord,
  FeedbackSubmitRequest,
  FeedbackSubmitResponse,
} from '@/types/feedback';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function apiError(response: Response): Promise<Error> {
  const body = await response.json().catch(() => null) as { error?: string } | null;
  return new Error(body?.error || `Feedback API failed (${response.status})`);
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    throw await apiError(response);
  }
  return await response.json() as T;
}

export function submitFeedback(payload: FeedbackSubmitRequest): Promise<FeedbackSubmitResponse> {
  return requestJson('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function fetchFeedbackAnalytics(): Promise<FeedbackAnalytics> {
  return requestJson('/api/feedback/analytics');
}

export async function fetchPendingFeedback(): Promise<FeedbackRecord[]> {
  const data = await requestJson<{ feedback: FeedbackRecord[] }>('/api/feedback/review?status=PENDING');
  return data.feedback ?? [];
}

export function reviewFeedback(
  feedbackId: string,
  decision: 'APPROVED' | 'REJECTED',
  reviewNote?: string,
): Promise<FeedbackRecord> {
  return requestJson(`/api/feedback/${encodeURIComponent(feedbackId)}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision, review_note: reviewNote }),
  });
}

export async function fetchDocumentSubmissions(): Promise<DocumentSubmission[]> {
  const data = await requestJson<{ submissions: DocumentSubmission[] }>('/api/documents/submissions');
  return data.submissions ?? [];
}

export async function submitDocument(payload: {
  title: string;
  content: string;
  source_type: 'PLAIN_TEXT' | 'TXT';
}): Promise<{ submission: DocumentSubmission; message: string; production_corpus_updated: false }> {
  return requestJson('/api/documents/submissions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function reviewDocument(
  submissionId: string,
  decision: 'APPROVED' | 'REJECTED',
  reviewNote?: string,
): Promise<DocumentSubmission> {
  return requestJson(`/api/documents/submissions/${encodeURIComponent(submissionId)}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision, review_note: reviewNote }),
  });
}
