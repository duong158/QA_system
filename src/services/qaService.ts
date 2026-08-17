import type { AskQuestionRequest, QaResponse, RetrieverComparisonRow } from '@/types/qa';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const QA_ERROR_MESSAGE = 'Không thể xử lý câu hỏi do hệ thống QA gặp lỗi.';

async function readApiError(response: Response): Promise<string> {
  const errorBody = await response.json().catch(() => null) as { error?: string } | null;
  return errorBody?.error || QA_ERROR_MESSAGE;
}

function normalizeQaResponse(data: QaResponse): QaResponse {
  return {
    ...data,
    confidence: data.confidence ?? null,
    answer_confidence: data.answer_confidence ?? null,
    has_answer: data.has_answer ?? Boolean(data.answer),
    selected_passage_id: data.selected_passage_id ?? data.answer_source?.passage_id ?? data.source?.passage_id ?? null,
    source: data.answer_source ?? data.source ?? null,
    passages: data.passages ?? [],
  };
}

export async function askQuestion(request: AskQuestionRequest): Promise<QaResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
  } catch {
    throw new Error(QA_ERROR_MESSAGE);
  }

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  const data = (await response.json()) as QaResponse;
  return normalizeQaResponse(data);
}

export async function compareRetrievers(question: string): Promise<RetrieverComparisonRow[]> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new Error(QA_ERROR_MESSAGE);
  }

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  return (await response.json()) as RetrieverComparisonRow[];
}
