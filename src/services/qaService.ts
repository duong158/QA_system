import type { AskQuestionRequest, QaResponse, RetrieverComparisonRow } from '@/types/qa';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const USE_MOCK_API = String(import.meta.env.VITE_USE_MOCK_API ?? 'false').toLowerCase() === 'true';
const QA_ERROR_MESSAGE = 'Không thể xử lý câu hỏi do hệ thống QA gặp lỗi.';

function normalizeQaResponse(data: QaResponse): QaResponse {
  return {
    ...data,
    has_answer: data.has_answer ?? Boolean(data.answer),
    selected_passage_id: data.selected_passage_id ?? data.source?.passage_id ?? null,
    passages: data.passages ?? [],
  };
}

export async function askQuestion(request: AskQuestionRequest): Promise<QaResponse> {
  if (USE_MOCK_API) {
    const { askQuestionMock } = await import('@/services/mockQaService');
    return askQuestionMock(request);
  }

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
    throw new Error(QA_ERROR_MESSAGE);
  }

  const data = (await response.json()) as QaResponse;
  return normalizeQaResponse(data);
}

export async function compareRetrievers(question: string): Promise<RetrieverComparisonRow[]> {
  if (USE_MOCK_API) {
    const { compareRetrieversMock } = await import('@/services/mockQaService');
    return compareRetrieversMock(question);
  }

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
    throw new Error(QA_ERROR_MESSAGE);
  }

  return (await response.json()) as RetrieverComparisonRow[];
}
